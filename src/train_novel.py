"""
PII-DeBERTa: novel fine-tuning with three contributions.

  1. Source conditioning  -- prepend [SRC=<dataset>] token to every sequence
  2. Curriculum learning  -- three-phase training schedule (general NER ->
                             synthetic PII -> financial PII)
  3. Hierarchical head    -- coarse classification (10 groups) conditions
                             fine-grained BIO prediction (97 labels)

All three features are controlled by flags and can be combined independently.

Usage (standard baseline, no novelty):
    python src/train_novel.py \
        --splits-dir   data/splits \
        --output-dir   models/flat_baseline

Usage (source conditioning + curriculum + hierarchical head):
    python src/train_novel.py \
        --splits-dir      data/splits \
        --output-dir      models/full_novel \
        --source-cond \
        --curriculum \
        --hierarchical \
        --coarse-loss-weight 0.3

GCP V100 recommended settings:
    python src/train_novel.py \
        --splits-dir          data/splits \
        --output-dir          models/full_novel \
        --source-cond \
        --curriculum \
        --hierarchical \
        --coarse-loss-weight  0.3 \
        --batch-size          8 \
        --grad-accum          8 \
        --max-length          256 \
        --bf16 \
        --gradient-checkpointing
"""

import json
import argparse
import warnings
import math
from pathlib import Path
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    AutoConfig,
    PreTrainedModel,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from transformers.modeling_outputs import TokenClassifierOutput
from seqeval.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Coarse taxonomy
# ---------------------------------------------------------------------------

# Maps canonical entity type (without B-/I-) -> coarse group name
COARSE_GROUPS = {
    "PERSON":              "PERSON_GROUP",
    "PERSON_NAME":         "PERSON_GROUP",
    "PREFIX":              "PERSON_GROUP",
    "SUFFIX":              "PERSON_GROUP",
    "GENDER":              "PERSON_GROUP",
    "EMAIL":               "CONTACT",
    "PHONE":               "CONTACT",
    "PHONE_NUMBER":        "CONTACT",
    "FAX":                 "CONTACT",
    "CREDIT_CARD":         "FINANCIAL_ID",
    "CREDIT_CARD_NUMBER":  "FINANCIAL_ID",
    "ACCOUNT_NUMBER":      "FINANCIAL_ID",
    "IBAN":                "FINANCIAL_ID",
    "BBAN":                "FINANCIAL_ID",
    "BIC":                 "FINANCIAL_ID",
    "ROUTING_NUMBER":      "FINANCIAL_ID",
    "SWIFT_BIC_CODE":      "FINANCIAL_ID",
    "DATE":                "TEMPORAL",
    "TIME":                "TEMPORAL",
    "DOB":                 "TEMPORAL",
    "DATE_OF_BIRTH":       "TEMPORAL",
    "PASSWORD":            "CREDENTIAL",
    "ACCOUNT_PIN":         "CREDENTIAL",
    "PIN":                 "CREDENTIAL",
    "SSN":                 "CREDENTIAL",
    "TAX_ID":              "CREDENTIAL",
    "NATIONAL_ID":         "CREDENTIAL",
    "IP_ADDRESS":          "NETWORK",
    "MAC_ADDRESS":         "NETWORK",
    "URL":                 "NETWORK",
    "USERNAME":            "NETWORK",
    "USER_NAME":           "NETWORK",
    "ORGANIZATION":        "ORG_ROLE",
    "COMPANY":             "ORG_ROLE",
    "JOB_TITLE":           "ORG_ROLE",
    "JOB":                 "ORG_ROLE",
    "LOCATION":            "LOCATION",
    "ADDRESS":             "LOCATION",
    "STREET_ADDRESS":      "LOCATION",
    "CITY":                "LOCATION",
    "STATE":               "LOCATION",
    "COUNTRY":             "LOCATION",
    "ZIP_CODE":            "LOCATION",
    "VEHICLE":             "MISC",
    "AMOUNT":              "MISC",
    "CURRENCY":            "MISC",
    "CRYPTO_ADDRESS":      "MISC",
    "CREDIT_CARD_SECURITY_CODE": "MISC",
    "BLOOD_TYPE":          "MISC",
    "FINANCIAL_ENTITY":    "FINANCIAL_NER",
}

COARSE_NAMES = [
    "O",
    "B-PERSON_GROUP", "I-PERSON_GROUP",
    "B-CONTACT",       "I-CONTACT",
    "B-FINANCIAL_ID",  "I-FINANCIAL_ID",
    "B-TEMPORAL",      "I-TEMPORAL",
    "B-CREDENTIAL",    "I-CREDENTIAL",
    "B-NETWORK",       "I-NETWORK",
    "B-ORG_ROLE",      "I-ORG_ROLE",
    "B-LOCATION",      "I-LOCATION",
    "B-MISC",          "I-MISC",
    "B-FINANCIAL_NER", "I-FINANCIAL_NER",
]

COARSE2ID = {lbl: i for i, lbl in enumerate(COARSE_NAMES)}
ID2COARSE = {i: lbl for i, lbl in enumerate(COARSE_NAMES)}


def fine_to_coarse(fine_label: str) -> str:
    if fine_label == "O":
        return "O"
    prefix = fine_label[:2]   # "B-" or "I-"
    etype  = fine_label[2:]
    group  = COARSE_GROUPS.get(etype, "MISC")
    return f"{prefix}{group}"


# ---------------------------------------------------------------------------
# Source tokens
# ---------------------------------------------------------------------------

SOURCE_TOKENS = {
    "ai4privacy_400k":     "[SRC=ai4privacy_400k]",
    "ai4privacy_300k":     "[SRC=ai4privacy_300k]",
    "isotonic_pii_200k":   "[SRC=isotonic_pii_200k]",
    "nvidia_nemotron":     "[SRC=nvidia_nemotron]",
    "wikiann":             "[SRC=wikiann]",
    "multinerd":           "[SRC=multinerd]",
    "few_nerd":            "[SRC=few_nerd]",
    "conll2003":           "[SRC=conll2003]",
    "finer_139":           "[SRC=finer_139]",
    "gretel_finance":      "[SRC=gretel_finance]",
}
DEFAULT_SOURCE_TOKEN = "[SRC=general]"

ALL_SOURCE_TOKENS = list(SOURCE_TOKENS.values()) + [DEFAULT_SOURCE_TOKEN]

# Curriculum phase -> list of source names included in that phase
CURRICULUM_PHASES = [
    # Phase 1: general NER
    ["wikiann", "multinerd", "few_nerd", "conll2003"],
    # Phase 2: synthetic PII
    ["ai4privacy_400k", "ai4privacy_300k", "isotonic_pii_200k", "nvidia_nemotron"],
    # Phase 3: financial PII
    ["gretel_finance", "finer_139"],
]


# ---------------------------------------------------------------------------
# Dataset: streaming with optional source conditioning
# ---------------------------------------------------------------------------

class PIIDataset(IterableDataset):
    """
    Streams a JSONL split, optionally:
      - filtering to a subset of source datasets (curriculum phases)
      - prepending a source conditioning token
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        label2id: dict,
        max_length: int,
        source_cond: bool = False,
        allowed_sources: list = None,   # None = all sources
        coarse2id: dict = None,         # if set, also produce coarse_labels
    ):
        super().__init__()
        self.jsonl_path     = jsonl_path
        self.tokenizer      = tokenizer
        self.label2id       = label2id
        self.max_length     = max_length
        self.source_cond    = source_cond
        self.allowed_sources = set(allowed_sources) if allowed_sources else None
        self.coarse2id      = coarse2id

    def __iter__(self):
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                src = rec.get("source", "")
                if self.allowed_sources and src not in self.allowed_sources:
                    continue
                yield self._encode(rec["tokens"], rec["labels"], src)

    def _encode(self, tokens: list, labels: list, source: str) -> dict:
        src_tok = SOURCE_TOKENS.get(source, DEFAULT_SOURCE_TOKEN)

        if self.source_cond:
            # Prepend source token as a single-word prefix.
            # It gets its own word_id=0; all original tokens shift by 1.
            tokens = [src_tok] + tokens
            labels = ["O"] + labels  # source token is always O

        enc = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        word_ids       = enc.word_ids()
        aligned_fine   = []
        aligned_coarse = [] if self.coarse2id is not None else None
        prev_word_idx  = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_fine.append(-100)
                if aligned_coarse is not None:
                    aligned_coarse.append(-100)
            elif word_idx != prev_word_idx:
                raw = labels[word_idx] if word_idx < len(labels) else "O"
                fine_id = self.label2id.get(raw, self.label2id["O"])
                aligned_fine.append(fine_id)
                if aligned_coarse is not None:
                    coarse_lbl = fine_to_coarse(raw)
                    aligned_coarse.append(self.coarse2id.get(coarse_lbl, self.coarse2id["O"]))
            else:
                aligned_fine.append(-100)
                if aligned_coarse is not None:
                    aligned_coarse.append(-100)
            prev_word_idx = word_idx

        enc["labels"] = aligned_fine
        if aligned_coarse is not None:
            enc["coarse_labels"] = aligned_coarse

        return {k: torch.tensor(v) for k, v in enc.items()}


# ---------------------------------------------------------------------------
# Hierarchical classification head
# ---------------------------------------------------------------------------

class HierarchicalPIIModel(PreTrainedModel):
    """
    DeBERTa-v3-base with a two-stage classification head:

      Stage 1 (coarse): hidden (768) -> coarse_logits (21)
      Stage 2 (fine):   concat(hidden, softmax(coarse_logits)) (789) -> fine_logits (97)

    Loss: L = L_fine + coarse_weight * L_coarse
    """

    def __init__(self, config, num_fine_labels: int, num_coarse_labels: int, coarse_weight: float = 0.3):
        super().__init__(config)
        from transformers import DebertaV2Model
        self.deberta      = DebertaV2Model(config)
        self.dropout      = nn.Dropout(config.hidden_dropout_prob if hasattr(config, "hidden_dropout_prob") else 0.1)
        self.coarse_head  = nn.Linear(config.hidden_size, num_coarse_labels)
        # fine head takes hidden + softmax(coarse) as input
        self.fine_head    = nn.Linear(config.hidden_size + num_coarse_labels, num_fine_labels)
        self.num_fine_labels   = num_fine_labels
        self.num_coarse_labels = num_coarse_labels
        self.coarse_weight     = coarse_weight
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        inputs_embeds=None,
        labels=None,
        coarse_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        outputs = self.deberta(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
        )
        hidden = self.dropout(outputs.last_hidden_state)      # (B, T, 768)

        coarse_logits = self.coarse_head(hidden)               # (B, T, 21)
        coarse_probs  = torch.softmax(coarse_logits, dim=-1)   # (B, T, 21)

        fine_input  = torch.cat([hidden, coarse_probs], dim=-1)  # (B, T, 789)
        fine_logits = self.fine_head(fine_input)                  # (B, T, 97)

        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss()
            fine_loss = loss_fct(
                fine_logits.view(-1, self.num_fine_labels),
                labels.view(-1),
            )
            if coarse_labels is not None:
                coarse_loss = loss_fct(
                    coarse_logits.view(-1, self.num_coarse_labels),
                    coarse_labels.view(-1),
                )
                loss = fine_loss + self.coarse_weight * coarse_loss
            else:
                loss = fine_loss

        return TokenClassifierOutput(
            loss=loss,
            logits=fine_logits,           # Trainer reads .logits for metrics
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )


# ---------------------------------------------------------------------------
# Custom data collator (handles optional coarse_labels field)
# ---------------------------------------------------------------------------

class PIIDataCollator(DataCollatorForTokenClassification):
    """
    Extends DataCollatorForTokenClassification to also pad coarse_labels
    when present in the batch.
    """

    def __call__(self, features):
        has_coarse = "coarse_labels" in features[0]
        if has_coarse:
            coarse_labels_list = [f.pop("coarse_labels") for f in features]

        batch = super().__call__(features)

        if has_coarse:
            max_len = batch["input_ids"].shape[1]
            padded  = []
            for cl in coarse_labels_list:
                pad_len = max_len - len(cl)
                padded.append(
                    torch.cat([cl, torch.full((pad_len,), -100, dtype=torch.long)])
                )
            batch["coarse_labels"] = torch.stack(padded)
            # Restore for next calls
            for f, cl in zip(features, coarse_labels_list):
                f["coarse_labels"] = cl

        return batch


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(id2label: dict):
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
        # logits may be tuple (fine_logits, coarse_logits) or just fine_logits
        if isinstance(logits, tuple):
            logits = logits[0]
        predictions = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, label_ids):
            seq_labels, seq_preds = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                seq_labels.append(id2label[int(l)])
                seq_preds.append(id2label[int(p)])
            true_labels.append(seq_labels)
            true_preds.append(seq_preds)

        return {
            "f1":        f1_score(true_labels, true_preds),
            "precision": precision_score(true_labels, true_preds),
            "recall":    recall_score(true_labels, true_preds),
        }
    return compute_metrics


# ---------------------------------------------------------------------------
# Curriculum training
# ---------------------------------------------------------------------------

def count_jsonl_lines_filtered(path: Path, allowed_sources: set) -> int:
    count = 0
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if rec.get("source", "") in allowed_sources:
                count += 1
    return count


def run_curriculum_phase(
    phase_idx: int,
    allowed_sources: list,
    lr: float,
    trainer_kwargs: dict,
    train_path: Path,
    val_dataset,
    tokenizer,
    label2id: dict,
    id2label: dict,
    max_length: int,
    batch_size: int,
    grad_accum: int,
    source_cond: bool,
    hierarchical: bool,
    coarse_weight: float,
    output_dir: Path,
    bf16: bool,
    gradient_checkpointing: bool,
    model=None,
):
    print(f"\n{'='*60}")
    print(f"CURRICULUM PHASE {phase_idx + 1}: sources = {allowed_sources}")
    print(f"  Learning rate: {lr}")
    print(f"{'='*60}")

    coarse2id = COARSE2ID if hierarchical else None

    train_ds = PIIDataset(
        jsonl_path=train_path,
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=max_length,
        source_cond=source_cond,
        allowed_sources=allowed_sources,
        coarse2id=coarse2id,
    )

    # Count records in this phase for max_steps calculation
    allowed_set = set(allowed_sources)
    n_train = count_jsonl_lines_filtered(train_path, allowed_set)
    steps_per_epoch = math.ceil(n_train / (batch_size * grad_accum))
    max_steps = steps_per_epoch  # one epoch per phase
    warmup_steps = max(1, int(0.05 * max_steps))

    phase_output = output_dir / f"phase_{phase_idx + 1}"
    phase_output.mkdir(parents=True, exist_ok=True)

    args = TrainingArguments(
        output_dir=str(phase_output),
        max_steps=max_steps,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        warmup_steps=warmup_steps,
        lr_scheduler_type="linear",
        logging_steps=max(1, steps_per_epoch // 20),
        eval_strategy="steps",
        eval_steps=max(1, steps_per_epoch // 4),
        save_strategy="steps",
        save_steps=max(1, steps_per_epoch // 4),
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
        bf16=bf16,
        fp16=False,
        gradient_checkpointing=gradient_checkpointing,
        dataloader_num_workers=2,
        report_to="none",
        label_names=["labels"],
    )

    collator = PIIDataCollator(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_ds,
        eval_dataset=val_dataset,
        data_collator=collator,
        compute_metrics=make_compute_metrics(id2label),
    )

    trainer.train()
    return trainer.model


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def main(args):
    splits_dir = Path(args.splits_dir)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load label mapping
    label_map_path = splits_dir / "label_mapping.json"
    with open(label_map_path) as f:
        label_map = json.load(f)

    label2id = label_map["label2id"]
    id2label = {int(k): v for k, v in label_map["id2label"].items()}
    num_labels = label_map["num_labels"]

    print(f"Labels: {num_labels} total BIO labels")
    print(f"Source conditioning: {args.source_cond}")
    print(f"Curriculum learning: {args.curriculum}")
    print(f"Hierarchical head:   {args.hierarchical}")

    # Load tokenizer
    model_id = args.model_id
    tokenizer = AutoTokenizer.from_pretrained(model_id, use_fast=True)

    # Add source tokens if needed
    if args.source_cond:
        n_added = tokenizer.add_tokens(ALL_SOURCE_TOKENS, special_tokens=True)
        print(f"Added {n_added} source tokens to tokenizer vocabulary")

    # Build model
    coarse_weight = args.coarse_loss_weight
    if args.hierarchical:
        config = AutoConfig.from_pretrained(
            model_id,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
        )
        model = HierarchicalPIIModel(
            config=config,
            num_fine_labels=num_labels,
            num_coarse_labels=len(COARSE_NAMES),
            coarse_weight=coarse_weight,
        )
        # Load DeBERTa weights into .deberta sub-module
        from transformers import DebertaV2Model
        pretrained_deberta = DebertaV2Model.from_pretrained(model_id)
        model.deberta.load_state_dict(pretrained_deberta.state_dict(), strict=False)
        del pretrained_deberta
    else:
        from transformers import AutoModelForTokenClassification
        model = AutoModelForTokenClassification.from_pretrained(
            model_id,
            num_labels=num_labels,
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

    if args.source_cond:
        model.resize_token_embeddings(len(tokenizer))

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    # Prepare validation dataset (val_1p for fast intra-training eval)
    coarse2id = COARSE2ID if args.hierarchical else None
    val_ds = PIIDataset(
        jsonl_path=splits_dir / "val_1p.jsonl",
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        source_cond=args.source_cond,
        allowed_sources=None,
        coarse2id=coarse2id,
    )
    # Materialise val_1p into a list (small enough at 1,398 records)
    val_list = list(val_ds)

    train_path = splits_dir / "train.jsonl"

    if args.curriculum:
        # Phase learning rates: 2e-5 -> 1e-5 -> 5e-6
        phase_lrs = [2e-5, 1e-5, 5e-6]
        for phase_idx, (sources, lr) in enumerate(zip(CURRICULUM_PHASES, phase_lrs)):
            model = run_curriculum_phase(
                phase_idx=phase_idx,
                allowed_sources=sources,
                lr=lr,
                trainer_kwargs={},
                train_path=train_path,
                val_dataset=val_list,
                tokenizer=tokenizer,
                label2id=label2id,
                id2label=id2label,
                max_length=args.max_length,
                batch_size=args.batch_size,
                grad_accum=args.grad_accum,
                source_cond=args.source_cond,
                hierarchical=args.hierarchical,
                coarse_weight=coarse_weight,
                output_dir=output_dir,
                bf16=args.bf16,
                gradient_checkpointing=args.gradient_checkpointing,
                model=model,
            )
    else:
        # Flat single-phase training
        print("\nRunning flat single-phase training ...")
        train_ds = PIIDataset(
            jsonl_path=train_path,
            tokenizer=tokenizer,
            label2id=label2id,
            max_length=args.max_length,
            source_cond=args.source_cond,
            allowed_sources=None,
            coarse2id=coarse2id,
        )

        # Count training records
        n_train = sum(1 for _ in open(train_path, "rb").read().split(b"\n") if _)
        steps_per_epoch = math.ceil(n_train / (args.batch_size * args.grad_accum))
        max_steps = steps_per_epoch * args.epochs
        warmup_steps = max(1, int(0.05 * max_steps))

        train_args = TrainingArguments(
            output_dir=str(output_dir / "checkpoints"),
            max_steps=max_steps,
            per_device_train_batch_size=args.batch_size,
            gradient_accumulation_steps=args.grad_accum,
            learning_rate=args.lr,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            logging_steps=max(1, steps_per_epoch // 20),
            eval_strategy="steps",
            eval_steps=max(1, steps_per_epoch // 4),
            save_strategy="steps",
            save_steps=max(1, steps_per_epoch // 4),
            load_best_model_at_end=True,
            metric_for_best_model="f1",
            greater_is_better=True,
            bf16=args.bf16,
            fp16=False,
            gradient_checkpointing=args.gradient_checkpointing,
            dataloader_num_workers=2,
            report_to="none",
            label_names=["labels"],
        )

        collator = PIIDataCollator(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)

        trainer = Trainer(
            model=model,
            args=train_args,
            train_dataset=train_ds,
            eval_dataset=val_list,
            data_collator=collator,
            compute_metrics=make_compute_metrics(id2label),
        )
        trainer.train()
        model = trainer.model

    # Final evaluation on full test set
    print("\nRunning final evaluation on full test split ...")
    test_ds = PIIDataset(
        jsonl_path=splits_dir / "test.jsonl",
        tokenizer=tokenizer,
        label2id=label2id,
        max_length=args.max_length,
        source_cond=args.source_cond,
        allowed_sources=None,
        coarse2id=coarse2id,
    )
    # Final eval using a fresh Trainer (predict mode)
    collator = PIIDataCollator(tokenizer=tokenizer, padding=True, label_pad_token_id=-100)
    eval_args = TrainingArguments(
        output_dir=str(output_dir / "final_eval"),
        per_device_eval_batch_size=args.batch_size * 2,
        bf16=args.bf16,
        report_to="none",
        label_names=["labels"],
    )
    eval_trainer = Trainer(
        model=model,
        args=eval_args,
        data_collator=collator,
        compute_metrics=make_compute_metrics(id2label),
    )
    test_results = eval_trainer.evaluate(eval_dataset=list(test_ds))
    print("\nFinal test results:")
    for k, v in test_results.items():
        print(f"  {k}: {v:.4f}" if isinstance(v, float) else f"  {k}: {v}")

    # Save model and tokenizer
    model_save_path = output_dir / "final_model"
    model.save_pretrained(str(model_save_path))
    tokenizer.save_pretrained(str(model_save_path))
    print(f"\nModel saved -> {model_save_path}")

    # Save test results
    results_path = output_dir / "test_results.json"
    with open(results_path, "w") as f:
        json.dump({k: float(v) if isinstance(v, float) else v for k, v in test_results.items()}, f, indent=2)
    print(f"Results saved -> {results_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--splits-dir",   type=str, required=True, help="Directory with train/val/test jsonl splits")
    parser.add_argument("--output-dir",   type=str, required=True, help="Where to save checkpoints and final model")
    parser.add_argument("--model-id",     type=str, default="microsoft/deberta-v3-base")

    # Novel features
    parser.add_argument("--source-cond",        action="store_true", help="Enable source conditioning tokens")
    parser.add_argument("--curriculum",          action="store_true", help="Enable curriculum learning (3 phases)")
    parser.add_argument("--hierarchical",        action="store_true", help="Enable hierarchical classification head")
    parser.add_argument("--coarse-loss-weight",  type=float, default=0.3, help="Weight for coarse loss in hierarchical head")

    # Training
    parser.add_argument("--batch-size",           type=int,   default=8)
    parser.add_argument("--grad-accum",           type=int,   default=8)
    parser.add_argument("--max-length",           type=int,   default=256)
    parser.add_argument("--epochs",               type=int,   default=3,   help="Epochs for flat (non-curriculum) training")
    parser.add_argument("--lr",                   type=float, default=2e-5, help="Learning rate for flat training")
    parser.add_argument("--bf16",                 action="store_true")
    parser.add_argument("--gradient-checkpointing", action="store_true")

    args = parser.parse_args()
    main(args)
