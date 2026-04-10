"""
Fine-tunes microsoft/deberta-v3-base for PII named entity recognition.

Eval strategy
-------------
Intra-training eval (every --eval-steps) runs on val_1p.jsonl (~1% of val,
~1,400 records). This keeps each eval pass to ~seconds rather than hours.
Early stopping and best-checkpoint selection are based on this fast subset.

Final eval (after training completes) runs on the full test.jsonl (~140k
records) via trainer.predict(), giving a true held-out benchmark number.

Pre-tokenize the eval subset once before training (seconds, not hours):
    python src/train.py --pretokenize-only
    python src/train.py --use-pretokenized

This Arrow-backed eval is the recommended path on V100. Without it, each
eval pass tokenizes ~1,400 records on CPU in the dataloader, which is still
fast (~5s) but Arrow eliminates that cost entirely.

Cloud V100-SXM2-16GB:
    python src/train.py --pretokenize-only
    python src/train.py --use-pretokenized --skip-download ...

Local GPU (<=8GB VRAM):
    PYTORCH_ALLOC_CONF=expandable_segments:True python src/train.py \
        --batch-size 2 --grad-accum 32 --gradient-checkpointing

Notes on torch_compile
-----------------------
torch.compile is NOT enabled by default and is generally not worth it here:
  - Token classification batches have variable sequence lengths (different
    padding per batch), which triggers shape respecialisation on every new
    shape. On V100 / older CUDA the recompilation overhead exceeds the gain.
  - Use --torch-compile only if you have A100/H100 + CUDA >= 11.8 + PyTorch
    >= 2.1 and are running long enough that amortised compile cost pays off.
    Even then, benefit is marginal vs. bf16 + gradient checkpointing.
"""

import json
import random
import argparse
import subprocess
import numpy as np
from pathlib import Path
from collections import defaultdict

import torch
from torch.utils.data import IterableDataset
from transformers import (
    AutoTokenizer,
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification,
    EarlyStoppingCallback,
)
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

import warnings
warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

DATA_DIR          = Path("./data")
MODELS_DIR        = Path("./models")
LOCAL_MODEL_PATH  = Path("./models/deberta-v3-base")
HF_MODEL_ID       = "microsoft/deberta-v3-base"
PRETOKENIZED_DIR  = Path("./data/pretokenized")

SEED              = 42


# ---------------------------------------------------------------------------
# Line count helper (fast, no JSON parsing)
# ---------------------------------------------------------------------------

def count_jsonl_lines(path: Path) -> int:
    """
    Count lines in a JSONL file without loading it into memory.
    Uses a byte-level read so it's fast even on 1GB+ files.
    """
    count = 0
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            count += chunk.count(b"\n")
    return count


# ---------------------------------------------------------------------------
# Stratified training sample (5% / 10% of train.jsonl)
# ---------------------------------------------------------------------------

def create_stratified_train_sample(
    train_path: Path,
    fraction: float,
    seed: int,
    out_path: Path,
) -> Path:
    """
    Read train.jsonl, take a stratified sample by source field, write to
    out_path (e.g. data/train_5p.jsonl).

    Why stratified by source:
      Different source datasets have different entity-type distributions.
      Uniform random sampling would under-represent minority sources.
      Stratifying preserves the per-source ratio in the sample.

    Memory note:
      Stores raw JSON lines (strings) grouped by source — no full object
      deserialization. For 880k lines at ~300 bytes each that is ~264 MB,
      well within the headroom of a 16 GB V100 before model is loaded.

    Returns out_path so the caller can feed it straight to PIIIterableDataset.
    """
    print(f"\n  Creating stratified {fraction:.0%} training sample from {train_path} ...")
    by_source: dict[str, list[str]] = defaultdict(list)
    with open(train_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            # parse only enough to get the source key
            rec = json.loads(line)
            by_source[rec.get("source", "unknown")].append(line)

    rng = random.Random(seed)
    sampled: list[str] = []
    print(f"  {'Source':<30} {'Total':>8} {'Sampled':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    for source, lines in sorted(by_source.items()):
        n = max(1, int(len(lines) * fraction))
        chosen = rng.sample(lines, min(n, len(lines)))
        sampled.extend(chosen)
        print(f"  {source:<30} {len(lines):>8,} {len(chosen):>8,}")
    print(f"  {'-'*30} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<30} {sum(len(v) for v in by_source.values()):>8,} {len(sampled):>8,}")

    rng.shuffle(sampled)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        for line in sampled:
            f.write(line + "\n")

    size_mb = out_path.stat().st_size / 1e6
    print(f"  Saved -> {out_path} ({size_mb:.1f} MB)")
    return out_path


# ---------------------------------------------------------------------------
# IterableDataset — streams JSONL line-by-line, no in-memory list
# ---------------------------------------------------------------------------

class PIIIterableDataset(IterableDataset):
    """
    Streams a JSONL file one line at a time.

    Advantages over the list-based PIIDataset:
      - No startup cost: does not load 1.1M records into RAM at init.
      - Constant memory footprint regardless of dataset size.
      - Python object overhead from list-of-dicts is eliminated.

    Trade-offs:
      - No random access, so shuffle must be done at the file level before
        training (the data pipeline already shuffles each split).
      - __len__ is unavailable; the caller must supply num_lines for
        steps_per_epoch / warmup calculation.
      - HuggingFace Trainer requires max_steps when IterableDataset is used
        (no automatic epoch counting). PIITrainer handles this automatically.
    """

    def __init__(
        self,
        jsonl_path: Path,
        tokenizer,
        label2id: dict,
        max_length: int,
    ):
        super().__init__()
        self.jsonl_path = jsonl_path
        self.tokenizer  = tokenizer
        self.label2id   = label2id
        self.max_length = max_length

    def __iter__(self):
        with open(self.jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                rec    = json.loads(line)
                tokens = rec["tokens"]
                labels = rec["labels"]
                yield self._encode(tokens, labels)

    def _encode(self, tokens: list, labels: list) -> dict:
        encoding = self.tokenizer(
            tokens,
            is_split_into_words=True,
            max_length=self.max_length,
            truncation=True,
            padding=False,
        )

        word_ids       = encoding.word_ids()
        aligned_labels = []
        prev_word_idx  = None

        for word_idx in word_ids:
            if word_idx is None:
                aligned_labels.append(-100)
            elif word_idx != prev_word_idx:
                raw = labels[word_idx] if word_idx < len(labels) else "O"
                aligned_labels.append(self.label2id.get(raw, self.label2id["O"]))
            else:
                aligned_labels.append(-100)
            prev_word_idx = word_idx

        encoding["labels"] = aligned_labels
        return {k: torch.tensor(v) for k, v in encoding.items()}


# ---------------------------------------------------------------------------
# Pre-tokenized Arrow dataset (opt-in via --use-pretokenized)
# ---------------------------------------------------------------------------

def pretokenize_split(
    jsonl_path: Path,
    out_dir: Path,
    tokenizer,
    label2id: dict,
    max_length: int,
    split_name: str,
    num_proc: int = 4,
):
    """
    Tokenize a JSONL split once and save as Arrow (HuggingFace datasets format).

    Benefits:
      - Tokenization CPU cost is paid once, not on every epoch.
      - Arrow uses memory-mapped IO: training reads directly from disk with
        near-zero copy overhead, keeping RAM usage low.
      - Multiple training runs reuse the same Arrow files.

    Arrow files are typically 2-3x larger than the source JSONL because token
    IDs are stored as int32 arrays rather than variable-length strings, but
    sequential read bandwidth is much higher than JSONL parsing.
    """
    try:
        from datasets import Dataset as HFDataset
    except ImportError as exc:
        raise RuntimeError("pip install datasets to use pre-tokenization") from exc

    print(f"  Pre-tokenizing {split_name} from {jsonl_path} ...")

    records = []
    with open(jsonl_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))

    def tokenize_fn(batch):
        all_input_ids      = []
        all_attention_mask = []
        all_token_type_ids = []
        all_labels         = []

        for tokens, labels in zip(batch["tokens"], batch["labels"]):
            enc = tokenizer(
                tokens,
                is_split_into_words=True,
                max_length=max_length,
                truncation=True,
                padding=False,
            )
            word_ids       = enc.word_ids()
            aligned_labels = []
            prev_word_idx  = None
            for word_idx in word_ids:
                if word_idx is None:
                    aligned_labels.append(-100)
                elif word_idx != prev_word_idx:
                    raw = labels[word_idx] if word_idx < len(labels) else "O"
                    aligned_labels.append(label2id.get(raw, label2id["O"]))
                else:
                    aligned_labels.append(-100)
                prev_word_idx = word_idx

            all_input_ids.append(enc["input_ids"])
            all_attention_mask.append(enc["attention_mask"])
            if "token_type_ids" in enc:
                all_token_type_ids.append(enc["token_type_ids"])
            all_labels.append(aligned_labels)

        result = {
            "input_ids":      all_input_ids,
            "attention_mask": all_attention_mask,
            "labels":         all_labels,
        }
        if all_token_type_ids:
            result["token_type_ids"] = all_token_type_ids
        return result

    hf_ds = HFDataset.from_list(records)
    hf_ds = hf_ds.map(
        tokenize_fn,
        batched=True,
        batch_size=1000,
        num_proc=num_proc,
        remove_columns=hf_ds.column_names,
        desc=f"Tokenizing {split_name}",
    )
    hf_ds.set_format("torch")

    out_path = out_dir / split_name
    hf_ds.save_to_disk(str(out_path))
    size_mb = sum(f.stat().st_size for f in out_path.rglob("*") if f.is_file()) / 1e6
    print(f"  Saved {split_name} Arrow dataset -> {out_path} ({size_mb:.0f} MB, {len(hf_ds):,} rows)")
    return hf_ds


def load_pretokenized(split_name: str, pretokenized_dir: Path):
    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise RuntimeError("pip install datasets to use pre-tokenization") from exc

    path = pretokenized_dir / split_name
    if not path.exists():
        raise FileNotFoundError(
            f"Pre-tokenized split '{split_name}' not found at {path}. "
            "Run with --pretokenize-only first."
        )
    ds = load_from_disk(str(path))
    ds.set_format("torch")
    return ds


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def make_compute_metrics(id2label: dict):
    def compute_metrics(eval_pred):
        logits, label_ids = eval_pred
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
# Trainer
# ---------------------------------------------------------------------------

class PIITrainer:
    def __init__(
        self,
        batch_size: int                  = 16,
        eval_batch_size: int             = 32,
        grad_accum: int                  = 4,
        learning_rate: float             = 2e-5,
        num_epochs: int                  = 10,
        max_steps: int                   = -1,
        max_length: int                  = 512,
        warmup_ratio: float              = 0.06,
        weight_decay: float              = 0.01,
        early_stopping_patience: int     = 3,
        eval_steps: int                  = 2000,
        save_steps: int                  = 2000,
        logging_steps: int               = 50,
        use_gradient_checkpointing: bool = False,
        resume_from_checkpoint: bool     = False,
        fp16_full_eval: bool             = False,
        torch_compile: bool              = False,
        use_pretokenized: bool           = False,
        pretokenized_dir: Path           = PRETOKENIZED_DIR,
        eval_accumulation_steps: int     = 1,
        prediction_loss_only: bool       = False,
        train_sample_fraction: float     = 0.0,
        skip_final_eval: bool            = False,
    ):
        self.batch_size                  = batch_size
        self.eval_batch_size             = eval_batch_size
        self.grad_accum                  = grad_accum
        self.learning_rate               = learning_rate
        self.num_epochs                  = num_epochs
        self.max_steps                   = max_steps
        self.max_length                  = max_length
        self.warmup_ratio                = warmup_ratio
        self.weight_decay                = weight_decay
        self.early_stopping_patience     = early_stopping_patience
        self.eval_steps                  = eval_steps
        self.save_steps                  = save_steps
        self.logging_steps               = logging_steps
        self.use_gradient_checkpointing  = use_gradient_checkpointing
        self.resume_from_checkpoint      = resume_from_checkpoint
        self.fp16_full_eval              = fp16_full_eval
        self.torch_compile               = torch_compile
        self.use_pretokenized            = use_pretokenized
        self.pretokenized_dir            = pretokenized_dir
        self.eval_accumulation_steps     = eval_accumulation_steps
        self.prediction_loss_only        = prediction_loss_only
        self.train_sample_fraction       = train_sample_fraction
        self.skip_final_eval             = skip_final_eval

        # Label mapping
        with open(DATA_DIR / "label_mapping.json") as f:
            mapping = json.load(f)
        self.label2id   = mapping["label2id"]
        self.id2label   = {int(k): v for k, v in mapping["id2label"].items()}
        self.labels     = mapping["labels"]
        self.num_labels = mapping["num_labels"]

        print(f"Labels      : {self.num_labels} total ({len(self.labels)} including O)")

        model_source = str(LOCAL_MODEL_PATH) if LOCAL_MODEL_PATH.exists() else HF_MODEL_ID
        print(f"Model source: {model_source}")

        self.tokenizer = AutoTokenizer.from_pretrained(model_source)

        self.model = AutoModelForTokenClassification.from_pretrained(
            model_source,
            num_labels=self.num_labels,
            id2label=self.id2label,
            label2id=self.label2id,
            ignore_mismatched_sizes=True,
        )

        if self.use_gradient_checkpointing:
            self.model.gradient_checkpointing_enable()

        if self.torch_compile:
            # Only beneficial on A100/H100 + CUDA >= 11.8 + PyTorch >= 2.1
            # with long runs. Variable-length batches cause repeated
            # recompilation on older GPUs, negating any gain.
            print("torch.compile enabled (ensure CUDA >= 11.8 + PyTorch >= 2.1)")
            self.model = torch.compile(self.model)

        self.device   = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()

        print(f"Device           : {self.device}")
        print(f"Precision        : {'bf16' if self.use_bf16 else 'fp32'}")
        print(f"fp16_full_eval   : {self.fp16_full_eval}")
        print(f"torch_compile    : {self.torch_compile}")
        print(f"eval_accum_steps : {self.eval_accumulation_steps}")
        print(f"pred_loss_only   : {self.prediction_loss_only}")
        print(f"Grad ckpt        : {'enabled' if self.use_gradient_checkpointing else 'disabled'}")
        print(f"Max length       : {self.max_length}")
        print(f"Dataset mode     : {'val_1p Arrow' if self.use_pretokenized else 'val_1p streaming JSONL'} (intra-training eval)")
        if self.max_steps > 0:
            print(f"Max steps        : {self.max_steps} (overrides epochs)")

    # ------------------------------------------------------------------
    # Dataset loading
    # ------------------------------------------------------------------

    def pretokenize(self, num_proc: int = 4):
        """
        Pre-tokenize the eval subsets (val_1p and test_1p) and save as Arrow.

        Only the ~1% subsets are pre-tokenized, not the full val/test splits.
        This runs in seconds and produces small Arrow files (~10-20 MB each).
        The full test split is evaluated at the end of training via streaming,
        so it does not need to be pre-tokenized.

        Call once before training:
            python src/train.py --pretokenize-only
        Then train with:
            python src/train.py --use-pretokenized ...
        """
        self.pretokenized_dir.mkdir(parents=True, exist_ok=True)
        for split in ("val_1p", "test_1p"):
            jsonl_path = DATA_DIR / f"{split}.jsonl"
            if not jsonl_path.exists():
                raise FileNotFoundError(
                    f"{jsonl_path} not found. "
                    "Re-run the data pipeline first: python run_data_pipeline.py"
                )
            pretokenize_split(
                jsonl_path=jsonl_path,
                out_dir=self.pretokenized_dir,
                tokenizer=self.tokenizer,
                label2id=self.label2id,
                max_length=self.max_length,
                split_name=split,
                num_proc=num_proc,
            )
        print(f"\nPre-tokenized Arrow datasets saved to: {self.pretokenized_dir}")

    def load_datasets(self):
        """
        Training streams from train.jsonl (full) or a stratified sample of it.
        Pass --train-sample-fraction 0.05 to use 5% of training data
        (stratified by source), or 0.10 for 10%.  The sampled file is written
        to data/train_Xp.jsonl so it can be inspected and reused.

        Intra-training eval uses val_1p (Arrow or streaming JSONL, ~1,400 records).
        Final eval (called after training) uses the full test.jsonl via streaming,
        unless --skip-final-eval is set (recommended when you only want weights).
        """
        print("\nLoading datasets ...")

        # Resolve which training file to stream
        base_train_path = DATA_DIR / "train.jsonl"
        if self.train_sample_fraction > 0.0:
            pct = int(round(self.train_sample_fraction * 100))
            sampled_path = DATA_DIR / f"train_{pct}p.jsonl"
            if sampled_path.exists():
                print(f"  Re-using existing stratified sample: {sampled_path}")
            else:
                create_stratified_train_sample(
                    train_path=base_train_path,
                    fraction=self.train_sample_fraction,
                    seed=SEED,
                    out_path=sampled_path,
                )
            train_path = sampled_path
        else:
            train_path = base_train_path

        # Training dataset — always streaming, never loaded into RAM
        self.train_ds = PIIIterableDataset(
            train_path, self.tokenizer, self.label2id, self.max_length
        )
        print("  Counting training lines (byte scan) ...")
        self.train_line_count = count_jsonl_lines(train_path)
        print(f"  Train (streaming): {self.train_line_count:,}")

        # Intra-training eval dataset — val_1p (~1% of val, ~1,400 records)
        if self.use_pretokenized:
            self.val_ds = load_pretokenized("val_1p", self.pretokenized_dir)
            print(f"  Val eval (Arrow val_1p): {len(self.val_ds):,} records")
        else:
            self.val_ds = PIIIterableDataset(
                DATA_DIR / "val_1p.jsonl", self.tokenizer, self.label2id, self.max_length
            )
            val_1p_lines = count_jsonl_lines(DATA_DIR / "val_1p.jsonl")
            print(f"  Val eval (streaming val_1p): {val_1p_lines:,} records")

        # Full test set — loaded lazily, only used in evaluate() at the end
        # Kept as a streaming dataset to avoid loading 140k records into RAM.
        self.test_ds = PIIIterableDataset(
            DATA_DIR / "test.jsonl", self.tokenizer, self.label2id, self.max_length
        )
        test_lines = count_jsonl_lines(DATA_DIR / "test.jsonl")
        print(f"  Test (streaming, final eval only): {test_lines:,} records")

    # ------------------------------------------------------------------
    # Training
    # ------------------------------------------------------------------

    def train(self):
        MODELS_DIR.mkdir(parents=True, exist_ok=True)
        output_dir = MODELS_DIR / "checkpoints"

        effective_batch  = self.batch_size * self.grad_accum
        steps_per_epoch  = max(1, self.train_line_count // effective_batch)

        # IterableDataset requires explicit max_steps for the Trainer.
        # If the user hasn't set it, derive it from epochs.
        using_iterable = isinstance(self.train_ds, IterableDataset) and not self.use_pretokenized
        if using_iterable and self.max_steps <= 0:
            derived_max_steps = steps_per_epoch * self.num_epochs
        else:
            derived_max_steps = self.max_steps   # -1 means "use num_epochs" for map-style

        total_steps_for_warmup = (
            derived_max_steps if derived_max_steps > 0
            else steps_per_epoch * self.num_epochs
        )
        warmup_steps = int(total_steps_for_warmup * self.warmup_ratio)

        print(f"\n  Effective batch size  : {effective_batch}")
        print(f"  Steps per epoch       : {steps_per_epoch:,}")
        print(f"  Total optimizer steps : {total_steps_for_warmup:,}")
        print(f"  Warmup steps          : {warmup_steps:,}")
        print(f"  Early stop patience   : {self.early_stopping_patience} epochs")
        print(f"  Eval every            : {self.eval_steps} steps")
        print(f"  Save every            : {self.save_steps} steps")
        print(f"  Log every             : {self.logging_steps} steps")

        on_cloud    = not self.use_gradient_checkpointing
        pin_memory  = on_cloud
        num_workers = 2

        args = TrainingArguments(
            output_dir=str(output_dir),
            seed=SEED,

            # Core training
            num_train_epochs=self.num_epochs,
            # For IterableDataset, Trainer ignores num_train_epochs and uses
            # max_steps instead. We derive it above if user didn't set it.
            max_steps=derived_max_steps if using_iterable else self.max_steps,
            per_device_train_batch_size=self.batch_size,
            per_device_eval_batch_size=self.eval_batch_size,
            gradient_accumulation_steps=self.grad_accum,

            # Optimiser
            learning_rate=self.learning_rate,
            weight_decay=self.weight_decay,
            warmup_steps=warmup_steps,
            lr_scheduler_type="linear",
            max_grad_norm=1.0,

            # Precision
            # fp16 is set to True unconditionally for mixed-precision training.
            # bf16 overrides fp16 on hardware that supports it (A100/H100).
            fp16=not self.use_bf16,
            bf16=self.use_bf16,
            # fp16_full_eval: keep False unless eval is OOMing.
            # Setting True forces logit accumulation in fp16 during evaluation,
            # which halves eval memory but can reduce metric accuracy slightly.
            # Prefer lowering eval_batch_size first; only set True if that's
            # not enough (e.g. 97-label token classification on 512-length seqs).
            fp16_full_eval=self.fp16_full_eval,

            # Evaluation & checkpointing
            eval_strategy="steps",
            save_strategy="steps",
            eval_steps=self.eval_steps,
            save_steps=self.save_steps,
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            save_total_limit=3,
            # Prevents OOM during eval by flushing accumulated logits/predictions
            # to CPU every N batches instead of holding the entire eval set in
            # memory. With 139k val records x 512 tokens x 97 labels x 4 bytes
            # = ~27GB if accumulated all at once. Set to 1 to flush every batch.
            # Increase to 2 or 4 if CPU->GPU transfer overhead becomes noticeable
            # (unlikely on this workload).
            eval_accumulation_steps=self.eval_accumulation_steps,
            # Set prediction_loss_only=True only for sanity-check runs where you
            # want loss curves without the overhead of logit accumulation and
            # seqeval metric computation. Disables compute_metrics entirely.
            # Keep False for normal training so eval_f1 is available for
            # load_best_model_at_end and early stopping.
            prediction_loss_only=self.prediction_loss_only,

            # Logging
            logging_strategy="steps",
            logging_steps=self.logging_steps,
            report_to="none",

            # Dataloader
            dataloader_pin_memory=pin_memory,
            dataloader_num_workers=num_workers,

            # Memory
            gradient_checkpointing=self.use_gradient_checkpointing,

            # torch.compile: disabled by default.
            # Not worth it for token classification:
            #   - Variable-length padded batches trigger shape recompilation.
            #   - On V100/T4 the overhead exceeds any gain.
            #   - Enable only on A100/H100 with PyTorch >= 2.1 + long runs.
            torch_compile=self.torch_compile,
        )

        collator = DataCollatorForTokenClassification(
            tokenizer=self.tokenizer,
            padding=True,
            max_length=self.max_length,
        )

        trainer = Trainer(
            model=self.model,
            args=args,
            train_dataset=self.train_ds,
            eval_dataset=self.val_ds,
            data_collator=collator,
            processing_class=self.tokenizer,
            compute_metrics=make_compute_metrics(self.id2label),
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.early_stopping_patience
                )
            ],
        )

        print("\n" + "=" * 60)
        print("TRAINING")
        print("=" * 60)

        checkpoint = None
        if self.resume_from_checkpoint:
            checkpoints_dir = MODELS_DIR / "checkpoints"
            if checkpoints_dir.exists():
                ckpts = sorted(
                    checkpoints_dir.glob("checkpoint-*"),
                    key=lambda p: int(p.name.split("-")[-1]),
                )
                if ckpts:
                    checkpoint = str(ckpts[-1])
                    print(f"Resuming from checkpoint: {checkpoint}")
                else:
                    print("No checkpoint found, starting from scratch.")
            else:
                print("No checkpoints directory found, starting from scratch.")

        trainer.train(resume_from_checkpoint=checkpoint)

        best_model_dir = MODELS_DIR / "best_model"
        print(f"\nSaving best model to {best_model_dir} ...")
        trainer.save_model(str(best_model_dir))
        self.tokenizer.save_pretrained(str(best_model_dir))
        with open(best_model_dir / "label_mapping.json", "w") as f:
            json.dump(
                {
                    "labels":     self.labels,
                    "label2id":   self.label2id,
                    "id2label":   {str(k): v for k, v in self.id2label.items()},
                    "num_labels": self.num_labels,
                },
                f,
                indent=2,
            )

        return trainer

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    def evaluate(self, trainer):
        """
        Final evaluation on the full test set after training completes.
        This is the only place the full 140k test split is evaluated.
        Intra-training eval used val_1p; this gives the true held-out number.
        eval_accumulation_steps is applied here too to prevent OOM.
        """
        print("\n" + "=" * 60)
        print("FINAL EVALUATION ON FULL TEST SET")
        print("=" * 60)

        preds_output = trainer.predict(self.test_ds, metric_key_prefix="test")
        logits       = preds_output.predictions
        label_ids    = preds_output.label_ids
        predictions  = np.argmax(logits, axis=-1)

        true_labels, true_preds = [], []
        for pred_seq, label_seq in zip(predictions, label_ids):
            seq_labels, seq_preds = [], []
            for p, l in zip(pred_seq, label_seq):
                if l == -100:
                    continue
                seq_labels.append(self.id2label[int(l)])
                seq_preds.append(self.id2label[int(p)])
            true_labels.append(seq_labels)
            true_preds.append(seq_preds)

        print("\nPer-entity F1 (seqeval span-level):")
        print(classification_report(true_labels, true_preds, digits=4))

        results = {
            "test_f1":        f1_score(true_labels, true_preds),
            "test_precision": precision_score(true_labels, true_preds),
            "test_recall":    recall_score(true_labels, true_preds),
        }
        print(f"Overall F1        : {results['test_f1']:.4f}")
        print(f"Overall Precision : {results['test_precision']:.4f}")
        print(f"Overall Recall    : {results['test_recall']:.4f}")

        results_path = MODELS_DIR / "evaluation_results.json"
        with open(results_path, "w") as f:
            json.dump(results, f, indent=2)
        print(f"Results saved to  : {results_path}")

        return results


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Fine-tune DeBERTa-v3-base for PII NER")

    parser.add_argument("--batch-size",              type=int,   default=16)
    parser.add_argument("--eval-batch-size",          type=int,   default=32,
                        help="Eval/predict batch size (default: 32). Lower if eval OOMs.")
    parser.add_argument("--grad-accum",              type=int,   default=4)
    parser.add_argument("--epochs",                  type=int,   default=10)
    parser.add_argument("--max-steps",               type=int,   default=-1,
                        help="Hard step limit. Overrides --epochs when > 0.")
    parser.add_argument("--max-length",              type=int,   default=512)
    parser.add_argument("--lr",                      type=float, default=2e-5)
    parser.add_argument("--warmup-ratio",            type=float, default=0.06)
    parser.add_argument("--weight-decay",            type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int,   default=3)
    parser.add_argument("--eval-steps",              type=int,   default=2000)
    parser.add_argument("--save-steps",              type=int,   default=2000)
    parser.add_argument("--logging-steps",           type=int,   default=50)
    parser.add_argument("--gradient-checkpointing",  action="store_true")
    parser.add_argument("--resume-from-checkpoint",  action="store_true")
    parser.add_argument(
        "--fp16-full-eval",
        action="store_true",
        help=(
            "Run evaluation in fp16. Halves eval VRAM. Use only if eval OOMs "
            "after lowering --eval-batch-size. Slight risk of metric rounding."
        ),
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help=(
            "Enable torch.compile on the model. Only beneficial on A100/H100 "
            "with CUDA >= 11.8 + PyTorch >= 2.1. Variable-length batches cause "
            "repeated recompilation on V100/T4, negating any gain."
        ),
    )
    parser.add_argument(
        "--eval-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Flush accumulated logits/predictions to CPU every N eval batches. "
            "Default 1 prevents OOM from holding all 139k val logits in memory. "
            "Increase to 2-4 only if you see CPU<->GPU transfer stalls."
        ),
    )
    parser.add_argument(
        "--prediction-loss-only",
        action="store_true",
        help=(
            "Skip logit accumulation and metric computation during eval. "
            "Use for quick sanity-check runs to see loss curves only. "
            "Disables eval_f1 and therefore load_best_model_at_end."
        ),
    )
    parser.add_argument(
        "--pretokenize-only",
        action="store_true",
        help="Tokenize all splits to Arrow format and exit. Run once before training.",
    )
    parser.add_argument(
        "--use-pretokenized",
        action="store_true",
        help=(
            "Load pre-tokenized Arrow datasets instead of streaming JSONL. "
            "Requires --pretokenize-only to have been run first."
        ),
    )
    parser.add_argument(
        "--pretokenize-num-proc",
        type=int,
        default=4,
        help="Number of parallel processes for pre-tokenization (default: 4).",
    )
    parser.add_argument(
        "--pretokenized-dir",
        type=str,
        default=str(PRETOKENIZED_DIR),
        help=f"Directory for Arrow datasets (default: {PRETOKENIZED_DIR}).",
    )
    parser.add_argument(
        "--train-sample-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of train.jsonl to use for training, stratified by source. "
            "0.05 = 5%% (~44k records from 880k), 0.10 = 10%% (~88k records). "
            "The sampled file is written to data/train_Xp.jsonl and reused on "
            "subsequent runs. Set to 0 (default) to use the full training set."
        ),
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help=(
            "Skip final evaluation on the full test set after training. "
            "Use when you only need the trained weights (models/best_model/) "
            "and want to run inference locally instead of evaluating on the "
            "full test split here."
        ),
    )

    args = parser.parse_args()

    pii_trainer = PIITrainer(
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        use_gradient_checkpointing=args.gradient_checkpointing,
        resume_from_checkpoint=args.resume_from_checkpoint,
        fp16_full_eval=args.fp16_full_eval,
        torch_compile=args.torch_compile,
        use_pretokenized=args.use_pretokenized,
        pretokenized_dir=Path(args.pretokenized_dir),
        eval_accumulation_steps=args.eval_accumulation_steps,
        prediction_loss_only=args.prediction_loss_only,
        train_sample_fraction=args.train_sample_fraction,
        skip_final_eval=args.skip_final_eval,
    )

    if args.pretokenize_only:
        pii_trainer.pretokenize(num_proc=args.pretokenize_num_proc)
        return

    pii_trainer.load_datasets()
    trainer = pii_trainer.train()
    if not pii_trainer.skip_final_eval:
        pii_trainer.evaluate(trainer)
    else:
        print("\nSkipping final test evaluation (--skip-final-eval set).")
        print(f"Best model weights saved to: {MODELS_DIR / 'best_model'}")


if __name__ == "__main__":
    main()