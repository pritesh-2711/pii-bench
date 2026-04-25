"""
Evaluates publicly available NER/PII models on the PIIBench test split.

Each model was trained on one of the 10 PIIBench source datasets.
We run them on our unified test set (test_1p.jsonl by default, or test.jsonl
for the full evaluation) and map their predictions to PIIBench canonical labels
so results are directly comparable to the DeBERTa fine-tuned numbers.

Models evaluated:
  - dslim/bert-base-NER                              (CoNLL-2003)
  - iiiorg/piiranha-v1-detect-personal-information   (ai4privacy)
  - nbroad/finer-139-xtremedistil-l12-h384           (FiNER-139)
  - Davlan/xlm-roberta-base-wikiann-ner               (WikiANN)
  - tomaarsen/span-marker-mbert-base-multinerd        (MultiNERD)
  - tomaarsen/span-marker-bert-base-fewnerd-fine-super (FewNERD)

Requirements:
    pip install transformers torch seqeval datasets
    pip install span-marker   # for MultiNERD and FewNERD models only

Usage:
    # Fast eval on 1% stratified test subset (~1,400 records, ~minutes)
    python run_existing_models_benchmark.py \
        --test-path   data/splits/test_1p.jsonl \
        --output-path benchmark_results/existing_models_results.json

    # Full eval on complete test split (~140k records, ~hours)
    python run_existing_models_benchmark.py \
        --test-path   data/splits/test.jsonl \
        --output-path benchmark_results/existing_models_results_full.json

    # Run only specific models
    python run_existing_models_benchmark.py \
        --test-path   data/splits/test_1p.jsonl \
        --models      conll wikiann piiranha

    # Use CPU (slower but no VRAM needed)
    python run_existing_models_benchmark.py \
        --test-path   data/splits/test_1p.jsonl \
        --device      cpu
"""

import json
import argparse
import time
import gc
from pathlib import Path

import torch
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

# ---------------------------------------------------------------------------
# Label mappings: model output label -> PIIBench canonical entity type
# O means the model's label does not correspond to any PIIBench entity type.
# ---------------------------------------------------------------------------

# dslim/bert-base-NER  (CoNLL-2003)
# Outputs: PER, ORG, LOC, MISC  (returned as entity_group by pipeline)
CONLL_MAP = {
    "PER":  "PERSON",
    "ORG":  "ORGANIZATION",
    "LOC":  "LOCATION",
    "MISC": "O",   # too generic to map to any PIIBench PII type
}

# iiiorg/piiranha-v1-detect-personal-information  (ai4privacy)
PIIRANHA_MAP = {
    "ACCOUNTNUM":       "ACCOUNT_NUMBER",
    "BUILDINGNUM":      "O",           # sub-component of address, no distinct PIIBench type
    "CITY":             "O",           # city alone not a standalone PII type in PIIBench
    "CREDITCARDNUMBER": "CREDIT_CARD",
    "DATEOFBIRTH":      "DOB",
    "DRIVERLICENSENUM": "NATIONAL_ID",
    "EMAIL":            "EMAIL",
    "GIVENNAME":        "PERSON",
    "IDCARDNUM":        "NATIONAL_ID",
    "PASSWORD":         "PASSWORD",
    "SOCIALNUM":        "SSN",
    "STREET":           "STREET_ADDRESS",
    "SURNAME":          "PERSON",
    "TAXNUM":           "TAX_ID",
    "TELEPHONENUM":     "PHONE_NUMBER",
    "USERNAME":         "USERNAME",
    "ZIPCODE":          "ZIP_CODE",
}

# nbroad/finer-139-xtremedistil-l12-h384  (FiNER-139)
# 139 XBRL financial tags, all map to FINANCIAL_ENTITY in PIIBench.
# We use a wildcard: any non-O label -> FINANCIAL_ENTITY.
FINER_DEFAULT_LABEL = "FINANCIAL_ENTITY"

# Davlan/xlm-roberta-base-wikiann-ner  (WikiANN)
# Outputs: PER, ORG, LOC
WIKIANN_MAP = {
    "PER": "PERSON",
    "ORG": "ORGANIZATION",
    "LOC": "LOCATION",
}

# tomaarsen/span-marker-mbert-base-multinerd  (MultiNERD)
# Outputs: PER, ORG, LOC, ANIM, BIO, CEL, DIS, EVE, FOOD, INST, MEDIA, MYTH, PLANT, TIME, VEHI
MULTINERD_MAP = {
    "PER":   "PERSON",
    "ORG":   "ORGANIZATION",
    "LOC":   "LOCATION",
    "ANIM":  "O",
    "BIO":   "O",
    "CEL":   "O",
    "DIS":   "O",
    "EVE":   "O",
    "FOOD":  "O",
    "INST":  "O",
    "MEDIA": "O",
    "MYTH":  "O",
    "PLANT": "O",
    "TIME":  "TIME",
    "VEHI":  "VEHICLE",
}

# tomaarsen/span-marker-bert-base-fewnerd-fine-super  (FewNERD)
# Outputs entity_group like "person-actor", "organization-company", etc.
# We match by prefix (category before the hyphen).
FEWNERD_CATEGORY_MAP = {
    "person":       "PERSON",
    "organization": "ORGANIZATION",
    "location":     "LOCATION",
    "other":        "O",    # "other-currency", "other-disease" etc. don't cleanly map
    "art":          "O",
    "building":     "O",
    "event":        "O",
    "product":      "O",
}
# Exception: other-currency maps to CURRENCY
FEWNERD_EXACT_MAP = {
    "other-currency": "CURRENCY",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_records(jsonl_path: Path) -> list:
    records = []
    with open(jsonl_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    return records


def tokens_to_text(tokens: list) -> str:
    return " ".join(tokens)


def build_char_to_token_map(tokens: list) -> dict:
    """Map character offset -> token index for the space-joined text."""
    char_to_tok = {}
    pos = 0
    for i, tok in enumerate(tokens):
        for j in range(len(tok)):
            char_to_tok[pos + j] = i
        pos += len(tok) + 1  # +1 for the joining space
    return char_to_tok


def spans_to_bio(tokens: list, spans: list, entity_map_fn) -> list:
    """
    Convert a list of entity spans (with start/end char offsets and an entity
    group string) to word-level BIO labels aligned with `tokens`.

    entity_map_fn(entity_group: str) -> canonical_pii_label or "O"
    """
    bio = ["O"] * len(tokens)
    char_to_tok = build_char_to_token_map(tokens)
    text_len = sum(len(t) for t in tokens) + max(0, len(tokens) - 1)

    for span in spans:
        mapped = entity_map_fn(span.get("entity_group", span.get("label", "")))
        if mapped == "O":
            continue

        start_char = span["start"]
        end_char   = span["end"] - 1  # inclusive

        # find start token
        start_tok = char_to_tok.get(start_char)
        if start_tok is None:
            # try adjacent chars in case of off-by-one from pipeline tokenization
            for delta in range(1, 4):
                start_tok = char_to_tok.get(start_char + delta) or char_to_tok.get(start_char - delta)
                if start_tok is not None:
                    break
        if start_tok is None:
            continue

        # find end token
        end_tok = char_to_tok.get(end_char)
        if end_tok is None:
            for delta in range(1, 4):
                end_tok = char_to_tok.get(end_char - delta) or char_to_tok.get(end_char + delta)
                if end_tok is not None:
                    break
        if end_tok is None:
            end_tok = start_tok

        bio[start_tok] = f"B-{mapped}"
        for t in range(start_tok + 1, end_tok + 1):
            bio[t] = f"I-{mapped}"

    return bio


def convert_types(obj):
    """Recursively convert numpy/non-serializable types to Python natives."""
    import numpy as np
    if isinstance(obj, dict):
        return {k: convert_types(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [convert_types(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.ndarray,)):
        return obj.tolist()
    return obj


def compute_seqeval(true_labels: list, pred_labels: list) -> dict:
    return {
        "f1":        round(f1_score(true_labels, pred_labels), 4),
        "precision": round(precision_score(true_labels, pred_labels), 4),
        "recall":    round(recall_score(true_labels, pred_labels), 4),
        "report":    convert_types(classification_report(true_labels, pred_labels, output_dict=True)),
    }


def free_model(model_obj):
    del model_obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ---------------------------------------------------------------------------
# Model runners
# ---------------------------------------------------------------------------

def run_hf_pipeline_model(
    model_id: str,
    records: list,
    entity_map_fn,
    device: int,
    batch_size: int = 16,
) -> dict:
    """
    Runs a standard HuggingFace token classification pipeline.
    aggregation_strategy="simple" merges subword tokens into word spans.
    """
    from transformers import pipeline

    print(f"  Loading {model_id} ...")
    nlp = pipeline(
        "ner",
        model=model_id,
        aggregation_strategy="simple",
        device=device,
    )

    true_labels, pred_labels = [], []
    texts = [tokens_to_text(r["tokens"]) for r in records]

    print(f"  Running inference on {len(records)} records ...")
    t0 = time.time()

    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        batch_texts   = texts[i : i + batch_size]
        batch_spans   = nlp(batch_texts)

        for rec, spans in zip(batch_records, batch_spans):
            pred_bio = spans_to_bio(rec["tokens"], spans, entity_map_fn)
            true_labels.append(rec["labels"])
            pred_labels.append(pred_bio)

        if (i // batch_size) % 10 == 0:
            print(f"    {i + len(batch_records)}/{len(records)} records done ...")

    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s")
    free_model(nlp)
    return compute_seqeval(true_labels, pred_labels), elapsed


def run_span_marker_model(
    model_id: str,
    records: list,
    entity_map_fn,
    device_str: str,
    batch_size: int = 16,
) -> dict:
    """
    Runs a SpanMarker model (requires: pip install span-marker).
    SpanMarker returns {'label': ..., 'score': ..., 'span': ..., 'word_start_index': ..., 'word_end_index': ...}
    """
    try:
        from span_marker import SpanMarkerModel
    except ImportError:
        raise ImportError(
            "SpanMarker models require: pip install span-marker\n"
            "  Models affected: MultiNERD, FewNERD"
        )

    print(f"  Loading {model_id} ...")
    model = SpanMarkerModel.from_pretrained(model_id)
    model = model.to(device_str)

    true_labels, pred_labels = [], []
    t0 = time.time()

    print(f"  Running inference on {len(records)} records ...")
    for i in range(0, len(records), batch_size):
        batch_records = records[i : i + batch_size]
        batch_tokens  = [r["tokens"] for r in batch_records]

        # SpanMarker accepts pre-tokenized input
        batch_preds = model.predict(batch_tokens)

        for rec, spans in zip(batch_records, batch_preds):
            bio = ["O"] * len(rec["tokens"])
            for span in spans:
                raw_label = span.get("label", "")
                mapped    = entity_map_fn(raw_label)
                if mapped == "O":
                    continue
                ws = span["word_start_index"]
                we = span["word_end_index"]  # exclusive
                if ws >= len(bio):
                    continue
                bio[ws] = f"B-{mapped}"
                for t in range(ws + 1, min(we, len(bio))):
                    bio[t] = f"I-{mapped}"

            true_labels.append(rec["labels"])
            pred_labels.append(bio)

        if (i // batch_size) % 10 == 0:
            print(f"    {i + len(batch_records)}/{len(records)} records done ...")

    elapsed = time.time() - t0
    print(f"  Inference done in {elapsed:.1f}s")
    free_model(model)
    return compute_seqeval(true_labels, pred_labels), elapsed


# ---------------------------------------------------------------------------
# Entity map functions  (one per model)
# ---------------------------------------------------------------------------

def conll_map(entity_group: str) -> str:
    return CONLL_MAP.get(entity_group.upper(), "O")

def piiranha_map(entity_group: str) -> str:
    return PIIRANHA_MAP.get(entity_group.upper(), "O")

def finer_map(entity_group: str) -> str:
    # All XBRL tags (any non-O label) -> FINANCIAL_ENTITY
    return FINER_DEFAULT_LABEL if entity_group and entity_group.upper() != "O" else "O"

def wikiann_map(entity_group: str) -> str:
    return WIKIANN_MAP.get(entity_group.upper(), "O")

def multinerd_map(entity_group: str) -> str:
    return MULTINERD_MAP.get(entity_group.upper(), "O")

def fewnerd_map(entity_group: str) -> str:
    # Check exact map first (e.g. "other-currency")
    lower = entity_group.lower()
    if lower in FEWNERD_EXACT_MAP:
        return FEWNERD_EXACT_MAP[lower]
    # Then match by category prefix
    prefix = lower.split("-")[0] if "-" in lower else lower
    return FEWNERD_CATEGORY_MAP.get(prefix, "O")


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------

MODELS = {
    "conll": {
        "display_name":  "BERT-base CoNLL-2003 (dslim/bert-base-NER)",
        "model_id":      "dslim/bert-base-NER",
        "source_dataset": "conll2003",
        "runner":        "hf_pipeline",
        "map_fn":        conll_map,
    },
    "piiranha": {
        "display_name":  "Piiranha DeBERTa ai4privacy (iiiorg/piiranha-v1-detect-personal-information)",
        "model_id":      "iiiorg/piiranha-v1-detect-personal-information",
        "source_dataset": "ai4privacy_400k",
        "runner":        "hf_pipeline",
        "map_fn":        piiranha_map,
    },
    "finer": {
        "display_name":  "XtremeDistil FiNER-139 (nbroad/finer-139-xtremedistil-l12-h384)",
        "model_id":      "nbroad/finer-139-xtremedistil-l12-h384",
        "source_dataset": "finer_139",
        "runner":        "hf_pipeline",
        "map_fn":        finer_map,
    },
    "wikiann": {
        "display_name":  "XLM-RoBERTa WikiANN (Davlan/xlm-roberta-base-wikiann-ner)",
        "model_id":      "Davlan/xlm-roberta-base-wikiann-ner",
        "source_dataset": "wikiann",
        "runner":        "hf_pipeline",
        "map_fn":        wikiann_map,
    },
    "multinerd": {
        "display_name":  "SpanMarker mBERT MultiNERD (tomaarsen/span-marker-mbert-base-multinerd)",
        "model_id":      "tomaarsen/span-marker-mbert-base-multinerd",
        "source_dataset": "multinerd",
        "runner":        "span_marker",
        "map_fn":        multinerd_map,
    },
    "fewnerd": {
        "display_name":  "SpanMarker BERT FewNERD (tomaarsen/span-marker-bert-base-fewnerd-fine-super)",
        "model_id":      "tomaarsen/span-marker-bert-base-fewnerd-fine-super",
        "source_dataset": "few_nerd",
        "runner":        "span_marker",
        "map_fn":        fewnerd_map,
    },
}


# ---------------------------------------------------------------------------
# Per-source breakdown helper
# ---------------------------------------------------------------------------

def compute_per_source_metrics(records: list, pred_by_idx: list, id2label: dict = None) -> dict:
    """
    Group records by source and compute F1 per source.
    """
    source_true = {}
    source_pred = {}
    for rec, pred in zip(records, pred_by_idx):
        src = rec.get("source", "unknown")
        source_true.setdefault(src, []).append(rec["labels"])
        source_pred.setdefault(src, []).append(pred)

    per_source = {}
    for src in sorted(source_true):
        try:
            per_source[src] = {
                "f1":        round(f1_score(source_true[src], source_pred[src]), 4),
                "precision": round(precision_score(source_true[src], source_pred[src]), 4),
                "recall":    round(recall_score(source_true[src], source_pred[src]), 4),
                "n_records": len(source_true[src]),
            }
        except Exception:
            per_source[src] = {"f1": 0.0, "precision": 0.0, "recall": 0.0, "n_records": len(source_true[src])}
    return per_source


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(args):
    test_path  = Path(args.test_path)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    device_int = 0 if (torch.cuda.is_available() and args.device != "cpu") else -1
    device_str = "cuda:0" if device_int == 0 else "cpu"
    print(f"Device: {device_str}")

    print(f"Loading test records from {test_path} ...")
    records = load_records(test_path)
    print(f"  {len(records)} records loaded")

    # which models to run
    if args.models:
        model_keys = [k.strip() for k in args.models]
    else:
        model_keys = list(MODELS.keys())

    all_results = {
        "test_path":   str(test_path),
        "num_records": len(records),
        "device":      device_str,
        "models":      {},
    }

    # Load any existing results so we can resume partial runs
    if output_path.exists():
        try:
            with open(output_path, encoding="utf-8") as f:
                existing = json.load(f)
            all_results["models"].update(existing.get("models", {}))
            print(f"Resumed from existing results: {list(all_results['models'].keys())}")
        except (json.JSONDecodeError, ValueError):
            print("Existing results file is corrupt — starting fresh.")

    for key in model_keys:
        if key not in MODELS:
            print(f"Unknown model key '{key}', skipping. Valid keys: {list(MODELS.keys())}")
            continue
        if key in all_results["models"]:
            print(f"\nSkipping {key} (already in results file)")
            continue

        cfg = MODELS[key]
        print(f"\n{'='*60}")
        print(f"Model: {cfg['display_name']}")
        print(f"{'='*60}")

        try:
            # Collect predictions for per-source analysis
            preds_for_source = []
            original_true    = [r["labels"] for r in records]

            if cfg["runner"] == "hf_pipeline":
                from transformers import AutoTokenizer, pipeline
                tokenizer = AutoTokenizer.from_pretrained(
                    cfg["model_id"], model_max_length=512
                )
                nlp = pipeline(
                    "ner",
                    model=cfg["model_id"],
                    tokenizer=tokenizer,
                    aggregation_strategy="simple",
                    device=device_int,
                )
                texts = [tokens_to_text(r["tokens"]) for r in records]
                t0 = time.time()
                for i in range(0, len(records), args.batch_size):
                    batch_recs  = records[i : i + args.batch_size]
                    batch_texts = texts[i : i + args.batch_size]
                    batch_spans = nlp(batch_texts)
                    for rec, spans in zip(batch_recs, batch_spans):
                        preds_for_source.append(
                            spans_to_bio(rec["tokens"], spans, cfg["map_fn"])
                        )
                    if (i // args.batch_size) % 20 == 0:
                        print(f"  {i + len(batch_recs)}/{len(records)}")
                elapsed = time.time() - t0
                free_model(nlp)

            elif cfg["runner"] == "span_marker":
                from span_marker import SpanMarkerModel
                model = SpanMarkerModel.from_pretrained(cfg["model_id"])
                model = model.to(device_str)
                t0 = time.time()
                for i in range(0, len(records), args.batch_size):
                    batch_recs   = records[i : i + args.batch_size]
                    batch_tokens = [r["tokens"] for r in batch_recs]
                    batch_preds  = model.predict(batch_tokens)
                    for rec, spans in zip(batch_recs, batch_preds):
                        bio = ["O"] * len(rec["tokens"])
                        for span in spans:
                            mapped = cfg["map_fn"](span.get("label", ""))
                            if mapped == "O":
                                continue
                            ws = span["word_start_index"]
                            we = span["word_end_index"]
                            if ws >= len(bio):
                                continue
                            bio[ws] = f"B-{mapped}"
                            for t in range(ws + 1, min(we, len(bio))):
                                bio[t] = f"I-{mapped}"
                        preds_for_source.append(bio)
                    if (i // args.batch_size) % 20 == 0:
                        print(f"  {i + len(batch_recs)}/{len(records)}")
                elapsed = time.time() - t0
                free_model(model)

            metrics     = compute_seqeval(original_true, preds_for_source)
            per_source  = compute_per_source_metrics(records, preds_for_source)

            all_results["models"][key] = {
                "display_name":   cfg["display_name"],
                "model_id":       cfg["model_id"],
                "source_dataset": cfg["source_dataset"],
                "overall_f1":       metrics["f1"],
                "overall_precision": metrics["precision"],
                "overall_recall":    metrics["recall"],
                "elapsed_seconds":   round(elapsed, 1),
                "per_entity":        metrics["report"],
                "per_source":        per_source,
            }

            print(f"\nResults for {key}:")
            print(f"  F1:        {metrics['f1']:.4f}")
            print(f"  Precision: {metrics['precision']:.4f}")
            print(f"  Recall:    {metrics['recall']:.4f}")
            print(f"  Time:      {elapsed:.1f}s")

        except ImportError as e:
            print(f"  SKIPPED — {e}")
            all_results["models"][key] = {"error": str(e)}

        except Exception as e:
            print(f"  ERROR — {e}")
            all_results["models"][key] = {"error": str(e)}
            import traceback; traceback.print_exc()

        # Save after every model so partial results are never lost
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(convert_types(all_results), f, indent=2)
        print(f"  Results saved -> {output_path}")

    # Print summary table
    print(f"\n{'='*70}")
    print(f"{'Model':<35} {'F1':>8} {'P':>8} {'R':>8}")
    print(f"{'-'*70}")
    for key, res in all_results["models"].items():
        if "error" in res:
            print(f"{MODELS[key]['display_name'][:35]:<35}  SKIPPED: {res['error'][:30]}")
        else:
            print(f"{key:<35} {res['overall_f1']:>8.4f} {res['overall_precision']:>8.4f} {res['overall_recall']:>8.4f}")
    print(f"{'='*70}")
    print(f"\nFull results saved -> {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--test-path",   type=str, default="data/splits/test_1p.jsonl",
                        help="Path to test JSONL (test_1p.jsonl or test.jsonl)")
    parser.add_argument("--output-path", type=str, default="benchmark_results/existing_models_results.json",
                        help="Where to save results JSON")
    parser.add_argument("--models",      type=str, nargs="+",
                        choices=list(MODELS.keys()),
                        help="Subset of models to run. Default: all")
    parser.add_argument("--batch-size",  type=int, default=16,
                        help="Inference batch size per model")
    parser.add_argument("--device",      type=str, default="auto",
                        choices=["auto", "cpu", "cuda"],
                        help="Device: auto uses CUDA if available")
    main(parser.parse_args())
