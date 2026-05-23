"""
Benchmark three PII/NER systems on the project test set:

    1. Our model   — FastPIIDetector on models/best_model
    2. spaCy       — en_core_web_lg   (OntoNotes NER pipeline used in PIIBench)
    3. Presidio    — Microsoft Presidio AnalyzerEngine (rule + ML hybrid)

Evaluation metric: seqeval span-level F1 / Precision / Recall, identical to
the metric used during training. Per-entity-type breakdown is reported for
all three systems.

The ground truth is a prepared JSONL test file produced by the data pipeline
(`data/test_5k.jsonl` for paper comparisons). Each record has
whitespace-tokenised `tokens` and BIO `labels`. We reconstruct the original
text as " ".join(tokens), run each system against that text, convert predicted
character spans back to BIO token labels, then evaluate.

Usage:
    # Basic run (test.jsonl and best_model must already exist)
    python run_benchmarking.py

    # Download spaCy model + Presidio NLP engine before running
    python run_benchmarking.py --download-deps

    # Re-run the data pipeline first if test.jsonl is missing
    python run_benchmarking.py --download-deps --run-data-pipeline

    # Limit evaluation to first N records (fast smoke-test)
    python run_benchmarking.py --max-records 500

    # Custom paths
    python run_benchmarking.py \
        --test-file   ./data/test_5k.jsonl \
        --model-path  ./models/best_model \
        --system-name "Direct DeBERTa" \
        --confidence-threshold 0.0 \
        --max-length 256 \
        --device cuda \
        --output-dir  ./benchmark_results

    # Skip individual systems
    python run_benchmarking.py --skip-spacy
    python run_benchmarking.py --skip-presidio
    python run_benchmarking.py --skip-our-model

    # Override the spaCy pipeline for an additional comparison
    python run_benchmarking.py --spacy-model en_core_web_trf
"""

import sys
import json
import hashlib
import time
import argparse
import subprocess
import traceback
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm
from seqeval.metrics import (
    classification_report,
    f1_score,
    precision_score,
    recall_score,
)

sys.path.append(str(Path(__file__).parent / "src"))

from exceptions import ModelNotFoundError, ModelLoadError


# ---------------------------------------------------------------------------
# Entity type normalisation maps
# ---------------------------------------------------------------------------

# spaCy OntoNotes label -> our label (or None to discard)
SPACY_LABEL_MAP: Dict[str, Optional[str]] = {
    "PERSON":      "PERSON",
    "ORG":         "ORG",
    "GPE":         "LOC",       # geo-political entity -> LOC
    "LOC":         "LOC",
    "FAC":         "LOC",       # facility -> LOC
    "DATE":        "DATE",
    "TIME":        "TIME",
    "MONEY":       "AMOUNT",
    "CARDINAL":    None,        # numbers — not PII
    "ORDINAL":     None,
    "PERCENT":     None,
    "QUANTITY":    None,
    "PRODUCT":     None,
    "EVENT":       None,
    "WORK_OF_ART": None,
    "LAW":         None,
    "LANGUAGE":    None,
    "NORP":        None,        # nationalities / religions
}

# Presidio entity type -> our label (or None to discard)
PRESIDIO_LABEL_MAP: Dict[str, Optional[str]] = {
    "PERSON":                  "PERSON",
    "EMAIL_ADDRESS":           "EMAIL",
    "PHONE_NUMBER":            "PHONE_NUMBER",
    "US_SSN":                  "SSN",
    "US_BANK_NUMBER":          "ACCOUNT_NUMBER",
    "CREDIT_CARD":             "CREDIT_CARD",
    "IBAN_CODE":               "IBAN",
    "IP_ADDRESS":              "IP_ADDRESS",
    "URL":                     "URL",
    "DATE_TIME":               "DATE_TIME",
    "LOCATION":                "LOC",
    "ORGANIZATION":            "ORG",
    "US_DRIVER_LICENSE":       "DRIVER_LICENSE_NUMBER",
    "US_PASSPORT":             "PASSPORT_NUMBER",
    "US_ITIN":                 "TAX_ID",
    "MEDICAL_LICENSE":         None,
    "NRP":                     None,    # nationality / religion / political
    "CRYPTO":                  "CRYPTO_ADDRESS",
    "UK_NHS":                  None,
    "SG_NRIC_FIN":             None,
    "AU_ABN":                  None,
    "AU_ACN":                  None,
    "AU_TFN":                  None,
    "AU_MEDICARE":             None,
    "IN_PAN":                  "TAX_ID",
    "IN_AADHAAR":              "NATIONAL_ID",
    "IN_VEHICLE_REGISTRATION": None,
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------

def load_test_records(path: Path, max_records: Optional[int]) -> List[Dict]:
    if not path.exists():
        raise FileNotFoundError(
            f"Test file not found at '{path}'. "
            "Run the data pipeline first: python run_data_pipeline.py"
        )
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    if not records:
        raise ValueError(f"Test file '{path}' is empty.")
    if max_records:
        records = records[:max_records]
    print(f"Loaded {len(records):,} test records from {path}")
    return records


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Span -> BIO alignment
# ---------------------------------------------------------------------------

def spans_to_bio(
    tokens: List[str],
    spans: List[Tuple[int, int, str]],   # (char_start, char_end, entity_type)
) -> List[str]:
    """
    Convert predicted character-offset spans back to BIO token labels
    aligned to the whitespace-tokenised token list.

    The text is reconstructed as " ".join(tokens) which exactly matches how
    consolidate_pii_datasets.py builds its records (whitespace tokenisation).

    Overlapping spans are handled by first-span-wins (spans should be sorted
    by start offset before calling this function).
    """
    text   = " ".join(tokens)
    labels = ["O"] * len(tokens)

    # Build char_offset -> token_index map
    char_to_tok: Dict[int, int] = {}
    pos = 0
    for tok_idx, tok in enumerate(tokens):
        start = text.find(tok, pos)
        if start == -1:
            pos += 1
            continue
        for c in range(start, start + len(tok)):
            char_to_tok[c] = tok_idx
        pos = start + len(tok)

    for char_start, char_end, etype in sorted(spans, key=lambda s: s[0]):
        first_tok = char_to_tok.get(char_start)
        last_tok  = char_to_tok.get(char_end - 1)

        # Fallback: search nearby chars for token boundary
        if first_tok is None:
            for offset in range(1, 6):
                first_tok = char_to_tok.get(char_start + offset)
                if first_tok is not None:
                    break
        if last_tok is None:
            for offset in range(1, 6):
                last_tok = char_to_tok.get(char_end - 1 - offset)
                if last_tok is not None:
                    break
        if first_tok is None or last_tok is None:
            continue

        # Skip if the first token is already labelled (first-span-wins)
        if labels[first_tok] != "O":
            continue

        labels[first_tok] = f"B-{etype}"
        for i in range(first_tok + 1, last_tok + 1):
            if labels[i] == "O":
                labels[i] = f"I-{etype}"

    return labels


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------

def compute_metrics(
    true_labels: List[List[str]],
    pred_labels: List[List[str]],
    system_name: str,
) -> Dict:
    try:
        report = classification_report(
            true_labels, pred_labels, digits=4, output_dict=True
        )
        overall_f1        = f1_score(true_labels, pred_labels)
        overall_precision = precision_score(true_labels, pred_labels)
        overall_recall    = recall_score(true_labels, pred_labels)
    except Exception as exc:
        print(f"  WARNING: seqeval error for {system_name}: {exc}")
        report            = {}
        overall_f1        = 0.0
        overall_precision = 0.0
        overall_recall    = 0.0

    return {
        "system":            system_name,
        "overall_f1":        round(overall_f1, 4),
        "overall_precision": round(overall_precision, 4),
        "overall_recall":    round(overall_recall, 4),
        "per_entity":        report,
    }


# ---------------------------------------------------------------------------
# System runners
# ---------------------------------------------------------------------------

# ---- Our model -------------------------------------------------------------

def run_our_model(
    records: List[Dict],
    model_path: str,
    batch_size: int,
    confidence_threshold: float,
    max_length: int,
    device: Optional[str],
) -> Tuple[List[List[str]], List[List[str]], float]:
    """
    Returns (true_labels, pred_labels, elapsed_seconds).
    """
    from inference import FastPIIDetector

    try:
        detector = FastPIIDetector(
            model_path=model_path,
            confidence_threshold=confidence_threshold,
            batch_size=batch_size,
            device=device,
            max_length=max_length,
        )
    except (ModelNotFoundError, ModelLoadError) as exc:
        raise RuntimeError(str(exc)) from exc

    texts       = [" ".join(r["tokens"]) for r in records]
    true_labels = [r["labels"] for r in records]

    t0      = time.perf_counter()
    results = detector.batch_detect(texts)
    elapsed = time.perf_counter() - t0

    pred_labels = []
    for record, result in zip(records, results):
        if result.error:
            pred_labels.append(["O"] * len(record["tokens"]))
            continue
        spans = [
            (e["start"], e["end"], e["type"])
            for e in result.entities
        ]
        pred_labels.append(spans_to_bio(record["tokens"], spans))

    return true_labels, pred_labels, elapsed


# ---- spaCy -----------------------------------------------------------------

def run_spacy(
    records: List[Dict],
    batch_size: int,
    model_name: str,
) -> Tuple[List[List[str]], List[List[str]], float]:
    try:
        import spacy
    except ImportError as exc:
        raise RuntimeError(
            "spaCy is not installed. Run: pip install spacy"
        ) from exc

    try:
        nlp = spacy.load(model_name)
    except OSError as exc:
        raise RuntimeError(
            f"spaCy model '{model_name}' not found. "
            f"Run: python -m spacy download {model_name}"
        ) from exc

    texts       = [" ".join(r["tokens"]) for r in records]
    true_labels = [r["labels"] for r in records]

    pred_labels = []
    t0 = time.perf_counter()

    for doc, record in tqdm(
        zip(nlp.pipe(texts, batch_size=batch_size), records),
        total=len(records),
        desc="  spaCy",
        leave=False,
    ):
        spans = []
        for ent in doc.ents:
            mapped = SPACY_LABEL_MAP.get(ent.label_)
            if mapped is not None:
                spans.append((ent.start_char, ent.end_char, mapped))
        pred_labels.append(spans_to_bio(record["tokens"], spans))

    elapsed = time.perf_counter() - t0
    return true_labels, pred_labels, elapsed


# ---- Presidio --------------------------------------------------------------

def run_presidio(
    records: List[Dict],
) -> Tuple[List[List[str]], List[List[str]], float]:
    try:
        from presidio_analyzer import AnalyzerEngine
    except ImportError as exc:
        raise RuntimeError(
            "presidio-analyzer is not installed. "
            "Run: pip install presidio-analyzer"
        ) from exc

    try:
        from presidio_analyzer.nlp_engine import (
            NlpEngineProvider,
            TransformersNlpEngine,
        )
        # Use the spaCy-backed NLP engine (lighter than transformer engine for
        # Presidio, which adds its own rule-based recognisers on top)
        analyzer = AnalyzerEngine()
    except Exception as exc:
        raise RuntimeError(f"Failed to initialise Presidio: {exc}") from exc

    texts       = [" ".join(r["tokens"]) for r in records]
    true_labels = [r["labels"] for r in records]

    pred_labels = []
    t0 = time.perf_counter()

    for text, record in tqdm(
        zip(texts, records),
        total=len(records),
        desc="  Presidio",
        leave=False,
    ):
        try:
            results = analyzer.analyze(text=text, language="en")
        except Exception:
            pred_labels.append(["O"] * len(record["tokens"]))
            continue

        spans = []
        for res in results:
            mapped = PRESIDIO_LABEL_MAP.get(res.entity_type)
            if mapped is not None:
                spans.append((res.start, res.end, mapped))
        pred_labels.append(spans_to_bio(record["tokens"], spans))

    elapsed = time.perf_counter() - t0
    return true_labels, pred_labels, elapsed


# ---------------------------------------------------------------------------
# Dependency download
# ---------------------------------------------------------------------------

def download_deps(run_data_pipeline: bool, data_pipeline_args: List[str]):
    print("\n[DEPS] Downloading dependencies ...")

    # This is the spaCy pipeline used directly and by Presidio in the PIIBench comparison.
    print("  Downloading spaCy en_core_web_lg ...")
    try:
        subprocess.run(
            [sys.executable, "-m", "spacy", "download", "en_core_web_lg"],
            check=True,
        )
        print("  spaCy model downloaded.")
    except subprocess.CalledProcessError as exc:
        print(f"  WARNING: spaCy download failed: {exc}")

    # Data pipeline
    if run_data_pipeline:
        print("\n  Running data pipeline ...")
        cmd = [sys.executable, "run_data_pipeline.py"] + data_pipeline_args
        subprocess.run(cmd, check=True)
        print("  Data pipeline complete.")


# ---------------------------------------------------------------------------
# Report rendering
# ---------------------------------------------------------------------------

def _fmt(value: float) -> str:
    return f"{value:.4f}"


def print_summary_table(all_metrics: List[Dict], elapsed: Dict[str, float]):
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    header = f"{'System':<22} {'F1':>8} {'Precision':>10} {'Recall':>8} {'Time(s)':>10}"
    print(header)
    print("-" * 70)
    for m in all_metrics:
        name = m["system"]
        print(
            f"{name:<22} "
            f"{_fmt(m['overall_f1']):>8} "
            f"{_fmt(m['overall_precision']):>10} "
            f"{_fmt(m['overall_recall']):>8} "
            f"{elapsed.get(name, 0):>10.1f}"
        )
    print("=" * 70)


def print_per_entity_table(all_metrics: List[Dict]):
    # Collect all entity types seen across systems
    entity_types = set()
    for m in all_metrics:
        for key in m["per_entity"]:
            if key not in ("micro avg", "macro avg", "weighted avg"):
                entity_types.add(key)

    if not entity_types:
        return

    print("\n" + "=" * 70)
    print("PER-ENTITY F1 BREAKDOWN")
    print("=" * 70)

    system_names = [m["system"] for m in all_metrics]
    col_w = 12
    header = f"{'Entity':<22}" + "".join(f"{n:>{col_w}}" for n in system_names)
    print(header)
    print("-" * (22 + col_w * len(system_names)))

    for etype in sorted(entity_types):
        row = f"{etype:<22}"
        for m in all_metrics:
            entity_data = m["per_entity"].get(etype, {})
            f1 = entity_data.get("f1-score", 0.0) if entity_data else 0.0
            row += f"{_fmt(f1):>{col_w}}"
        print(row)

    print("=" * 70)


def print_classification_reports(all_metrics: List[Dict]):
    for m in all_metrics:
        print(f"\n{'=' * 70}")
        print(f"FULL CLASSIFICATION REPORT — {m['system']}")
        print("=" * 70)
        report = m["per_entity"]
        if not report:
            print("  (no report available)")
            continue
        header = f"{'Entity':<30} {'Precision':>10} {'Recall':>8} {'F1':>8} {'Support':>10}"
        print(header)
        print("-" * 70)
        for key, vals in report.items():
            if not isinstance(vals, dict):
                continue
            print(
                f"{key:<30} "
                f"{vals.get('precision', 0):>10.4f} "
                f"{vals.get('recall', 0):>8.4f} "
                f"{vals.get('f1-score', 0):>8.4f} "
                f"{int(vals.get('support', 0)):>10}"
            )


# ---------------------------------------------------------------------------
# Save results
# ---------------------------------------------------------------------------

def to_json_native(value):
    """Convert seqeval/numpy values into objects accepted by json.dump."""
    if isinstance(value, dict):
        return {key: to_json_native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_json_native(item) for item in value]
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


def save_results(
    all_metrics: List[Dict],
    elapsed: Dict[str, float],
    output_dir: Path,
    num_records: int,
    evaluation_config: Dict,
):
    output_dir.mkdir(parents=True, exist_ok=True)

    payload = {
        "num_test_records": num_records,
        "evaluation_config": evaluation_config,
        "systems":          all_metrics,
        "elapsed_seconds":  elapsed,
    }
    out_path = output_dir / "benchmark_results.json"
    with open(out_path, "w") as f:
        json.dump(to_json_native(payload), f, indent=2)
    print(f"\nResults saved to: {out_path}")

    # Per-system seqeval report as text
    for m in all_metrics:
        report_path = output_dir / f"report_{m['system'].lower().replace(' ', '_')}.json"
        with open(report_path, "w") as f:
            json.dump(to_json_native(m), f, indent=2)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Benchmark custom model vs spaCy vs Presidio on the PII test set"
    )
    parser.add_argument(
        "--test-file",
        type=str,
        default="./data/test.jsonl",
        help="Path to test.jsonl (default: ./data/test.jsonl)",
    )
    parser.add_argument(
        "--model-path",
        type=str,
        default="./models/best_model",
        help="Path to fine-tuned model directory (default: ./models/best_model)",
    )
    parser.add_argument(
        "--system-name",
        type=str,
        default="Our Model",
        help="Name written into the trained-model results row (default: Our Model)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./benchmark_results",
        help="Directory to write result JSON files (default: ./benchmark_results)",
    )
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Cap evaluation at N records (useful for quick smoke-tests)",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for our model and spaCy pipe (default: 32)",
    )
    parser.add_argument(
        "--confidence-threshold",
        type=float,
        default=0.0,
        help=(
            "Entity confidence threshold for trained models. Default 0.0 "
            "matches argmax token-classification evaluation."
        ),
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=256,
        help="Maximum sequence length for trained-model inference (default: 256).",
    )
    parser.add_argument(
        "--device",
        choices=["auto", "cpu", "cuda"],
        default="auto",
        help="Device for trained-model inference. Use cuda for GPU benchmark runs.",
    )
    parser.add_argument(
        "--spacy-model",
        type=str,
        default="en_core_web_lg",
        help=(
            "spaCy pipeline to benchmark. Default is en_core_web_lg, matching "
            "the original PIIBench paper."
        ),
    )
    parser.add_argument(
        "--download-deps",
        action="store_true",
        help="Download spaCy models and Presidio NLP models before benchmarking",
    )
    parser.add_argument(
        "--run-data-pipeline",
        action="store_true",
        help="Run the data pipeline before benchmarking (implies --download-deps)",
    )
    parser.add_argument(
        "--skip-our-model",
        action="store_true",
        help="Skip our fine-tuned model",
    )
    parser.add_argument(
        "--skip-spacy",
        action="store_true",
        help="Skip the configured spaCy model",
    )
    parser.add_argument(
        "--skip-presidio",
        action="store_true",
        help="Skip Microsoft Presidio",
    )

    args = parser.parse_args()

    if args.device == "cuda":
        import torch
        if not torch.cuda.is_available():
            parser.error("--device cuda requested, but CUDA is not available.")

    print("=" * 70)
    print("PII DETECTION — BENCHMARKING")
    print("=" * 70)

    # ------------------------------------------------------------------
    # Optional dependency download
    # ------------------------------------------------------------------
    if args.download_deps or args.run_data_pipeline:
        download_deps(
            run_data_pipeline=args.run_data_pipeline,
            data_pipeline_args=[],
        )

    # ------------------------------------------------------------------
    # Load test data
    # ------------------------------------------------------------------
    print("\n[1/2] Loading test data ...")
    try:
        records = load_test_records(Path(args.test_file), args.max_records)
    except (FileNotFoundError, ValueError) as exc:
        print(f"ERROR: {exc}")
        sys.exit(1)

    all_metrics: List[Dict] = []
    elapsed: Dict[str, float] = {}

    # ------------------------------------------------------------------
    # Our model
    # ------------------------------------------------------------------
    if not args.skip_our_model:
        print("\n[2/2] Running our model ...")
        try:
            true_labels, pred_labels, secs = run_our_model(
                records,
                args.model_path,
                args.batch_size,
                args.confidence_threshold,
                args.max_length,
                None if args.device == "auto" else args.device,
            )
            metrics = compute_metrics(true_labels, pred_labels, args.system_name)
            all_metrics.append(metrics)
            elapsed[args.system_name] = secs
            print(f"  Done in {secs:.1f}s — F1: {metrics['overall_f1']:.4f}")
        except Exception as exc:
            print(f"  ERROR running our model: {exc}")
            if "--debug" in sys.argv:
                traceback.print_exc()

    # ------------------------------------------------------------------
    # spaCy
    # ------------------------------------------------------------------
    if not args.skip_spacy:
        system_name = f"spaCy {args.spacy_model}"
        print(f"\n[2/2] Running {system_name} ...")
        try:
            true_labels, pred_labels, secs = run_spacy(
                records, args.batch_size, args.spacy_model
            )
            metrics = compute_metrics(true_labels, pred_labels, system_name)
            all_metrics.append(metrics)
            elapsed[system_name] = secs
            print(f"  Done in {secs:.1f}s — F1: {metrics['overall_f1']:.4f}")
        except Exception as exc:
            print(f"  ERROR running spaCy: {exc}")
            if "--debug" in sys.argv:
                traceback.print_exc()

    # ------------------------------------------------------------------
    # Presidio
    # ------------------------------------------------------------------
    if not args.skip_presidio:
        print("\n[2/2] Running Microsoft Presidio ...")
        try:
            true_labels, pred_labels, secs = run_presidio(records)
            metrics = compute_metrics(true_labels, pred_labels, "Microsoft Presidio")
            all_metrics.append(metrics)
            elapsed["Microsoft Presidio"] = secs
            print(f"  Done in {secs:.1f}s — F1: {metrics['overall_f1']:.4f}")
        except Exception as exc:
            print(f"  ERROR running Presidio: {exc}")
            if "--debug" in sys.argv:
                traceback.print_exc()

    if not all_metrics:
        print("\nNo systems produced results. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Print reports
    # ------------------------------------------------------------------
    print_summary_table(all_metrics, elapsed)
    print_per_entity_table(all_metrics)
    print_classification_reports(all_metrics)

    # ------------------------------------------------------------------
    # Save
    # ------------------------------------------------------------------
    save_results(
        all_metrics,
        elapsed,
        Path(args.output_dir),
        len(records),
        {
            "test_file": args.test_file,
            "test_file_sha256": sha256(Path(args.test_file)),
            "model_path": args.model_path,
            "trained_model_confidence_threshold": args.confidence_threshold,
            "trained_model_max_length": args.max_length,
            "trained_model_device": args.device,
            "spacy_model": args.spacy_model,
            "metric": "seqeval exact span and entity type match",
            "taxonomy": "corrected PIIBench 82-entity taxonomy",
            "label_alignment_revision": "corrected_canonical_v1",
        },
    )


if __name__ == "__main__":
    main()
