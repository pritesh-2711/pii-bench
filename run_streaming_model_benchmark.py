"""Stream a trained PII model over a large prepared JSONL test split.

This evaluator is intended for full-test evaluation of locally exported
fine-tuned models. It reads only one host-memory chunk at a time, while
FastPIIDetector performs GPU inference in smaller minibatches. Exact seqeval
span counts are accumulated incrementally, so the full label/prediction
arrays never need to be retained in memory.

Example:
    python run_streaming_model_benchmark.py \
        --test-file ./data/test.jsonl \
        --model-path ../cloud_runs/baseline/models/best_model \
        --system-name "Direct Fine-tuned DeBERTa" \
        --device cuda \
        --chunk-size 5000 \
        --batch-size 8 \
        --max-length 256 \
        --output-dir ./benchmark_results/full_test/direct_deberta
"""

import argparse
import gc
import hashlib
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterator, List

import torch
from seqeval.metrics.sequence_labeling import get_entities

sys.path.append(str(Path(__file__).parent / "src"))

from run_benchmarking import spans_to_bio
from exceptions import ModelLoadError, ModelNotFoundError
from inference import FastPIIDetector


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def iter_jsonl_chunks(
    path: Path,
    chunk_size: int,
    max_records: int = None,
) -> Iterator[List[Dict]]:
    """Yield at most chunk_size records without materialising the full split."""
    chunk = []
    yielded = 0
    with open(path, encoding="utf-8") as handle:
        for line in handle:
            if max_records is not None and yielded >= max_records:
                break
            line = line.strip()
            if not line:
                continue
            chunk.append(json.loads(line))
            yielded += 1
            if len(chunk) == chunk_size:
                yield chunk
                chunk = []
    if chunk:
        yield chunk


def safe_div(numerator: int, denominator: int) -> float:
    return numerator / denominator if denominator else 0.0


def metric_row(tp: int, fp: int, fn: int, support: int = None) -> Dict:
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    f1 = safe_div(2 * precision * recall, precision + recall)
    return {
        "precision": precision,
        "recall": recall,
        "f1-score": f1,
        "support": tp + fn if support is None else support,
    }


class StreamingSeqevalAccumulator:
    """Accumulate seqeval-equivalent exact-span statistics per record."""

    def __init__(self):
        self.counts = defaultdict(lambda: {"tp": 0, "fp": 0, "fn": 0})

    def update(self, true_labels: List[List[str]], pred_labels: List[List[str]]):
        if len(true_labels) != len(pred_labels):
            raise ValueError("Ground-truth and prediction batch lengths differ.")

        for gold_sequence, predicted_sequence in zip(true_labels, pred_labels):
            gold = set(get_entities(gold_sequence))
            predicted = set(get_entities(predicted_sequence))

            for entity_type, _start, _end in gold & predicted:
                self.counts[entity_type]["tp"] += 1
            for entity_type, _start, _end in predicted - gold:
                self.counts[entity_type]["fp"] += 1
            for entity_type, _start, _end in gold - predicted:
                self.counts[entity_type]["fn"] += 1

    def metrics(self, system_name: str) -> Dict:
        per_entity = {}
        for entity_type in sorted(self.counts):
            counts = self.counts[entity_type]
            per_entity[entity_type] = metric_row(
                counts["tp"], counts["fp"], counts["fn"]
            )

        total_tp = sum(row["tp"] for row in self.counts.values())
        total_fp = sum(row["fp"] for row in self.counts.values())
        total_fn = sum(row["fn"] for row in self.counts.values())
        total_support = total_tp + total_fn
        micro = metric_row(total_tp, total_fp, total_fn, total_support)

        rows = list(per_entity.values())
        if rows:
            macro = {
                "precision": sum(row["precision"] for row in rows) / len(rows),
                "recall": sum(row["recall"] for row in rows) / len(rows),
                "f1-score": sum(row["f1-score"] for row in rows) / len(rows),
                "support": total_support,
            }
            weighted = {
                "precision": safe_div(
                    sum(row["precision"] * row["support"] for row in rows),
                    total_support,
                ),
                "recall": safe_div(
                    sum(row["recall"] * row["support"] for row in rows),
                    total_support,
                ),
                "f1-score": safe_div(
                    sum(row["f1-score"] * row["support"] for row in rows),
                    total_support,
                ),
                "support": total_support,
            }
        else:
            macro = metric_row(0, 0, 0, 0)
            weighted = metric_row(0, 0, 0, 0)

        report = {
            **per_entity,
            "micro avg": micro,
            "macro avg": macro,
            "weighted avg": weighted,
        }
        return {
            "system": system_name,
            "overall_f1": round(micro["f1-score"], 4),
            "overall_precision": round(micro["precision"], 4),
            "overall_recall": round(micro["recall"], 4),
            "per_entity": report,
        }


def atomic_json_dump(payload: Dict, destination: Path):
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with open(temporary, "w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)
    temporary.replace(destination)


def clear_cuda_cache():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def infer_chunk_with_oom_retry(
    detector: FastPIIDetector,
    texts: List[str],
) -> List[List[Dict]]:
    """Retry a complete chunk with a reduced CUDA minibatch after an OOM."""
    while True:
        try:
            entities = []
            for start in range(0, len(texts), detector.batch_size):
                entities.extend(
                    detector._run_batch_inference(
                        texts[start: start + detector.batch_size]
                    )
                )
            return entities
        except Exception as exc:
            oom_error = "out of memory" in str(exc).lower()
            if not oom_error or detector.batch_size <= 1:
                raise RuntimeError(
                    f"Inference failed within streamed chunk: {exc}"
                ) from exc

            reduced_size = max(1, detector.batch_size // 2)
            print(
                "  CUDA out of memory detected; retrying current chunk with "
                f"GPU batch size {reduced_size}."
            )
            detector.batch_size = reduced_size
            del entities
            clear_cuda_cache()


def build_payload(
    metrics: Dict,
    elapsed_seconds: float,
    num_records: int,
    args,
    test_hash: str,
    chunks_processed: int,
    detector_batch_size: int,
    status: str,
) -> Dict:
    streaming = {
        "status": status,
        "host_chunk_size": args.chunk_size,
        "gpu_inference_batch_size_requested": args.batch_size,
        "gpu_inference_batch_size_used": detector_batch_size,
        "chunks_processed": chunks_processed,
        "records_processed": num_records,
        "max_records": args.max_records,
    }
    if args.device == "cuda":
        streaming["peak_cuda_memory_allocated_gib"] = round(
            torch.cuda.max_memory_allocated() / (1024 ** 3), 3
        )
        streaming["peak_cuda_memory_reserved_gib"] = round(
            torch.cuda.max_memory_reserved() / (1024 ** 3), 3
        )

    return {
        "num_test_records": num_records,
        "evaluation_config": {
            "test_file": args.test_file,
            "test_file_sha256": test_hash,
            "model_path": args.model_path,
            "trained_model_confidence_threshold": args.confidence_threshold,
            "trained_model_max_length": args.max_length,
            "trained_model_device": args.device,
            "metric": "seqeval exact span and entity type match",
            "taxonomy": "corrected PIIBench 82-entity taxonomy",
            "label_alignment_revision": "corrected_canonical_v1",
            "streamed_evaluation": True,
        },
        "systems": [metrics],
        "elapsed_seconds": {args.system_name: elapsed_seconds},
        "streaming": streaming,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Stream a trained PII model over the complete prepared test split."
    )
    parser.add_argument("--test-file", default="./data/test.jsonl")
    parser.add_argument("--model-path", default="./models/best_model")
    parser.add_argument("--system-name", default="Direct Fine-tuned DeBERTa")
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results/full_test/direct_deberta",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=5000,
        help="Records read and consolidated at one time in host memory (default: 5000).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=8,
        help="CUDA inference minibatch; automatically reduced after CUDA OOM (default: 8).",
    )
    parser.add_argument("--max-length", type=int, default=256)
    parser.add_argument("--confidence-threshold", type=float, default=0.0)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    parser.add_argument(
        "--max-records",
        type=int,
        default=None,
        help="Optional smoke-test limit; still reads records incrementally.",
    )
    args = parser.parse_args()

    if args.chunk_size < 1 or args.batch_size < 1:
        parser.error("--chunk-size and --batch-size must be positive integers.")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("--device cuda requested, but CUDA is not available.")

    test_path = Path(args.test_file)
    if not test_path.exists():
        parser.error(f"Test file does not exist: {test_path}")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("PII DETECTION - STREAMED TRAINED-MODEL EVALUATION")
    print("=" * 72)
    print(f"Test file             : {test_path}")
    print(f"Model                 : {args.model_path}")
    print(f"Device                : {args.device}")
    print(f"Host chunk size       : {args.chunk_size:,}")
    print(f"CUDA minibatch size   : {args.batch_size}")
    print(f"Max sequence length   : {args.max_length}")

    test_hash = sha256(test_path)
    if args.device == "cuda":
        torch.cuda.reset_peak_memory_stats()

    try:
        detector = FastPIIDetector(
            model_path=args.model_path,
            confidence_threshold=args.confidence_threshold,
            batch_size=args.batch_size,
            device=args.device,
            max_length=args.max_length,
        )
    except (ModelNotFoundError, ModelLoadError) as exc:
        raise SystemExit(f"Unable to load trained model: {exc}") from exc

    accumulator = StreamingSeqevalAccumulator()
    records_processed = 0
    chunks_processed = 0
    start_time = time.perf_counter()
    progress_path = output_dir / "streaming_progress.json"

    for chunk in iter_jsonl_chunks(test_path, args.chunk_size, args.max_records):
        chunks_processed += 1
        texts = [" ".join(record["tokens"]) for record in chunk]
        true_labels = [record["labels"] for record in chunk]
        predicted_entities = infer_chunk_with_oom_retry(detector, texts)
        pred_labels = []
        for record, entities in zip(chunk, predicted_entities):
            spans = [
                (entity["start"], entity["end"], entity["type"])
                for entity in entities
            ]
            pred_labels.append(spans_to_bio(record["tokens"], spans))

        accumulator.update(true_labels, pred_labels)
        records_processed += len(chunk)
        elapsed = time.perf_counter() - start_time
        metrics = accumulator.metrics(args.system_name)
        payload = build_payload(
            metrics,
            elapsed,
            records_processed,
            args,
            test_hash,
            chunks_processed,
            detector.batch_size,
            status="running",
        )
        atomic_json_dump(payload, progress_path)
        print(
            f"Chunk {chunks_processed:>3}: records={records_processed:>7,}  "
            f"F1={metrics['overall_f1']:.4f}  "
            f"P={metrics['overall_precision']:.4f}  "
            f"R={metrics['overall_recall']:.4f}  "
            f"time={elapsed:.1f}s"
        )

        del chunk, texts, true_labels, predicted_entities, pred_labels

    if records_processed == 0:
        raise SystemExit("No test records were read.")

    elapsed = time.perf_counter() - start_time
    final_metrics = accumulator.metrics(args.system_name)
    final_payload = build_payload(
        final_metrics,
        elapsed,
        records_processed,
        args,
        test_hash,
        chunks_processed,
        detector.batch_size,
        status="complete",
    )
    results_path = output_dir / "benchmark_results.json"
    report_slug = re.sub(
        r"[^a-z0-9_+-]",
        "_",
        args.system_name.lower().replace(" ", "_"),
    )
    report_path = output_dir / f"report_{report_slug}.json"
    atomic_json_dump(final_payload, results_path)
    atomic_json_dump(final_metrics, report_path)
    atomic_json_dump(final_payload, progress_path)

    print("\n" + "=" * 72)
    print("STREAMED EVALUATION COMPLETE")
    print("=" * 72)
    print(f"Records evaluated : {records_processed:,}")
    print(f"F1                : {final_metrics['overall_f1']:.4f}")
    print(f"Precision         : {final_metrics['overall_precision']:.4f}")
    print(f"Recall            : {final_metrics['overall_recall']:.4f}")
    print(f"Elapsed seconds   : {elapsed:.1f}")
    if args.device == "cuda":
        print(
            "Peak CUDA memory   : "
            f"{final_payload['streaming']['peak_cuda_memory_allocated_gib']:.3f} GiB allocated, "
            f"{final_payload['streaming']['peak_cuda_memory_reserved_gib']:.3f} GiB reserved"
        )
    print(f"Results saved     : {results_path}")


if __name__ == "__main__":
    main()
