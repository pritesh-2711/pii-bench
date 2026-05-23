"""Create a reproducible held-out subset for comparative paper evaluation.

This samples only from the already prepared held-out `data/test.jsonl`; it
never changes train/validation/test membership or retrains a model.
"""

import argparse
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from data_preparation import RANDOM_SEED, make_stratified_subset_by_size, save_split


def load_records(path: Path) -> list:
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def summarize(records: list) -> dict:
    sources = Counter()
    explicit_entities = Counter()
    seqeval_entities = Counter()
    continuation_starts = Counter()
    all_o = 0
    for record in records:
        sources[record["source"]] += 1
        has_seqeval_entity = False
        previous_type = None
        previous_prefix = "O"
        for label in record["labels"]:
            if label.startswith(("B-", "I-")):
                prefix, entity_type = label.split("-", 1)
                if prefix == "B":
                    explicit_entities[entity_type] += 1
                starts_span = (
                    prefix == "B"
                    or previous_prefix == "O"
                    or previous_type != entity_type
                )
                if starts_span:
                    seqeval_entities[entity_type] += 1
                    has_seqeval_entity = True
                    if prefix == "I":
                        continuation_starts[entity_type] += 1
                previous_prefix = prefix
                previous_type = entity_type
            else:
                previous_prefix = "O"
                previous_type = None
        all_o += not has_seqeval_entity
    return {
        "records": len(records),
        "entity_mentions": sum(explicit_entities.values()),
        "seqeval_entity_spans": sum(seqeval_entities.values()),
        "bio_continuation_span_starts": sum(continuation_starts.values()),
        "observed_entity_types": len(seqeval_entities),
        "all_o_records": all_o,
        "source_record_counts": dict(sorted(sources.items())),
        "entity_mention_counts": dict(explicit_entities.most_common()),
        "seqeval_entity_span_counts": dict(seqeval_entities.most_common()),
        "bio_continuation_span_start_counts": dict(continuation_starts.most_common()),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Create an exact-size source-stratified subset of prepared test data."
    )
    parser.add_argument("--test-file", default="./data/test.jsonl")
    parser.add_argument("--output-file", default="./data/test_5k.jsonl")
    parser.add_argument("--summary-file", default="./data/test_5k_summary.json")
    parser.add_argument("--size", type=int, default=5_000)
    parser.add_argument("--seed", type=int, default=RANDOM_SEED)
    args = parser.parse_args()

    test_path = Path(args.test_file)
    output_path = Path(args.output_file)
    summary_path = Path(args.summary_file)

    records = load_records(test_path)
    subset = make_stratified_subset_by_size(records, args.size, args.seed)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_split(subset, output_path)

    report = {
        "method": "source-stratified proportional sampling with largest-remainder allocation",
        "seed": args.seed,
        "requested_records": args.size,
        "source_test_file": str(test_path),
        "output_file": str(output_path),
        "source_test_sha256": sha256(test_path),
        "output_sha256": sha256(output_path),
        "source_test": summarize(records),
        "subset": summarize(subset),
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)

    print(json.dumps(report, indent=2))
    print(f"\nSummary saved -> {summary_path}")


if __name__ == "__main__":
    main()
