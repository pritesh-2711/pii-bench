"""Compile and validate the complete corrected PIIBench comparison table.

The compiler deliberately refuses partial result sets: the reported table must
contain the original eight PIIBench comparators and both trained models, all
evaluated on the identical corrected held-out subset. A separately trained
curriculum variant can be added once its held-out result exists.
"""

import argparse
import csv
import json
from pathlib import Path


ORIGINAL_SYSTEMS = [
    {
        "system": "Microsoft Presidio",
        "source_dataset": "Multiple (rule-based)",
        "type": "Rule-based",
        "result_source": "novelty",
        "result_key": "Microsoft Presidio",
        "legacy_key": "Presidio",
    },
    {
        "system": "spaCy en_core_web_lg",
        "source_dataset": "OntoNotes 5.0",
        "type": "General NER",
        "result_source": "novelty",
        "result_key": "spaCy en_core_web_lg",
    },
    {
        "system": "SpanMarker mBERT",
        "source_dataset": "MultiNERD",
        "type": "General NER",
        "result_source": "public",
        "result_key": "multinerd",
    },
    {
        "system": "SpanMarker BERT",
        "source_dataset": "FewNERD",
        "type": "General NER",
        "result_source": "public",
        "result_key": "fewnerd",
    },
    {
        "system": "BERT-base NER",
        "source_dataset": "CoNLL-2003",
        "type": "General NER",
        "result_source": "public",
        "result_key": "conll",
    },
    {
        "system": "XLM-RoBERTa NER",
        "source_dataset": "WikiANN (en)",
        "type": "General NER",
        "result_source": "public",
        "result_key": "wikiann",
    },
    {
        "system": "Piiranha DeBERTa",
        "source_dataset": "ai4privacy-400k",
        "type": "PII-specific",
        "result_source": "public",
        "result_key": "piiranha",
    },
    {
        "system": "XtremeDistil FiNER",
        "source_dataset": "FiNER-139",
        "type": "Financial NER",
        "result_source": "public",
        "result_key": "finer",
    },
]

TRAINED_SYSTEMS = [
    {
        "system": "Direct Fine-tuned DeBERTa",
        "source_dataset": "Corrected PIIBench train split",
        "type": "Fine-tuned baseline",
        "result_source": "direct",
        "result_key": "Direct Fine-tuned DeBERTa",
    },
    {
        "system": "Source-conditioned Hierarchical DeBERTa",
        "source_dataset": "Corrected PIIBench train split",
        "type": "Proposed method",
        "result_source": "novelty",
        "result_key": "Source-conditioned Hierarchical DeBERTa",
    },
]

CURRICULUM_SYSTEM = {
    "system": "Source-conditioned Hierarchical DeBERTa + Curriculum",
    "source_dataset": "Corrected PIIBench train split",
    "type": "Proposed method + curriculum",
    "result_source": "curriculum",
    "result_key": "Source-conditioned Hierarchical DeBERTa + Curriculum",
}


def load_json(path: Path) -> dict:
    if not path.exists():
        raise FileNotFoundError(f"Required result file not found: {path}")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def indexed_trained_results(payload: dict) -> dict:
    return {row["system"]: row for row in payload.get("systems", [])}


def dataset_hash(payload: dict, source: str) -> str:
    if source == "public":
        return payload.get("test_sha256", "")
    return payload.get("evaluation_config", {}).get("test_file_sha256", "")


def record_count(payload: dict, source: str) -> int:
    if source == "public":
        return payload.get("num_records", 0)
    return payload.get("num_test_records", 0)


def extract_metrics(spec: dict, result_sets: dict) -> dict:
    payload = result_sets[spec["result_source"]]
    if spec["result_source"] == "public":
        record = payload.get("models", {}).get(spec["result_key"])
        if not record:
            raise ValueError(f"Missing public comparison result: {spec['system']}")
        if "error" in record:
            raise ValueError(f"Failed public comparison result for {spec['system']}: {record['error']}")
        precision = record["overall_precision"]
        recall = record["overall_recall"]
        f1 = record["overall_f1"]
    else:
        models = indexed_trained_results(payload)
        record = models.get(spec["result_key"])
        if record is None and spec.get("legacy_key"):
            record = models.get(spec["legacy_key"])
        if not record:
            raise ValueError(f"Missing benchmark result: {spec['system']}")
        precision = record["overall_precision"]
        recall = record["overall_recall"]
        f1 = record["overall_f1"]
    return {
        "system": spec["system"],
        "source_dataset": spec["source_dataset"],
        "type": spec["type"],
        "f1": f1,
        "precision": precision,
        "recall": recall,
    }


def write_markdown(rows: list, path: Path):
    lines = [
        "| System | Source Dataset | Type | F1 | Precision | Recall |",
        "|---|---|---|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['system']} | {row['source_dataset']} | {row['type']} | "
            f"{row['f1']:.4f} | {row['precision']:.4f} | {row['recall']:.4f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Validate and compile the ten required corrected benchmark systems, "
            "optionally adding the curriculum variant."
        )
    )
    parser.add_argument(
        "--novelty-results",
        default="./benchmark_results/corrected_test_5k/novelty_spacy_presidio/benchmark_results.json",
    )
    parser.add_argument(
        "--direct-results",
        default="./benchmark_results/corrected_test_5k/direct_deberta/benchmark_results.json",
    )
    parser.add_argument(
        "--public-results",
        default="./benchmark_results/corrected_test_5k/public_models.json",
    )
    parser.add_argument(
        "--curriculum-results",
        default=None,
        help="Optional held-out benchmark_results.json for the curriculum-enabled model.",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results/corrected_test_5k",
    )
    args = parser.parse_args()

    result_sets = {
        "novelty": load_json(Path(args.novelty_results)),
        "direct": load_json(Path(args.direct_results)),
        "public": load_json(Path(args.public_results)),
    }
    trained_systems = list(TRAINED_SYSTEMS)
    if args.curriculum_results:
        result_sets["curriculum"] = load_json(Path(args.curriculum_results))
        trained_systems.append(CURRICULUM_SYSTEM)

    hashes = {source: dataset_hash(payload, source) for source, payload in result_sets.items()}
    if not all(hashes.values()) or len(set(hashes.values())) != 1:
        raise ValueError(f"Result files do not share one non-empty test SHA-256: {hashes}")
    counts = {source: record_count(payload, source) for source, payload in result_sets.items()}
    if not all(counts.values()) or len(set(counts.values())) != 1:
        raise ValueError(f"Result files do not share one non-zero record count: {counts}")
    num_records = next(iter(counts.values()))
    if num_records < 5_000:
        raise ValueError(
            f"Comparative paper table requires at least 5,000 records; got {num_records}."
        )

    rows = [
        extract_metrics(spec, result_sets)
        for spec in ORIGINAL_SYSTEMS + trained_systems
    ]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "comparative_results_all_systems.json"
    csv_path = output_dir / "comparative_results_all_systems.csv"
    md_path = output_dir / "comparative_results_all_systems.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "test_file_sha256": next(iter(hashes.values())),
                "num_test_records": num_records,
                "required_original_comparators": len(ORIGINAL_SYSTEMS),
                "trained_models": len(trained_systems),
                "systems": rows,
            },
            f,
            indent=2,
        )
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    write_markdown(rows, md_path)

    print(f"Validated {len(ORIGINAL_SYSTEMS)} original comparators and {len(trained_systems)} trained models.")
    print(f"Test SHA-256: {next(iter(hashes.values()))}")
    print(f"Test records: {num_records:,}")
    print(f"Results table -> {md_path}")
    print(f"JSON results  -> {json_path}")
    print(f"CSV results   -> {csv_path}")


if __name__ == "__main__":
    main()
