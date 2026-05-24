"""Compare direct and source-conditioned hierarchical DeBERTa on full test.

The two input result files are produced by run_streaming_model_benchmark.py.
This script validates that they refer to the identical test artifact, then
writes overall, fine-entity and coarse-group comparison tables for reporting.

Group scores are support-weighted means of the stored per-entity F1 scores.
This can be reproduced exactly from the result artifacts without retaining
the record-level predictions used during streamed evaluation.
"""

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from train_novel import COARSE_GROUPS


AVERAGE_KEYS = {"micro avg", "macro avg", "weighted avg"}


def load_result(path: Path) -> dict:
    with open(path, encoding="utf-8") as handle:
        payload = json.load(handle)
    if len(payload.get("systems", [])) != 1:
        raise ValueError(f"Expected one trained-model result in {path}.")
    return payload


def model_row(payload: dict) -> dict:
    row = payload["systems"][0]
    return {
        "system": row["system"],
        "f1": row["overall_f1"],
        "precision": row["overall_precision"],
        "recall": row["overall_recall"],
    }


def validate_shared_evaluation(direct: dict, hierarchical: dict) -> dict:
    direct_config = direct["evaluation_config"]
    hierarchical_config = hierarchical["evaluation_config"]
    direct_hash = direct_config["test_file_sha256"]
    hierarchical_hash = hierarchical_config["test_file_sha256"]
    if direct_hash != hierarchical_hash:
        raise ValueError("Models were evaluated on different test-file hashes.")
    if direct["num_test_records"] != hierarchical["num_test_records"]:
        raise ValueError("Models were evaluated on different record counts.")
    if direct_config.get("metric") != hierarchical_config.get("metric"):
        raise ValueError("Models were evaluated using different metric definitions.")
    return {
        "test_file_sha256": direct_hash,
        "num_test_records": direct["num_test_records"],
        "metric": direct_config["metric"],
        "taxonomy": direct_config.get("taxonomy"),
    }


def per_entity_rows(direct: dict, hierarchical: dict) -> list:
    direct_report = direct["systems"][0]["per_entity"]
    hierarchical_report = hierarchical["systems"][0]["per_entity"]
    entities = sorted(set(direct_report) - AVERAGE_KEYS)
    if entities != sorted(set(hierarchical_report) - AVERAGE_KEYS):
        raise ValueError("Per-entity reports do not cover identical entity types.")

    rows = []
    for entity in entities:
        direct_f1 = direct_report[entity]["f1-score"]
        hierarchical_f1 = hierarchical_report[entity]["f1-score"]
        delta = direct_f1 - hierarchical_f1
        if delta > 0:
            winner = "Direct Fine-tuned DeBERTa"
        elif delta < 0:
            winner = "Source-conditioned Hierarchical DeBERTa"
        else:
            winner = "Tie"
        rows.append(
            {
                "entity": entity,
                "coarse_group": COARSE_GROUPS.get(entity, "MISC"),
                "support": int(direct_report[entity]["support"]),
                "direct_f1": direct_f1,
                "hierarchical_f1": hierarchical_f1,
                "direct_minus_hierarchical_f1": delta,
                "winner": winner,
            }
        )
    return rows


def group_rows(entity_rows: list) -> list:
    groups = {}
    for row in entity_rows:
        data = groups.setdefault(
            row["coarse_group"],
            {
                "coarse_group": row["coarse_group"],
                "support": 0,
                "entity_types": 0,
                "direct_weighted_f1_total": 0.0,
                "hierarchical_weighted_f1_total": 0.0,
                "direct_entity_wins": 0,
                "hierarchical_entity_wins": 0,
                "ties": 0,
            },
        )
        data["support"] += row["support"]
        data["entity_types"] += 1
        data["direct_weighted_f1_total"] += row["support"] * row["direct_f1"]
        data["hierarchical_weighted_f1_total"] += (
            row["support"] * row["hierarchical_f1"]
        )
        if row["winner"] == "Direct Fine-tuned DeBERTa":
            data["direct_entity_wins"] += 1
        elif row["winner"] == "Source-conditioned Hierarchical DeBERTa":
            data["hierarchical_entity_wins"] += 1
        else:
            data["ties"] += 1

    rows = []
    for data in groups.values():
        direct_f1 = data.pop("direct_weighted_f1_total") / data["support"]
        hierarchical_f1 = (
            data.pop("hierarchical_weighted_f1_total") / data["support"]
        )
        rows.append(
            {
                **data,
                "direct_support_weighted_entity_f1": direct_f1,
                "hierarchical_support_weighted_entity_f1": hierarchical_f1,
                "direct_minus_hierarchical_f1": direct_f1 - hierarchical_f1,
                "winner": (
                    "Direct Fine-tuned DeBERTa"
                    if direct_f1 > hierarchical_f1
                    else "Source-conditioned Hierarchical DeBERTa"
                    if hierarchical_f1 > direct_f1
                    else "Tie"
                ),
            }
        )
    return sorted(rows, key=lambda row: row["support"], reverse=True)


def write_csv(path: Path, rows: list):
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(value: float) -> str:
    return f"{value:.4f}"


def write_markdown(path: Path, summary: dict):
    overall = summary["overall"]
    groups = summary["groups"]
    entities = summary["entities"]
    top_direct = sorted(
        entities, key=lambda row: row["direct_minus_hierarchical_f1"], reverse=True
    )[:15]
    top_hierarchical = sorted(
        entities, key=lambda row: row["direct_minus_hierarchical_f1"]
    )[:15]

    lines = [
        "# Full Test Direct vs Source-Conditioned Hierarchical Analysis",
        "",
        "## Evaluation Integrity",
        "",
        f"- Records: `{summary['evaluation']['num_test_records']:,}`",
        f"- Test SHA-256: `{summary['evaluation']['test_file_sha256']}`",
        f"- Metric: `{summary['evaluation']['metric']}`",
        "",
        "## Overall Results",
        "",
        "| Model | F1 | Precision | Recall |",
        "|---|---:|---:|---:|",
    ]
    for row in overall:
        lines.append(
            f"| {row['system']} | {fmt(row['f1'])} | "
            f"{fmt(row['precision'])} | {fmt(row['recall'])} |"
        )

    delta = summary["overall_delta"]
    lines.extend(
        [
            "",
            "Direct DeBERTa minus source-conditioned hierarchical DeBERTa:",
            "",
            f"- F1: `{delta['f1']:+.4f}`",
            f"- Precision: `{delta['precision']:+.4f}`",
            f"- Recall: `{delta['recall']:+.4f}`",
            "",
            "## Coarse Group Analysis",
            "",
            "Group scores are support-weighted means of fine-entity F1 scores.",
            "",
            "| Group | Support | Types | Direct F1 | SC+H F1 | Delta | Fine-Type Wins (Direct / SC+H) |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in groups:
        lines.append(
            f"| {row['coarse_group']} | {row['support']:,} | "
            f"{row['entity_types']} | "
            f"{fmt(row['direct_support_weighted_entity_f1'])} | "
            f"{fmt(row['hierarchical_support_weighted_entity_f1'])} | "
            f"{row['direct_minus_hierarchical_f1']:+.4f} | "
            f"{row['direct_entity_wins']} / {row['hierarchical_entity_wins']} |"
        )

    lines.extend(
        [
            "",
            "## Largest Direct DeBERTa Advantages",
            "",
            "| Entity | Group | Support | Direct F1 | SC+H F1 | Delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_direct:
        lines.append(
            f"| {row['entity']} | {row['coarse_group']} | {row['support']:,} | "
            f"{fmt(row['direct_f1'])} | {fmt(row['hierarchical_f1'])} | "
            f"{row['direct_minus_hierarchical_f1']:+.4f} |"
        )
    lines.extend(
        [
            "",
            "## Largest Source-Conditioned Hierarchical Advantages",
            "",
            "| Entity | Group | Support | Direct F1 | SC+H F1 | SC+H Delta |",
            "|---|---|---:|---:|---:|---:|",
        ]
    )
    for row in top_hierarchical:
        lines.append(
            f"| {row['entity']} | {row['coarse_group']} | {row['support']:,} | "
            f"{fmt(row['direct_f1'])} | {fmt(row['hierarchical_f1'])} | "
            f"{-row['direct_minus_hierarchical_f1']:+.4f} |"
        )

    lines.extend(
        [
            "",
            "## Fine-Entity Winner Count",
            "",
            f"- Direct Fine-tuned DeBERTa wins: `{summary['entity_win_counts']['Direct Fine-tuned DeBERTa']}` entity types",
            f"- Source-conditioned Hierarchical DeBERTa wins: `{summary['entity_win_counts']['Source-conditioned Hierarchical DeBERTa']}` entity types",
            f"- Ties: `{summary['entity_win_counts']['Tie']}` entity types",
            "",
            "The direct model wins overall and in every coarse group, while the hierarchical "
            "model retains improvements for a minority of fine entity types. The full entity "
            "table is written to `direct_vs_source_conditioned_hierarchical_entities.csv`.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--direct-results",
        default="./benchmark_results/full_test/direct_deberta/benchmark_results.json",
    )
    parser.add_argument(
        "--hierarchical-results",
        default="./benchmark_results/full_test/source_conditioned_hierarchical/benchmark_results.json",
    )
    parser.add_argument(
        "--output-dir",
        default="./benchmark_results/full_test",
    )
    args = parser.parse_args()

    direct = load_result(Path(args.direct_results))
    hierarchical = load_result(Path(args.hierarchical_results))
    evaluation = validate_shared_evaluation(direct, hierarchical)
    overall = [model_row(direct), model_row(hierarchical)]
    entity_rows = per_entity_rows(direct, hierarchical)
    groups = group_rows(entity_rows)
    winner_counts = {
        "Direct Fine-tuned DeBERTa": sum(
            row["winner"] == "Direct Fine-tuned DeBERTa" for row in entity_rows
        ),
        "Source-conditioned Hierarchical DeBERTa": sum(
            row["winner"] == "Source-conditioned Hierarchical DeBERTa"
            for row in entity_rows
        ),
        "Tie": sum(row["winner"] == "Tie" for row in entity_rows),
    }
    summary = {
        "evaluation": evaluation,
        "overall": overall,
        "overall_delta": {
            metric: overall[0][metric] - overall[1][metric]
            for metric in ("f1", "precision", "recall")
        },
        "group_metric_definition": "support-weighted mean fine-entity F1 within each coarse group",
        "entity_win_counts": winner_counts,
        "groups": groups,
        "entities": entity_rows,
    }

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    json_path = output_dir / "direct_vs_source_conditioned_hierarchical_analysis.json"
    entity_csv = output_dir / "direct_vs_source_conditioned_hierarchical_entities.csv"
    group_csv = output_dir / "direct_vs_source_conditioned_hierarchical_groups.csv"
    md_path = output_dir / "direct_vs_source_conditioned_hierarchical_analysis.md"
    json_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    write_csv(entity_csv, entity_rows)
    write_csv(group_csv, groups)
    write_markdown(md_path, summary)

    print(f"Validated full-test comparison on {evaluation['num_test_records']:,} records.")
    print(f"Test SHA-256: {evaluation['test_file_sha256']}")
    print(
        "Overall F1: "
        f"direct={overall[0]['f1']:.4f}, "
        f"source-conditioned hierarchical={overall[1]['f1']:.4f}, "
        f"delta={summary['overall_delta']['f1']:+.4f}"
    )
    print(
        "Entity-type wins: "
        f"direct={winner_counts['Direct Fine-tuned DeBERTa']}, "
        f"source-conditioned hierarchical="
        f"{winner_counts['Source-conditioned Hierarchical DeBERTa']}, "
        f"ties={winner_counts['Tie']}"
    )
    print(f"Analysis JSON -> {json_path}")
    print(f"Analysis report -> {md_path}")


if __name__ == "__main__":
    main()
