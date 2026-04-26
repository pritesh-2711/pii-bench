"""
Streaming consolidation pipeline.

Streams each dataset directly from HuggingFace (no raw JSONL saved to disk),
normalises to unified BIO format, and writes consolidated.jsonl + stats.

Usage:
    python stream_consolidate.py --output-dir /path/to/output
"""

import json
import re
import argparse
import sys
from pathlib import Path
from collections import defaultdict, Counter

# Reuse the normalisation logic from the original module
sys.path.insert(0, str(Path(__file__).parent / "src"))
from consolidate_pii_datasets import (
    normalise_label,
    span_to_bio,
    parse_span_field,
    read_nvidia_jsonl,
)


# ---------------------------------------------------------------------------
# HuggingFace import (lazy to avoid import errors if not installed)
# ---------------------------------------------------------------------------

def get_hf():
    from datasets import load_dataset, get_dataset_config_names
    return load_dataset, get_dataset_config_names


# ---------------------------------------------------------------------------
# Inline readers that yield records (memory-efficient)
# ---------------------------------------------------------------------------

def yield_bio_records(rows, token_col, label_col, source, label_names=None):
    for row in rows:
        tokens = row.get(token_col)
        labels = row.get(label_col)
        if not tokens or labels is None:
            continue
        if label_names and labels and isinstance(labels[0], int):
            labels = [label_names[i] if i < len(label_names) else "O" for i in labels]
        labels = [normalise_label(str(l)) for l in labels]
        n = min(len(tokens), len(labels))
        yield {"tokens": tokens[:n], "labels": labels[:n], "source": source}


def yield_fewnerd_records(rows, label_names, source="few_nerd"):
    for row in rows:
        tokens = row.get("tokens")
        ner_tags = row.get("ner_tags")
        if not tokens or ner_tags is None:
            continue
        labels = []
        prev_label = None
        for tag_id in ner_tags:
            if tag_id == 0:
                labels.append("O")
                prev_label = None
            else:
                raw_label = label_names[tag_id] if tag_id < len(label_names) else "other"
                canonical = normalise_label(raw_label)
                if prev_label == canonical:
                    labels.append(f"I-{canonical}")
                else:
                    labels.append(f"B-{canonical}")
                prev_label = canonical
        n = min(len(tokens), len(labels))
        yield {"tokens": tokens[:n], "labels": labels[:n], "source": source}


def yield_finer_records(rows):
    for row in rows:
        tokens = row.get("tokens")
        ner_tags = row.get("ner_tags")
        if not tokens or ner_tags is None:
            continue
        labels = []
        for tag in ner_tags:
            if tag == 0:
                labels.append("O")
            elif tag % 2 == 1:
                labels.append("B-FINANCIAL_ENTITY")
            else:
                labels.append("I-FINANCIAL_ENTITY")
        yield {"tokens": tokens, "labels": labels, "source": "finer_139"}


def yield_span_records(rows, text_col, span_col, source):
    for row in rows:
        text = row.get(text_col, "")
        spans = parse_span_field(row.get(span_col))
        if not text:
            continue
        tokens, labels = span_to_bio(text, spans)
        labels = [normalise_label(l) for l in labels]
        if tokens:
            yield {"tokens": tokens, "labels": labels, "source": source}


def yield_nvidia_records(rows):
    tag_re = re.compile(r'<(\w+)>(.*?)</\1>', re.DOTALL)
    for row in rows:
        text = row.get("text", "")
        spans_raw = row.get("spans")
        if not text:
            continue
        spans = parse_span_field(spans_raw)
        tokens, labels = span_to_bio(text, spans)
        labels = [normalise_label(l) for l in labels]
        has_entities = any(l != "O" for l in labels)
        if not has_entities:
            text_tagged = row.get("text_tagged", "")
            if text_tagged:
                fallback_spans = []
                clean = ""
                remaining = text_tagged
                while remaining:
                    m = re.search(r'<(\w+)>(.*?)</\1>', remaining, re.DOTALL)
                    if not m:
                        clean += remaining
                        break
                    clean += remaining[:m.start()]
                    entity_start = len(clean)
                    entity_text = m.group(2)
                    clean += entity_text
                    entity_end = len(clean)
                    fallback_spans.append({
                        "start": entity_start,
                        "end": entity_end,
                        "type": m.group(1),
                    })
                    remaining = remaining[m.end():]
                if fallback_spans and clean.strip():
                    tokens, labels = span_to_bio(clean, fallback_spans)
                    labels = [normalise_label(l) for l in labels]
        if tokens:
            yield {"tokens": tokens, "labels": labels, "source": "nvidia_nemotron"}


# ---------------------------------------------------------------------------
# Stats collector
# ---------------------------------------------------------------------------

class Stats:
    def __init__(self):
        self.source_counts = Counter()
        self.entity_counts = Counter()  # (source, entity_type) -> count
        self.token_counts = Counter()   # source -> total tokens
        self.total = 0

    def update(self, rec):
        src = rec["source"]
        self.source_counts[src] += 1
        self.total += 1
        self.token_counts[src] += len(rec["tokens"])
        for lbl in rec["labels"]:
            if lbl.startswith("B-"):
                etype = lbl[2:]
                self.entity_counts[(src, etype)] += 1

    def to_dict(self):
        per_source = defaultdict(lambda: {"records": 0, "tokens": 0, "entity_counts": {}})
        for src, count in self.source_counts.items():
            per_source[src]["records"] = count
            per_source[src]["tokens"] = self.token_counts[src]
        for (src, etype), count in self.entity_counts.items():
            per_source[src]["entity_counts"][etype] = count

        global_entity_counts = Counter()
        for (src, etype), count in self.entity_counts.items():
            global_entity_counts[etype] += count

        return {
            "total_records": self.total,
            "per_source": dict(per_source),
            "global_entity_counts": dict(global_entity_counts),
        }


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def main(output_dir: str):
    load_dataset, get_dataset_config_names = get_hf()
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    consolidated_path = output_dir / "consolidated.jsonl"
    stats = Stats()

    print(f"Writing consolidated.jsonl -> {consolidated_path}")

    with open(consolidated_path, "w", encoding="utf-8") as out_f:

        def write_records(gen):
            for rec in gen:
                stats.update(rec)
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            src = list(stats.source_counts.keys())[-1] if stats.source_counts else "?"
            print(f"  {src}: {stats.source_counts.get(src, 0):,} records so far")

        # 1. ai4privacy/pii-masking-400k
        print("[1/10] ai4privacy/pii-masking-400k ...")
        ds = load_dataset("ai4privacy/pii-masking-400k", split="train", streaming=False)
        write_records(yield_bio_records(ds, "mbert_tokens", "mbert_token_classes", "ai4privacy_400k"))

        # 2. ai4privacy/pii-masking-300k
        print("[2/10] ai4privacy/pii-masking-300k ...")
        ds = load_dataset("ai4privacy/pii-masking-300k", split="train", streaming=False)
        write_records(yield_bio_records(ds, "mbert_text_tokens", "mbert_bio_labels", "ai4privacy_300k"))

        # 3. gretelai/synthetic_pii_finance_multilingual
        print("[3/10] gretelai/synthetic_pii_finance_multilingual ...")
        for split in ["train", "test"]:
            try:
                ds = load_dataset("gretelai/synthetic_pii_finance_multilingual", split=split, streaming=False)
                write_records(yield_span_records(ds, "generated_text", "pii_spans", "gretel_finance"))
            except Exception as e:
                print(f"  split {split} skipped: {e}")

        # 4. nvidia/Nemotron-PII
        print("[4/10] nvidia/Nemotron-PII ...")
        ds = load_dataset("nvidia/Nemotron-PII", split="train", streaming=False)
        write_records(yield_nvidia_records(ds))

        # 5. wikiann (en)
        print("[5/10] wikiann (en) ...")
        wikiann_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
        for split in ["train", "validation", "test"]:
            ds = load_dataset("wikiann", "en", split=split, streaming=False)
            write_records(yield_bio_records(ds, "tokens", "ner_tags", "wikiann", label_names=wikiann_labels))

        # 6. Babelscape/multinerd (en only)
        print("[6/10] Babelscape/multinerd (en) ...")
        multinerd_labels = [
            "O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
            "B-ANIM", "I-ANIM", "B-BIO", "I-BIO", "B-CEL", "I-CEL",
            "B-DIS", "I-DIS", "B-EVE", "I-EVE", "B-FOOD", "I-FOOD",
            "B-INST", "I-INST", "B-MEDIA", "I-MEDIA", "B-MYTH", "I-MYTH",
            "B-PLANT", "I-PLANT", "B-TIME", "I-TIME", "B-VEHI", "I-VEHI",
        ]
        ds = load_dataset("Babelscape/multinerd", split="train", verification_mode="no_checks", streaming=False)
        if "lang" in ds.column_names:
            ds = ds.filter(lambda x: x["lang"] == "en")
        write_records(yield_bio_records(ds, "tokens", "ner_tags", "multinerd", label_names=multinerd_labels))

        # 7. DFKI-SLT/few-nerd
        print("[7/10] DFKI-SLT/few-nerd ...")
        fewnerd_labels = ["O", "art", "building", "event", "location",
                          "organization", "other", "person", "product"]
        for split in ["train", "validation", "test"]:
            ds = load_dataset("DFKI-SLT/few-nerd", "supervised", split=split, streaming=False)
            write_records(yield_fewnerd_records(ds, fewnerd_labels))

        # 8. conll2003
        print("[8/10] conll2003 ...")
        conll_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                        "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
        for split in ["train", "validation", "test"]:
            ds = load_dataset("conll2003", split=split, revision="refs/convert/parquet", streaming=False)
            write_records(yield_bio_records(ds, "tokens", "ner_tags", "conll2003", label_names=conll_labels))

        # 9. nlpaueb/finer-139
        print("[9/10] nlpaueb/finer-139 ...")
        for split in ["train", "validation", "test"]:
            ds = load_dataset("nlpaueb/finer-139", split=split, revision="refs/convert/parquet", streaming=False)
            write_records(yield_finer_records(ds))

        # 10. Isotonic/pii-masking-200k
        print("[10/10] Isotonic/pii-masking-200k ...")
        ds_iso = load_dataset("Isotonic/pii-masking-200k", streaming=False)
        split_key = list(ds_iso.keys())[0]
        write_records(yield_bio_records(ds_iso[split_key], "tokenised_text", "bio_labels", "isotonic_pii_200k"))

    # Save stats
    stats_dict = stats.to_dict()
    stats_path = output_dir / "raw_stats.json"
    with open(stats_path, "w") as f:
        json.dump(stats_dict, f, indent=2)

    print(f"\nConsolidation complete.")
    print(f"  Total records : {stats.total:,}")
    print(f"  consolidated  : {consolidated_path} ({consolidated_path.stat().st_size / 1e6:.0f} MB)")
    print(f"  stats         : {stats_path}")

    for src, cnt in sorted(stats.source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<40} {cnt:>10,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output-dir", type=str, required=True)
    args = parser.parse_args()
    main(args.output_dir)
