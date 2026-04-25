"""
Two-pass streaming data preparation.

Pass 1: Count records per source + entity mention counts (to identify rare types).
Pass 2: Stream through consolidated.jsonl, apply filtering, write splits.

Does not load the full dataset into memory.

Usage:
    python stream_prepare.py \
        --consolidated /path/to/consolidated.jsonl \
        --output-dir   /path/to/splits
"""

import json
import random
import argparse
from pathlib import Path
from collections import Counter, defaultdict

FINER_CAP = 150_000
RARE_THRESHOLD = 500
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SUBSET_FRACTION = 0.01
RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Pass 1: count records per source + entity mentions
# ---------------------------------------------------------------------------

def pass1(consolidated_path: Path):
    source_counts = Counter()
    entity_counts = Counter()  # entity_type -> count of B- mentions

    with open(consolidated_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec["source"]
            source_counts[src] += 1
            for lbl in rec["labels"]:
                if lbl.startswith("B-"):
                    entity_counts[lbl[2:]] += 1

    return source_counts, entity_counts


# ---------------------------------------------------------------------------
# Determine entity types to keep
# ---------------------------------------------------------------------------

def get_kept_dropped(entity_counts: Counter, threshold: int):
    kept = sorted(t for t, c in entity_counts.items() if c >= threshold)
    dropped = sorted(t for t, c in entity_counts.items() if c < threshold)
    return kept, dropped


# ---------------------------------------------------------------------------
# Build label mapping
# ---------------------------------------------------------------------------

def build_label_mapping(kept_types):
    labels = ["O"]
    for etype in sorted(kept_types):
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    return labels, label2id, id2label


# ---------------------------------------------------------------------------
# Pass 2: stream + split + filter
# ---------------------------------------------------------------------------

def pass2(consolidated_path: Path, source_counts: Counter,
          dropped_types: set, output_dir: Path, seed: int):
    """
    Assign each record to train/val/test based on a deterministic index per source.
    Applies finer_139 cap and rare entity dropping.
    """
    rng = random.Random(seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Pre-compute split indices per source:
    # shuffle indices [0..N-1], first 80% train, next 10% val, rest test
    source_indices = {}
    finer_cap_indices = set()

    for src, total in source_counts.items():
        idxs = list(range(total))
        rng.shuffle(idxs)
        if src == "finer_139" and total > FINER_CAP:
            idxs = idxs[:FINER_CAP]
            finer_cap_indices = set(idxs)
        n_train = int(len(idxs) * TRAIN_RATIO)
        n_val = int(len(idxs) * VAL_RATIO)
        source_indices[src] = {
            "train": set(idxs[:n_train]),
            "val": set(idxs[n_train:n_train + n_val]),
            "test": set(idxs[n_train + n_val:]),
            "cap": set(idxs),  # for finer cap
        }

    # Open output files
    train_f = open(output_dir / "train.jsonl", "w")
    val_f   = open(output_dir / "val.jsonl", "w")
    test_f  = open(output_dir / "test.jsonl", "w")

    counts = {"train": 0, "val": 0, "test": 0}
    source_record_idx = defaultdict(int)  # current record index per source

    with open(consolidated_path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            src = rec["source"]
            idx = source_record_idx[src]
            source_record_idx[src] += 1

            # finer_139 cap
            if src == "finer_139" and idx not in source_indices[src]["cap"]:
                continue

            # Apply rare entity filtering
            new_labels = []
            for lbl in rec["labels"]:
                if lbl == "O":
                    new_labels.append("O")
                elif lbl.startswith("B-") or lbl.startswith("I-"):
                    etype = lbl[2:]
                    new_labels.append("O" if etype in dropped_types else lbl)
                else:
                    new_labels.append("O")
            rec["labels"] = new_labels

            # Assign to split
            split_idx = source_indices[src]
            if idx in split_idx["train"]:
                train_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["train"] += 1
            elif idx in split_idx["val"]:
                val_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["val"] += 1
            elif idx in split_idx["test"]:
                test_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
                counts["test"] += 1

    train_f.close()
    val_f.close()
    test_f.close()

    return counts


# ---------------------------------------------------------------------------
# Build 1% subset (reservoir sampling from a JSONL file)
# ---------------------------------------------------------------------------

def reservoir_sample(jsonl_path: Path, fraction: float, seed: int) -> list:
    """Reservoir sampling — reads the file once, returns fraction of records."""
    rng = random.Random(seed)
    reservoir = []
    n_seen = 0
    k = None  # will be set after first pass to determine reservoir size

    # First pass to count
    count = 0
    with open(jsonl_path) as f:
        for line in f:
            if line.strip():
                count += 1
    k = max(1, int(count * fraction))

    with open(jsonl_path) as f:
        for i, line in enumerate(f):
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            if len(reservoir) < k:
                reservoir.append(rec)
            else:
                j = rng.randint(0, i)
                if j < k:
                    reservoir[j] = rec
    rng.shuffle(reservoir)
    return reservoir


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(consolidated_path: str, output_dir: str):
    consolidated_path = Path(consolidated_path)
    output_dir = Path(output_dir)

    print("=" * 60)
    print("PASS 1: Counting records and entity types")
    print("=" * 60)
    source_counts, entity_counts = pass1(consolidated_path)

    print(f"Total records: {sum(source_counts.values()):,}")
    for src, cnt in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<40} {cnt:>10,}")

    kept_types, dropped_types = get_kept_dropped(entity_counts, RARE_THRESHOLD)
    print(f"\nEntity types kept  ({len(kept_types)}): {kept_types}")
    print(f"Entity types dropped ({len(dropped_types)}): {dropped_types}")

    labels, label2id, id2label = build_label_mapping(kept_types)
    print(f"\nTotal BIO labels (including O): {len(labels)}")

    print("\n" + "=" * 60)
    print("PASS 2: Splitting and filtering")
    print("=" * 60)
    counts = pass2(consolidated_path, source_counts, set(dropped_types), output_dir, RANDOM_SEED)
    print(f"  train : {counts['train']:,}")
    print(f"  val   : {counts['val']:,}")
    print(f"  test  : {counts['test']:,}")

    print("\nBuilding 1% eval subsets ...")
    val_1p  = reservoir_sample(output_dir / "val.jsonl",  SUBSET_FRACTION, RANDOM_SEED)
    test_1p = reservoir_sample(output_dir / "test.jsonl", SUBSET_FRACTION, RANDOM_SEED)

    for recs, name in [(val_1p, "val_1p.jsonl"), (test_1p, "test_1p.jsonl")]:
        p = output_dir / name
        with open(p, "w") as f:
            for rec in recs:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        print(f"  {name}: {len(recs):,} records")

    # Save label mapping
    mapping = {
        "labels": labels,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "kept_entity_types": kept_types,
        "dropped_entity_types": dropped_types,
        "num_labels": len(labels),
        "source_counts": dict(source_counts),
        "entity_mention_counts": {t: entity_counts[t] for t in sorted(entity_counts)},
    }
    label_path = output_dir / "label_mapping.json"
    with open(label_path, "w") as f:
        json.dump(mapping, f, indent=2)

    print(f"\nLabel mapping -> {label_path}")
    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Kept   : {len(kept_types)} entity types")
    print(f"  Dropped: {len(dropped_types)} entity types")
    print(f"  Labels : {len(labels)} total (with O, B-, I-)")
    print(f"  Train  : {counts['train']:,}")
    print(f"  Val    : {counts['val']:,}")
    print(f"  Test   : {counts['test']:,}")
    print(f"  val_1p : {len(val_1p):,}")
    print(f"  test_1p: {len(test_1p):,}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--consolidated", type=str, required=True)
    parser.add_argument("--output-dir",   type=str, required=True)
    args = parser.parse_args()
    main(args.consolidated, args.output_dir)
