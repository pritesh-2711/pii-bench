"""
Loads the consolidated PII dataset, applies:
  - finer_139 source cap (150k records)
  - rare entity type dropping (< 500 B- mentions -> collapsed to O)
  - nvidia_nemotron cleanup: keep PII-bearing rows only, then target 10% share
  - stratified 80/10/10 split by source
  - 1% eval subsets: data/val_1p.jsonl and data/test_1p.jsonl
  - paper benchmark subset: data/test_5k.jsonl (5,000 held-out test records)

Outputs:
  data/train.jsonl
  data/val.jsonl
  data/test.jsonl
  data/val_1p.jsonl      <- ~1,400 records, used for fast intra-training eval
  data/test_1p.jsonl     <- ~1,400 records, used for fast milestone checks
  data/test_5k.jsonl     <- 5,000 records, used for external-system comparison
  data/label_mapping.json

The 1p subsets are stratified by source so entity type distribution matches
the full splits. They are written once here and never regenerated unless you
re-run data preparation. Arrow pre-tokenization targets these subsets, not
the full val/test splits, so pre-tokenization is fast (~seconds).

By default, prepared records keep the compact trainable schema:
tokens, labels, source. Pass --include-text to also write a text field.
"""

import argparse
import json
import random
from collections import Counter, defaultdict
from pathlib import Path

CONSOLIDATED_FILE = Path("./pii_datasets/consolidated/consolidated.jsonl")
OUTPUT_DIR = Path("./data")
FINER_CAP = 150_000
RARE_THRESHOLD = 500
NVIDIA_SOURCE = "nvidia_nemotron"
NVIDIA_KEEP_PII_ONLY = True
NVIDIA_TARGET_SHARE = 0.10
TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
SUBSET_FRACTION = 0.01   # 1% of val/test for fast intra-training eval
PAPER_EVAL_SUBSET_SIZE = 5_000
RANDOM_SEED = 42


def tokens_to_text(tokens: list) -> str:
    return " ".join(str(tok) for tok in tokens)


def normalise_output_record(rec: dict, include_text: bool) -> dict:
    out = {
        "tokens": rec["tokens"],
        "labels": rec["labels"],
        "source": rec["source"],
    }
    if include_text:
        out["text"] = rec.get("text") or tokens_to_text(rec["tokens"])
    return out


def normalise_output_records(records: list, include_text: bool) -> list:
    return [normalise_output_record(rec, include_text) for rec in records]


# ---------------------------------------------------------------------------
# Load
# ---------------------------------------------------------------------------

def load_consolidated(path: Path) -> list:
    records = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if line:
                records.append(json.loads(line))
    print(f"Loaded {len(records):,} records from {path}")
    return records


# ---------------------------------------------------------------------------
# Finer-139 cap
# ---------------------------------------------------------------------------

def cap_finer(records: list, cap: int, seed: int) -> list:
    finer = [r for r in records if r["source"] == "finer_139"]
    rest = [r for r in records if r["source"] != "finer_139"]
    rng = random.Random(seed)
    if len(finer) > cap:
        finer = rng.sample(finer, cap)
        print(f"finer_139 capped: {len(finer):,} records kept (was {len(records) - len(rest):,})")
    else:
        print(f"finer_139 under cap: {len(finer):,} records (no capping needed)")
    return rest + finer


# ---------------------------------------------------------------------------
# Rare type dropping
# ---------------------------------------------------------------------------

def drop_rare_entities(records: list, threshold: int) -> tuple:
    mention_counts = Counter()
    for rec in records:
        for lbl in rec["labels"]:
            if lbl.startswith("B-"):
                mention_counts[lbl[2:]] += 1

    dropped_types = {t for t, c in mention_counts.items() if c < threshold}
    kept_types = {t for t, c in mention_counts.items() if c >= threshold}

    print("\nEntity type mention counts (before dropping):")
    for etype, count in sorted(mention_counts.items(), key=lambda x: -x[1]):
        status = "KEEP" if count >= threshold else "DROP"
        print(f"  [{status}] {etype:<35} {count:>10,}")

    if dropped_types:
        print(f"\nDropping {len(dropped_types)} rare entity type(s): {sorted(dropped_types)}")
        updated = []
        for rec in records:
            new_labels = []
            for lbl in rec["labels"]:
                if lbl == "O":
                    new_labels.append("O")
                elif lbl.startswith("B-") or lbl.startswith("I-"):
                    etype = lbl[2:]
                    new_labels.append("O" if etype in dropped_types else lbl)
                else:
                    new_labels.append("O")
            updated_rec = dict(rec)
            updated_rec["labels"] = new_labels
            updated.append(updated_rec)
        records = updated
    else:
        print("\nNo rare entity types to drop.")

    return records, sorted(kept_types), sorted(dropped_types)


# ---------------------------------------------------------------------------
# Nemotron handling
# ---------------------------------------------------------------------------

def has_entity(rec: dict) -> bool:
    return any(lbl != "O" for lbl in rec["labels"])


def prepare_nvidia_share(records: list, target_share: float, seed: int) -> list:
    """
    Keep all PII-bearing Nemotron rows and sample the non-Nemotron pool so
    Nemotron contributes the configured share of every source-stratified split.

    Because stratified_split applies the same 80/10/10 ratios per source, a
    10% source share here remains ~10% in train, val, and test.
    """
    if not 0 < target_share < 1:
        return records

    nvidia = [r for r in records if r["source"] == NVIDIA_SOURCE]
    rest = [r for r in records if r["source"] != NVIDIA_SOURCE]

    if NVIDIA_KEEP_PII_ONLY:
        before = len(nvidia)
        nvidia = [r for r in nvidia if has_entity(r)]
        print(
            f"{NVIDIA_SOURCE}: kept {len(nvidia):,} PII-bearing records "
            f"(dropped {before - len(nvidia):,} all-O records after rare-label cleanup)"
        )

    if not nvidia:
        print(f"{NVIDIA_SOURCE}: no PII-bearing records found; skipping 10% share balancing.")
        return rest

    target_rest_count = int(round(len(nvidia) * (1.0 - target_share) / target_share))
    rng = random.Random(seed)

    if len(rest) > target_rest_count:
        rest = rng.sample(rest, target_rest_count)
        action = "sampled"
    else:
        action = "kept"

    combined = rest + nvidia
    actual_share = len(nvidia) / len(combined)
    print(
        f"{NVIDIA_SOURCE}: {action} non-Nemotron records to {len(rest):,}; "
        f"Nemotron share = {len(nvidia):,}/{len(combined):,} ({actual_share:.2%})"
    )
    return combined


# ---------------------------------------------------------------------------
# Stratified split by source
# ---------------------------------------------------------------------------

def stratified_split(records: list, train_ratio: float, val_ratio: float,
                     seed: int) -> tuple:
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for rec in records:
        by_source[rec["source"]].append(rec)

    train, val, test = [], [], []
    print("\nStratified split by source:")
    print(f"  {'Source':<30} {'Total':>8} {'Train':>8} {'Val':>8} {'Test':>8}")
    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")

    for source, recs in sorted(by_source.items()):
        rng.shuffle(recs)
        n = len(recs)
        n_train = int(n * train_ratio)
        n_val = int(n * val_ratio)
        src_train = recs[:n_train]
        src_val = recs[n_train:n_train + n_val]
        src_test = recs[n_train + n_val:]
        train.extend(src_train)
        val.extend(src_val)
        test.extend(src_test)
        print(f"  {source:<30} {n:>8,} {len(src_train):>8,} {len(src_val):>8,} {len(src_test):>8,}")

    print(f"  {'-'*30} {'-'*8} {'-'*8} {'-'*8} {'-'*8}")
    print(f"  {'TOTAL':<30} {len(train)+len(val)+len(test):>8,} {len(train):>8,} {len(val):>8,} {len(test):>8,}")

    rng.shuffle(train)
    rng.shuffle(val)
    rng.shuffle(test)

    return train, val, test


# ---------------------------------------------------------------------------
# Stratified subset (1% of a split, preserving source distribution)
# ---------------------------------------------------------------------------

def make_stratified_subset(records: list, fraction: float, seed: int) -> list:
    """
    Sample `fraction` of records while preserving per-source proportions.
    Guarantees at least 1 record per source.
    Used to produce val_1p.jsonl and test_1p.jsonl.
    """
    rng = random.Random(seed)
    by_source = defaultdict(list)
    for rec in records:
        by_source[rec["source"]].append(rec)

    subset = []
    for source, recs in sorted(by_source.items()):
        n = max(1, int(len(recs) * fraction))
        subset.extend(rng.sample(recs, min(n, len(recs))))

    rng.shuffle(subset)
    return subset


def make_stratified_subset_by_size(records: list, target_size: int, seed: int) -> list:
    """
    Sample an exact target number of records while preserving source ratios.

    Allocation uses largest remainders after proportional allocation and
    guarantees representation of every source when the target permits it.
    """
    if not records:
        return []
    if target_size <= 0:
        raise ValueError("target_size must be greater than zero")
    if target_size >= len(records):
        return list(records)

    rng = random.Random(seed)
    by_source = defaultdict(list)
    for rec in records:
        by_source[rec["source"]].append(rec)

    if target_size < len(by_source):
        raise ValueError("target_size must be at least the number of sources")

    total = len(records)
    quota = {
        source: target_size * len(recs) / total
        for source, recs in by_source.items()
    }
    allocation = {
        source: min(len(recs), max(1, int(quota[source])))
        for source, recs in by_source.items()
    }

    while sum(allocation.values()) < target_size:
        candidates = [
            source for source, recs in by_source.items()
            if allocation[source] < len(recs)
        ]
        source = max(
            candidates,
            key=lambda s: (quota[s] - allocation[s], len(by_source[s]), s),
        )
        allocation[source] += 1

    while sum(allocation.values()) > target_size:
        candidates = [source for source in by_source if allocation[source] > 1]
        source = min(
            candidates,
            key=lambda s: (quota[s] - allocation[s], len(by_source[s]), s),
        )
        allocation[source] -= 1

    subset = []
    for source in sorted(by_source):
        subset.extend(rng.sample(by_source[source], allocation[source]))
    rng.shuffle(subset)
    return subset


# ---------------------------------------------------------------------------
# Label mapping
# ---------------------------------------------------------------------------

def build_label_mapping(kept_types: list) -> tuple:
    labels = ["O"]
    for etype in sorted(kept_types):
        labels.append(f"B-{etype}")
        labels.append(f"I-{etype}")
    label2id = {lbl: idx for idx, lbl in enumerate(labels)}
    id2label = {idx: lbl for lbl, idx in label2id.items()}
    return labels, label2id, id2label


# ---------------------------------------------------------------------------
# Save
# ---------------------------------------------------------------------------

def save_split(records: list, path: Path):
    with open(path, "w") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    size_mb = path.stat().st_size / 1e6
    print(f"  Saved {len(records):,} records -> {path} ({size_mb:.1f} MB)")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def prepare(include_text: bool = False):
    random.seed(RANDOM_SEED)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("PII DATA PREPARATION")
    print("=" * 60)

    # 1. Load
    records = load_consolidated(CONSOLIDATED_FILE)
    if include_text:
        with_source_text = sum(1 for rec in records if rec.get("text"))
        if with_source_text == 0:
            print(
                "WARNING: --include-text requested, but consolidated records "
                "do not contain text. Falling back to ' '.join(tokens). "
                "Re-run consolidation with --include-text to preserve "
                "source-native text where available."
            )
        else:
            print(f"Text available in {with_source_text:,}/{len(records):,} consolidated records.")

    # 2. Cap finer_139
    print(f"\nCapping finer_139 to {FINER_CAP:,} records ...")
    records = cap_finer(records, FINER_CAP, RANDOM_SEED)
    print(f"Total after cap: {len(records):,}")

    # 3. Drop rare entity types
    records, kept_types, dropped_types = drop_rare_entities(records, RARE_THRESHOLD)

    # 4. Keep Nemotron positives and balance its source share.
    print(f"\nBalancing {NVIDIA_SOURCE} to {NVIDIA_TARGET_SHARE:.0%} source share ...")
    records = prepare_nvidia_share(records, NVIDIA_TARGET_SHARE, RANDOM_SEED)
    print(f"Total after {NVIDIA_SOURCE} balancing: {len(records):,}")

    # 5. Build label mapping
    labels, label2id, id2label = build_label_mapping(kept_types)
    print(f"\nFinal label set ({len(labels)} labels including O):")
    for lbl in labels:
        print(f"  {lbl}")

    # 6. Stratified split
    train, val, test = stratified_split(records, TRAIN_RATIO, VAL_RATIO, RANDOM_SEED)
    train = normalise_output_records(train, include_text)
    val = normalise_output_records(val, include_text)
    test = normalise_output_records(test, include_text)

    # 7. Save full splits
    print("\nSaving splits ...")
    if include_text:
        print("  Including text field in prepared records.")
    save_split(train, OUTPUT_DIR / "train.jsonl")
    save_split(val,   OUTPUT_DIR / "val.jsonl")
    save_split(test,  OUTPUT_DIR / "test.jsonl")

    # 8. Save 1% eval subsets (stratified by source)
    print(f"\nCreating {SUBSET_FRACTION:.0%} eval subsets ...")
    val_1p  = make_stratified_subset(val,  SUBSET_FRACTION, RANDOM_SEED)
    test_1p = make_stratified_subset(test, SUBSET_FRACTION, RANDOM_SEED)
    save_split(val_1p,  OUTPUT_DIR / "val_1p.jsonl")
    save_split(test_1p, OUTPUT_DIR / "test_1p.jsonl")
    print(
        f"  val_1p  : {len(val_1p):,} records "
        f"({len(val_1p)/len(val)*100:.1f}% of val)"
    )
    print(
        f"  test_1p : {len(test_1p):,} records "
        f"({len(test_1p)/len(test)*100:.1f}% of test)"
    )

    # 9. Save an exact-size held-out benchmark subset for paper comparisons.
    print(f"\nCreating {PAPER_EVAL_SUBSET_SIZE:,}-record paper benchmark subset ...")
    test_5k = make_stratified_subset_by_size(test, PAPER_EVAL_SUBSET_SIZE, RANDOM_SEED)
    save_split(test_5k, OUTPUT_DIR / "test_5k.jsonl")
    print(
        f"  test_5k : {len(test_5k):,} records "
        f"({len(test_5k)/len(test)*100:.1f}% of test)"
    )

    # 10. Save label mapping
    mapping = {
        "labels": labels,
        "label2id": label2id,
        "id2label": {str(k): v for k, v in id2label.items()},
        "kept_entity_types": kept_types,
        "dropped_entity_types": dropped_types,
        "num_labels": len(labels),
    }
    label_path = OUTPUT_DIR / "label_mapping.json"
    with open(label_path, "w") as f:
        json.dump(mapping, f, indent=2)
    print(f"  Label mapping -> {label_path}")

    print("\n" + "=" * 60)
    print("DATA PREPARATION COMPLETE")
    print("=" * 60)
    print(f"  Entity types kept   : {len(kept_types)}")
    print(f"  Entity types dropped: {len(dropped_types)}")
    print(f"  Total labels        : {len(labels)}")
    print(f"  Train records       : {len(train):,}")
    print(f"  Val records         : {len(val):,}")
    print(f"  Test records        : {len(test):,}")
    print(f"  val_1p records      : {len(val_1p):,}  <- fast intra-training eval")
    print(f"  test_1p records     : {len(test_1p):,}  <- fast milestone checks")
    print(f"  test_5k records     : {len(test_5k):,}  <- paper benchmark comparison")

    return mapping


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare train/val/test PIIBench JSONL splits"
    )
    parser.add_argument("--include-text", action="store_true",
                        help="Include a text field in each prepared record")
    args = parser.parse_args()
    prepare(include_text=args.include_text)
