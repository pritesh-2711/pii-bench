"""
End-to-end data pipeline before training PII detection model.

Steps:
    1. Download datasets from HuggingFace -> pii_datasets/
    2. Consolidate all sources into a single JSONL -> pii_datasets/consolidated/
    3. Prepare train/val/test splits + label mapping -> data/

Usage:
    python run_data_pipeline.py

    # Skip steps you've already run:
    python run_data_pipeline.py --skip-download
    python run_data_pipeline.py --skip-download --skip-consolidate

    # Override directories:
    python run_data_pipeline.py \
        --raw-data-dir ./pii_datasets \
        --consolidated-dir ./pii_datasets/consolidated \
        --output-dir ./data
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from download_datasets import main as download_datasets
from consolidate_pii_datasets import main as consolidate_datasets
from data_preparation import prepare as prepare_data


def run_pipeline(
    raw_data_dir: str = "./pii_datasets",
    consolidated_dir: str = "./pii_datasets/consolidated",
    output_dir: str = "./data",
    skip_download: bool = False,
    skip_consolidate: bool = False,
):
    print("=" * 80)
    print("PII DETECTION — DATA PIPELINE")
    print("=" * 80)

    raw_data_path = Path(raw_data_dir)
    consolidated_path = Path(consolidated_dir)
    output_path = Path(output_dir)

    # ------------------------------------------------------------------
    # Step 1: Download datasets
    # ------------------------------------------------------------------
    if not skip_download:
        print("\n[STEP 1/3] Downloading datasets")
        print("-" * 80)
        download_datasets(output_dir=str(raw_data_path))
        print(f"\nDownload complete. Raw datasets saved to: {raw_data_path}")
    else:
        print("\n[STEP 1/3] Skipping download (using existing data in "
              f"{raw_data_path})")

    # ------------------------------------------------------------------
    # Step 2: Consolidate datasets
    # ------------------------------------------------------------------
    if not skip_consolidate:
        print("\n[STEP 2/3] Consolidating datasets")
        print("-" * 80)
        consolidate_datasets(
            data_dir=raw_data_path,
            output_dir=consolidated_path,
        )
        print(f"\nConsolidation complete. Output: {consolidated_path}")
    else:
        print("\n[STEP 2/3] Skipping consolidation (using existing data in "
              f"{consolidated_path})")

    # ------------------------------------------------------------------
    # Step 3: Data preparation
    # ------------------------------------------------------------------
    print("\n[STEP 3/3] Preparing train/val/test splits")
    print("-" * 80)

    # data_preparation reads CONSOLIDATED_FILE and writes to OUTPUT_DIR,
    # both of which are module-level constants. Override them at runtime
    # if the caller passed non-default paths.
    import data_preparation as dp
    dp.CONSOLIDATED_FILE = consolidated_path / "consolidated.jsonl"
    dp.OUTPUT_DIR = output_path

    mapping = prepare_data()
    print(f"\nData preparation complete. Splits saved to: {output_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("DATA PIPELINE COMPLETE")
    print("=" * 80)
    print(f"  Raw datasets      : {raw_data_path}")
    print(f"  Consolidated JSONL: {consolidated_path / 'consolidated.jsonl'}")
    print(f"  Train split       : {output_path / 'train.jsonl'}")
    print(f"  Val split         : {output_path / 'val.jsonl'}")
    print(f"  Test split        : {output_path / 'test.jsonl'}")
    print(f"  Paper eval subset : {output_path / 'test_5k.jsonl'}")
    print(f"  Label mapping     : {output_path / 'label_mapping.json'}")
    print(f"  Entity types kept : {len(mapping['kept_entity_types'])}")
    print(f"  Total labels      : {mapping['num_labels']}")
    print("\nNext step: python src/train.py")


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end data pipeline for PII detection model"
    )
    parser.add_argument(
        "--raw-data-dir",
        type=str,
        default="./pii_datasets",
        help="Directory to download raw datasets into (default: ./pii_datasets)",
    )
    parser.add_argument(
        "--consolidated-dir",
        type=str,
        default="./pii_datasets/consolidated",
        help="Directory to write consolidated JSONL (default: ./pii_datasets/consolidated)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./data",
        help="Directory to write train/val/test splits (default: ./data)",
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip dataset download and use existing files in --raw-data-dir",
    )
    parser.add_argument(
        "--skip-consolidate",
        action="store_true",
        help="Skip consolidation and use existing consolidated.jsonl in --consolidated-dir",
    )

    args = parser.parse_args()

    run_pipeline(
        raw_data_dir=args.raw_data_dir,
        consolidated_dir=args.consolidated_dir,
        output_dir=args.output_dir,
        skip_download=args.skip_download,
        skip_consolidate=args.skip_consolidate,
    )


if __name__ == "__main__":
    main()
