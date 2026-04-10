"""
End-to-end training pipeline for PII detection model.

Steps:
    1. Download microsoft/deberta-v3-base weights -> models/deberta-v3-base/
    2. Fine-tune on PII NER data -> models/best_model/

Usage:
    python run_training_pipeline.py

    # Skip model download if already done:
    python run_training_pipeline.py --skip-download

    # Fix eval OOM (most common first run issue):
    python run_training_pipeline.py --eval-accumulation-steps 1

    # Pre-tokenize once then reuse across runs:
    python run_training_pipeline.py --pretokenize-only
    python run_training_pipeline.py --use-pretokenized

    # Custom training hyperparameters:
    python run_training_pipeline.py \
        --batch-size 16 \
        --grad-accum 4 \
        --max-length 512 \
        --epochs 10 \
        --eval-steps 2000 \
        --save-steps 2000 \
        --logging-steps 50

    # Full options:
    python run_training_pipeline.py --help
"""

import sys
import argparse
from pathlib import Path

sys.path.append(str(Path(__file__).parent / "src"))

from download_model import main as download_model
from train import PIITrainer, PRETOKENIZED_DIR


def run_pipeline(
    skip_download: bool = False,
    batch_size: int = 16,
    eval_batch_size: int = 32,
    grad_accum: int = 4,
    learning_rate: float = 2e-5,
    num_epochs: int = 10,
    max_steps: int = -1,
    max_length: int = 512,
    warmup_ratio: float = 0.06,
    weight_decay: float = 0.01,
    early_stopping_patience: int = 3,
    eval_steps: int = 2000,
    save_steps: int = 2000,
    logging_steps: int = 50,
    use_gradient_checkpointing: bool = False,
    resume_from_checkpoint: bool = False,
    fp16_full_eval: bool = False,
    torch_compile: bool = False,
    eval_accumulation_steps: int = 1,
    prediction_loss_only: bool = False,
    pretokenize_only: bool = False,
    use_pretokenized: bool = False,
    pretokenized_dir: Path = PRETOKENIZED_DIR,
    pretokenize_num_proc: int = 4,
    train_sample_fraction: float = 0.0,
    skip_final_eval: bool = False,
):
    print("=" * 80)
    print("PII DETECTION — TRAINING PIPELINE")
    print("=" * 80)

    # ------------------------------------------------------------------
    # Step 1: Download base model
    # ------------------------------------------------------------------
    if not skip_download:
        print("\n[STEP 1/2] Downloading microsoft/deberta-v3-base")
        print("-" * 80)
        download_model()
        print("\nModel download complete.")
    else:
        print("\n[STEP 1/2] Skipping model download (using existing weights in "
              "./models/deberta-v3-base)")

    # ------------------------------------------------------------------
    # Step 2: Fine-tune
    # ------------------------------------------------------------------
    print("\n[STEP 2/2] Fine-tuning on PII NER data")
    print("-" * 80)

    trainer = PIITrainer(
        batch_size=batch_size,
        eval_batch_size=eval_batch_size,
        grad_accum=grad_accum,
        learning_rate=learning_rate,
        num_epochs=num_epochs,
        max_steps=max_steps,
        max_length=max_length,
        warmup_ratio=warmup_ratio,
        weight_decay=weight_decay,
        early_stopping_patience=early_stopping_patience,
        eval_steps=eval_steps,
        save_steps=save_steps,
        logging_steps=logging_steps,
        use_gradient_checkpointing=use_gradient_checkpointing,
        resume_from_checkpoint=resume_from_checkpoint,
        fp16_full_eval=fp16_full_eval,
        torch_compile=torch_compile,
        eval_accumulation_steps=eval_accumulation_steps,
        prediction_loss_only=prediction_loss_only,
        use_pretokenized=use_pretokenized,
        pretokenized_dir=pretokenized_dir,
        train_sample_fraction=train_sample_fraction,
        skip_final_eval=skip_final_eval,
    )

    if pretokenize_only:
        trainer.pretokenize(num_proc=pretokenize_num_proc)
        print("\nPre-tokenization complete. Re-run without --pretokenize-only to train.")
        return {}

    trainer.load_datasets()
    hf_trainer = trainer.train()

    if not skip_final_eval:
        results = trainer.evaluate(hf_trainer)
    else:
        results = {}
        print("\nSkipping final test evaluation (--skip-final-eval set).")
        print("  Best model weights saved to: ./models/best_model")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("TRAINING PIPELINE COMPLETE")
    print("=" * 80)
    print("  Best model        : ./models/best_model")
    if not skip_final_eval:
        print("  Evaluation results: ./models/evaluation_results.json")
        print(f"  Test F1           : {results.get('test_f1', 0):.4f}")
        print(f"  Test Precision    : {results.get('test_precision', 0):.4f}")
        print(f"  Test Recall       : {results.get('test_recall', 0):.4f}")
    print("\nNext step: python src/api.py --model-path ./models/best_model")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="End-to-end training pipeline for PII detection model"
    )
    parser.add_argument(
        "--skip-download",
        action="store_true",
        help="Skip base model download and use existing weights in ./models/deberta-v3-base",
    )
    parser.add_argument("--batch-size",              type=int,   default=16)
    parser.add_argument("--eval-batch-size",          type=int,   default=32,
                        help="Eval/predict batch size (default: 32). Lower if eval OOMs.")
    parser.add_argument("--grad-accum",              type=int,   default=4)
    parser.add_argument("--epochs",                  type=int,   default=10)
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Hard step limit. Overrides --epochs when > 0.",
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Token sequence length (default: 512, use 256 to halve memory/time)",
    )
    parser.add_argument("--lr",                      type=float, default=2e-5)
    parser.add_argument("--warmup-ratio",            type=float, default=0.06)
    parser.add_argument("--weight-decay",            type=float, default=0.01)
    parser.add_argument("--early-stopping-patience", type=int,   default=3)
    parser.add_argument("--eval-steps",              type=int,   default=2000)
    parser.add_argument("--save-steps",              type=int,   default=2000)
    parser.add_argument("--logging-steps",           type=int,   default=50)
    parser.add_argument("--gradient-checkpointing",  action="store_true")
    parser.add_argument(
        "--resume-from-checkpoint",
        action="store_true",
        help="Resume from latest checkpoint in models/checkpoints/",
    )
    parser.add_argument(
        "--fp16-full-eval",
        action="store_true",
        help=(
            "Run evaluation in fp16. Halves eval VRAM. Use only if eval OOMs "
            "after lowering --eval-batch-size. Try --eval-accumulation-steps first."
        ),
    )
    parser.add_argument(
        "--torch-compile",
        action="store_true",
        help=(
            "Enable torch.compile. Only beneficial on A100/H100 + CUDA >= 11.8 "
            "+ PyTorch >= 2.1. Not recommended on V100/T4."
        ),
    )
    parser.add_argument(
        "--eval-accumulation-steps",
        type=int,
        default=1,
        help=(
            "Flush eval logits to CPU every N batches to prevent OOM. "
            "Default 1 is safest. Increase to 2-4 if transfer overhead is visible."
        ),
    )
    parser.add_argument(
        "--prediction-loss-only",
        action="store_true",
        help=(
            "Skip logit accumulation and metric computation during eval. "
            "Use for quick loss-curve sanity checks only. Disables eval_f1."
        ),
    )
    parser.add_argument(
        "--pretokenize-only",
        action="store_true",
        help=(
            "Tokenize val_1p and test_1p subsets to Arrow format and exit. "
            "Run once after data prep, before training. Fast (~seconds)."
        ),
    )
    parser.add_argument(
        "--use-pretokenized",
        action="store_true",
        help="Load Arrow val_1p for intra-training eval. Requires --pretokenize-only first.",
    )
    parser.add_argument(
        "--pretokenize-num-proc",
        type=int,
        default=4,
        help="Parallel processes for pre-tokenization (default: 4).",
    )
    parser.add_argument(
        "--pretokenized-dir",
        type=str,
        default=str(PRETOKENIZED_DIR),
        help=f"Directory for Arrow datasets (default: {PRETOKENIZED_DIR}).",
    )
    parser.add_argument(
        "--train-sample-fraction",
        type=float,
        default=0.0,
        help=(
            "Fraction of train.jsonl to use, stratified by source. "
            "0.05 = 5%% (~44k records), 0.10 = 10%% (~88k records). "
            "Saved to data/train_Xp.jsonl and reused on subsequent runs. "
            "Default 0 uses the full training set."
        ),
    )
    parser.add_argument(
        "--skip-final-eval",
        action="store_true",
        help=(
            "Skip final evaluation on the full test set. "
            "Use when you only need the trained weights (models/best_model/) "
            "and plan to run inference locally."
        ),
    )

    args = parser.parse_args()

    run_pipeline(
        skip_download=args.skip_download,
        batch_size=args.batch_size,
        eval_batch_size=args.eval_batch_size,
        grad_accum=args.grad_accum,
        learning_rate=args.lr,
        num_epochs=args.epochs,
        max_steps=args.max_steps,
        max_length=args.max_length,
        warmup_ratio=args.warmup_ratio,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.early_stopping_patience,
        eval_steps=args.eval_steps,
        save_steps=args.save_steps,
        logging_steps=args.logging_steps,
        use_gradient_checkpointing=args.gradient_checkpointing,
        resume_from_checkpoint=args.resume_from_checkpoint,
        fp16_full_eval=args.fp16_full_eval,
        torch_compile=args.torch_compile,
        eval_accumulation_steps=args.eval_accumulation_steps,
        prediction_loss_only=args.prediction_loss_only,
        pretokenize_only=args.pretokenize_only,
        use_pretokenized=args.use_pretokenized,
        pretokenized_dir=Path(args.pretokenized_dir),
        pretokenize_num_proc=args.pretokenize_num_proc,
        train_sample_fraction=args.train_sample_fraction,
        skip_final_eval=args.skip_final_eval,
    )


if __name__ == "__main__":
    main()