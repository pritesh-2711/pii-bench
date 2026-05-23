# DeBERTa Fine-Tuning for PII Detection on PIIBench dataset

## Summary

This experiment fine-tunes `microsoft/deberta-v3-base` directly as a token
classification model for BIO-tagged PII detection. It is the main baseline
experiment before running the novelty stack with source conditioning,
curriculum learning, and hierarchical classification.

The goal is to establish a strong, reproducible DeBERTa baseline on the
corrected prepared PIIBench splits:

- Nemotron spans are parsed correctly.
- Nemotron contributes about 10% of each train/validation/test split.
- Training uses prepared JSONL splits only.
- Rare labels are dropped during data preparation when below threshold.
- The token-classification loss downweights `O` to avoid all-`O` collapse.

## Method

The model is trained as a standard Hugging Face token-classification model:

```text
Base encoder: microsoft/deberta-v3-base
Task head: token classification
Labels: BIO PII labels
Objective: weighted cross-entropy over supervised tokens
Metric: seqeval span-level precision, recall, F1
```

The implementation path is:

```text
run_training_pipeline.py
  -> src/train.py
  -> PIITrainer
  -> AutoModelForTokenClassification
  -> WeightedTokenClassificationTrainer
```

This is intentionally a direct fine-tuning baseline. It does not use:

- Source-conditioning tokens
- Curriculum phases
- Hierarchical coarse-to-fine head

Those are reserved for the separate novelty experiment.

## Data Processing

The experiment uses the prepared splits produced by:

```bash
python run_data_pipeline.py
```

The data pipeline:

1. Downloads and consolidates multiple public PII/NER datasets.
2. Normalizes labels into a unified BIO format.
3. Fixes Nemotron span parsing for Python-style span strings.
4. Caps `finer_139` to reduce source domination.
5. Drops rare entity types below the configured threshold.
6. Keeps PII-bearing Nemotron rows and balances Nemotron to roughly 10% of each split.
7. Writes source-stratified train/validation/test splits.

Prepared split sizes used in this experiment:

| Split | Records |
|---|---:|
| Train | 799,948 |
| Validation | 99,990 |
| Test | 100,002 |
| `val_1p` | 994 |
| `test_1p` | 995 |

Label space:

| Item | Count |
|---|---:|
| Entity types | 82 |
| BIO labels including `O` | 165 |

Training distribution observed in logs:

| Statistic | Value |
|---|---:|
| Training records | 799,948 |
| All-`O` records | 156,800 / 19.60% |
| Supervised tokens | 52,846,281 |
| `O` tokens | 44,016,269 / 83.29% |
| Non-`O` tokens | 8,830,012 / 16.71% |

Validation subset distribution observed in logs:

| Statistic | Value |
|---|---:|
| `val_1p` records | 994 |
| All-`O` records | 193 / 19.42% |
| Supervised tokens | 66,155 |
| `O` tokens | 55,295 / 83.58% |
| Non-`O` tokens | 10,860 / 16.42% |

## Infrastructure

The completed run was executed on a Google Cloud GPU VM.

Known infrastructure:

| Field | Value |
|---|---|
| Cloud provider | Google Cloud Platform |
| Instance name | `instance-20260521-183630` |
| Region | asia-southeast1-c |
| Machine type  | g2-standard-12 |
| GPU | 1 x NVIDIA L4 |
| CPU count | 12 vCPUs |
| Memory | 48 GB |
| Boot Disk Size | 250 GB |
| Device used by training | CUDA |
| Mixed precision | BF16 |
| NVIDIA Driver Version | 580.126.20 |
| CUDA Driver | 13.0 |
| Torch Driver | 2.12.0 |
| Dataset | [`pritesh-2711/pii-bench` from HuggingFace](https://huggingface.co/datasets/Pritesh-2711/pii-bench) |


## Training Configuration

The stable baseline run used direct DeBERTa fine-tuning with these effective
settings:

| Parameter | Value |
|---|---:|
| Base model | `microsoft/deberta-v3-base` |
| Local model path | `models/deberta-v3-base` |
| Max sequence length | 256 |
| Per-device train batch size | 6 |
| Per-device eval batch size | 8 |
| Gradient accumulation | 10 |
| Effective batch size | 60 |
| Learning rate | 2e-5 |
| Warmup ratio | 0.06 |
| Weight decay | 0.01 |
| `O` label loss weight | 0.1 |
| Entity label loss weight | 1.0 |
| Eval accumulation steps | 1 |
| Checkpoint interval | 1000 steps |
| Eval interval | 1000 steps |
| Logging interval | 50 steps |

The long baseline command was:

```bash
python run_training_pipeline.py \
  --skip-download \
  --max-length 256 \
  --batch-size 6 \
  --eval-batch-size 8 \
  --grad-accum 10 \
  --epochs 6 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --logging-steps 50 \
  --eval-accumulation-steps 1 \
  --gradient-checkpointing
```


## Runtime

Observed training runtime:

| Item | Value |
|---|---:|
| Completed training steps before best-model save | 26,000 |
| Reported train runtime | 42,496 sec |
| Approx runtime | 11.8 hours |
| Train samples/sec | 112.94 |
| Train steps/sec | 1.882 |

The run saved the best checkpoint to:

```text
models/best_model
```

The exported model archive was:

```text
pii-best-model.tar.gz
```

Archive size observed:

```text
639 MB
```

Model directory size observed:

```text
710 MB
```

## Performance

A 500-step sanity run confirmed that the weighted loss avoids all-`O` collapse:

| Step | Eval F1 | Precision | Recall | Predicted `O` Ratio |
|---:|---:|---:|---:|---:|
| 100 | 0.0969 | 0.0826 | 0.1174 | 73.40% |
| 200 | 0.2106 | 0.1811 | 0.2741 | 67.21% |
| 300 | 0.2788 | 0.2362 | 0.3401 | 70.61% |
| 400 | 0.2930 | 0.2449 | 0.3644 | 70.00% |
| 500 | 0.3024 | 0.2553 | 0.3710 | 70.17% |

During the longer baseline run, validation on `val_1p` reached:

| Step | Eval Loss | Eval F1 | Precision | Recall | Predicted `O` Ratio |
|---:|---:|---:|---:|---:|---:|
| 24,000 | 0.2285 | 0.6445 | 0.5771 | 0.7298 | 76.83% |
| 25,000 | 0.2355 | 0.6337 | 0.5638 | 0.7235 | 76.82% |
| 26,000 | 0.2408 | 0.6315 | 0.5557 | 0.7312 | 76.94% |

Best observed validation F1 from the shared logs:

```text
eval_f1 ~= 0.6445 at step 24,000
```

The saved model was written after training:

```text
Saving best model to models/best_model ...
```

Full-test metrics are not available from this run because final full-test
evaluation was interrupted after training completed.

## Results

The direct DeBERTa fine-tuning baseline successfully trained and produced a
usable exported model:

```text
models/best_model/model.safetensors
models/best_model/config.json
models/best_model/tokenizer.json
models/best_model/label_mapping.json
```

The best validation signal observed in logs was:

```text
F1:        ~0.6445
Precision: ~0.5771
Recall:    ~0.7298
```

The completed controlled comparison on the corrected held-out
`data/test_5k.jsonl` benchmark produced:

| Evaluation Split | Records | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| `val_1p` model-selection subset | 994 | 0.6445 | 0.5771 | 0.7298 |
| `test_5k` controlled comparison subset | 5,000 | **0.6476** | **0.6300** | **0.6662** |

On `test_5k`, direct fine-tuned DeBERTa is the strongest of all ten evaluated
systems. It exceeds the strongest original PIIBench comparator in this new
benchmark, SpanMarker BERT (`F1 0.1723`), by `0.4753` absolute F1.

The model was exported as:

```bash
tar -czf pii-best-model.tar.gz \
  models/best_model \
  data/label_mapping.json
```

## Inference Usage

After extracting the archive:

```bash
tar -xzf pii-best-model.tar.gz
```

Minimal inference:

```python
import sys
sys.path.append("src")

from inference import PIIDetector

detector = PIIDetector("models/best_model", confidence_threshold=0.5)
result = detector.detect(
    "My name is Pritesh Jha and my email is pritesh@example.com"
)
print(result.to_dict())
```

## Known Limitations

- Final full-test evaluation did not complete due to memory-heavy prediction
  accumulation in the default Trainer evaluation loop.
- A final controlled test result is available on `test_5k`; a metric on the
  complete 100,002-record `test.jsonl` split is not available.
- The best checkpoint selection is based on the fast validation subset.
- The controlled benchmark contains BIO continuation irregularities documented
  in `docs/corrected-test-5k-benchmark.md`.
