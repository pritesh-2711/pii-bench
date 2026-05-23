# Source-Conditioned Hierarchical DeBERTa Fine-Tuning for PII Detection on PIIBench Dataset

## Summary

This experiment evaluates the proposed novelty approach for PII detection:
a source-conditioned, hierarchical token-classification model built on
`microsoft/deberta-v3-base`.

The experiment was run after fixing data preparation issues observed in the
initial exploratory analysis, including Nemotron span parsing and source
balancing. It uses the same prepared PIIBench splits as the direct DeBERTa
baseline so that the model changes can be compared on a common validation
subset.

The completed novelty run uses:

- Source-conditioning tokens to indicate the originating dataset.
- A hierarchical coarse-to-fine classification head.
- Weighted loss to reduce collapse toward the dominant `O` label.
- The complete prepared training split.
- Fast repeated model selection on `val_1p`.

Curriculum learning is implemented in the codebase but was **not enabled** in
the completed run reported here.

## Proposed Approach

### Source Conditioning

Each record is prefixed with a learned source token before tokenization:

```text
[SRC=<dataset>] original text
```

Examples include:

```text
[SRC=nvidia_nemotron]
[SRC=ai4privacy_400k]
[SRC=gretel_finance]
[SRC=general]
```

This allows the model to learn source-specific annotation and language
characteristics while sharing the DeBERTa encoder across all data sources.
At inference time, unknown or general inputs use:

```text
[SRC=general]
```

### Hierarchical Coarse-to-Fine Classification

The model predicts each PII token at two related levels:

1. A coarse BIO category, such as `PERSON_GROUP`, `CONTACT`, `FINANCIAL_ID`,
   `CREDENTIAL`, `NETWORK`, `LOCATION`, or `FINANCIAL_NER`.
2. The fine-grained BIO entity type used for final evaluation, such as
   `B-EMAIL`, `I-ACCOUNT_NUMBER`, or `B-IP_ADDRESS`.

The hierarchical head is intended to help the model first recognize broad PII
semantics and then resolve fine entity distinctions. The corrected coarse
taxonomy covers all 82 retained entity types in the prepared label mapping.

### Training Objective

The loss combines fine-grained and coarse predictions:

```text
total loss = fine BIO loss + 0.3 * coarse BIO loss
```

Both loss terms downweight the non-entity label:

| Label Group | Weight |
|---|---:|
| `O` | 0.1 |
| Entity BIO labels | 1.0 |

This weighting is used because `O` tokens account for more than 83% of
supervised training tokens. It addresses the early failure mode where
precision, recall, and F1 remained at zero because the model favored
predicting only non-entity tokens.

### Implemented But Not Used: Curriculum Learning

The novelty implementation also supports source-aware curriculum learning:

```text
general NER -> synthetic PII -> financial PII
```

The completed experiment was deliberately run as a single training phase:

```text
Source conditioning: True
Curriculum learning: False
Hierarchical head: True
```

Therefore, the reported results isolate source conditioning and hierarchical
classification without attributing gains to curriculum scheduling.

## Implementation

The implementation path for the completed experiment is:

```text
run_training_pipeline.py --novel --source-cond --hierarchical
  -> src/train_novel.py
  -> HierarchicalPIIModel
  -> DeBERTa encoder + coarse BIO head + fine BIO head
```

The exported model configuration preserves the approach metadata:

| Configuration Field | Value |
|---|---|
| `architectures` | `["HierarchicalPIIModel"]` |
| `pii_model_architecture` | `hierarchical` |
| `pii_source_conditioned` | `true` |
| `pii_default_source_token` | `[SRC=general]` |
| `pii_coarse_loss_weight` | `0.3` |
| Vocabulary size after source tokens | `128012` |

Two implementation adjustments were required before the run succeeded:

1. The hierarchical model was extended to expose and resize DeBERTa input
   embeddings after source tokens were added to the tokenizer.
2. Gradient checkpointing support was enabled for the hierarchical wrapper so
   the model could run within L4 GPU memory limits.

## Data Processing

The novelty experiment uses the same prepared splits as the direct DeBERTa
baseline. The preparation pipeline:

1. Consolidates the source datasets into canonical BIO annotations.
2. Correctly parses Nemotron PII spans.
3. Caps the dominating `finer_139` source.
4. Drops entity types below the rare-label threshold.
5. Keeps PII-bearing Nemotron records and balances Nemotron to approximately
   10% of the prepared corpus.
6. Writes source-stratified train/validation/test JSONL splits.

Prepared split sizes:

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
| Retained entity types | 82 |
| Fine BIO labels including `O` | 165 |
| Coarse BIO labels including `O` | 21 |

Observed training-label distribution:

| Statistic | Value |
|---|---:|
| Training records | 799,948 |
| All-`O` records | 156,800 / 19.60% |
| Supervised tokens | 52,846,281 |
| `O` tokens | 44,016,269 / 83.29% |
| Non-`O` tokens | 8,830,012 / 16.71% |

## Infrastructure

The completed run was executed on the same Google Cloud GPU VM used for the
direct DeBERTa baseline.

| Field | Value |
|---|---|
| Cloud provider | Google Cloud Platform |
| Instance name | `instance-20260521-183630` |
| Zone | `asia-southeast1-c` |
| Machine type | `g2-standard-12` |
| GPU | 1 x NVIDIA L4 |
| CPU count | 12 vCPUs |
| Memory | 48 GB |
| Boot disk size | 250 GB |
| Device used by training | CUDA |
| Mixed precision | BF16 |
| Gradient checkpointing | Enabled |
| NVIDIA driver version | 580.126.20 |
| CUDA driver | 13.0 |
| PyTorch / CUDA build | Not preserved in the downloaded model artifact |
| Transformers version saved in model config | 4.46.3 |

## Training Configuration

Effective training arguments recovered from the saved checkpoint:

| Parameter | Value |
|---|---:|
| Base encoder | `microsoft/deberta-v3-base` |
| Local model path | `models/deberta-v3-base` |
| Architecture | Hierarchical PII model |
| Source conditioning | Enabled |
| Curriculum | Disabled |
| Coarse loss weight | 0.3 |
| Max sequence length | 256 |
| Per-device train batch size | 6 |
| Per-device eval batch size | 8 |
| Gradient accumulation | 10 |
| Effective batch size | 60 |
| Learning rate | 2e-5 |
| Warmup steps | 1,999 |
| LR scheduler | Linear |
| Maximum optimizer steps | 39,999 |
| Evaluation interval | 1,000 steps |
| Checkpoint interval | 1,000 steps |
| Logging interval | 50 steps |
| Eval accumulation steps | 1 |
| Saved checkpoint limit | 3 |
| Best-model criterion | `eval_f1` |
| Load best model at end | Enabled |
| Final full-test evaluation during training | Skipped |

The completed run can be represented by the following command:

```bash
export TOKENIZERS_PARALLELISM=false

python run_training_pipeline.py \
  --skip-download \
  --novel \
  --source-cond \
  --hierarchical \
  --novel-output-dir ./models/full_novel \
  --max-length 256 \
  --batch-size 6 \
  --eval-batch-size 8 \
  --grad-accum 10 \
  --epochs 3 \
  --eval-steps 1000 \
  --save-steps 1000 \
  --logging-steps 50 \
  --eval-accumulation-steps 1 \
  --gradient-checkpointing \
  --skip-final-eval
```

## Runtime

| Item | Value |
|---|---:|
| Total optimizer steps | 39,999 |
| Reported train runtime | 64,432.95 sec |
| Approximate runtime | 17 h 54 min |
| Train samples/sec | 37.247 |
| Train steps/sec | 0.621 |
| Final reported training loss | 0.39245 |
| Best selected checkpoint | Step 37,000 |

The Hugging Face trainer reports an epoch value of approximately `1.33` at the
end of this streaming-dataset run. The planned duration is best documented by
the configured and completed optimizer-step count of `39,999`.

## Validation Performance

The trainer evaluates on the fixed `val_1p` subset during training and selects
the checkpoint with the highest span-level F1.

Best selected novelty checkpoint:

| Step | Eval Loss | Eval F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| **37,000** | **0.26770** | **0.66123** | **0.59655** | **0.74165** |

Nearby late-training validation results:

| Step | Eval Loss | Eval F1 | Precision | Recall |
|---:|---:|---:|---:|---:|
| 35,000 | 0.27184 | 0.64839 | 0.57703 | 0.73989 |
| 36,000 | 0.27451 | 0.65199 | 0.58319 | 0.73919 |
| **37,000** | **0.26770** | **0.66123** | **0.59655** | **0.74165** |
| 38,000 | 0.27244 | 0.64715 | 0.57613 | 0.73814 |
| 39,000 | 0.27078 | 0.64902 | 0.57997 | 0.73673 |

The model-selection state confirms:

```text
best_model_checkpoint: models/full_novel/checkpoints/checkpoint-37000
best_metric: 0.6612347226574742
```

## Comparison With Direct Fine-Tuning

Both results below are validation scores on `val_1p`, using the same prepared
data pipeline and label space.

| Approach | Best Step | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Direct DeBERTa fine-tuning baseline | 24,000 | 0.64453 | 0.57710 | 0.72980 |
| Source-conditioned hierarchical DeBERTa | 37,000 | **0.66123** | **0.59655** | **0.74165** |

Observed improvement of the novelty approach over direct fine-tuning:

| Metric | Absolute Gain |
|---|---:|
| F1 | **+0.01670** |
| Precision | **+0.01945** |
| Recall | **+0.01186** |

These results provide positive validation evidence for the source-conditioned
hierarchical approach. 

## Saved Results And Verification

The completed run produced:

```text
models/full_novel/final_model
models/full_novel/test_results.json
models/full_novel/checkpoints/checkpoint-37000
```

The exported local archive is:

```text
cloud_runs/novelty/pii-novelty-run.tar.gz
```

Archive contents verified locally include:

```text
novelty/best_model/model.safetensors
novelty/best_model/config.json
novelty/best_model/tokenizer.json
novelty/best_model/label_mapping.json
novelty/novelty_results_summary.json
novelty/trainer_state.json
novelty/checkpoints/checkpoint-37000/
novelty/checkpoints/checkpoint-39000/
novelty/checkpoints/checkpoint-39999/
```

The exported `best_model` weights were checked against the selected checkpoint.
Both `model.safetensors` files have the same SHA-256 hash:

```text
c6833f5b4d3ce5285061c84e0acef6b12530db58a6910661e80763e1dd484c44
```

Therefore, `novelty/best_model` is verified to contain the weights from
checkpoint `37000`.

## Inference Usage

After extracting the archive:

```bash
tar -xzf cloud_runs/novelty/pii-novelty-run.tar.gz -C cloud_runs
```

Minimal inference:

```python
import sys

sys.path.append("src")

from inference import PIIDetector

detector = PIIDetector(
    "../cloud_runs/novelty/best_model",
    confidence_threshold=0.5,
)
result = detector.detect(
    "My name is Pritesh Jha and my email is pritesh@example.com"
)
print(result.to_dict())
```

The inference loader identifies the hierarchical model metadata from its
configuration and applies the default source token for general inputs.

## Known Limitations

- The reported scores are model-selection scores on `val_1p`, not final test
  performance.
- The completed run does not measure the incremental contribution of source
  conditioning versus the hierarchical head separately.
- Curriculum learning is implemented in the codebase but was not evaluated in
  this completed run.

## Required Follow-Up For Final Reporting

Before using final benchmark numbers in a paper:

1. Generate the fixed corrected held-out `data/test_5k.jsonl` benchmark with
   the current preparation method and seed `42`.
2. Run the eight systems from the PIIBench comparison and both trained
   DeBERTa variants on that identical subset, following
   `docs/corrected-test-5k-benchmark.md`.
3. Report its precision, recall, and F1 as the controlled comparison; retain
   the `val_1p` scores only as model-selection evidence.
4. Optionally run the two trained models on the complete held-out
   `test.jsonl` as an additional internal comparison.

## Result Summary

The completed novelty run successfully trained and exported a verified
source-conditioned hierarchical DeBERTa model. On the fixed `val_1p`
validation subset, its best checkpoint at step `37,000` achieved:

```text
F1:        0.66123
Precision: 0.59655
Recall:    0.74165
```

Compared with direct DeBERTa fine-tuning, this improves validation F1 by
`1.67%` absolute while also improving precision and recall.
