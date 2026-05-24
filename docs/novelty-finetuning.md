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

Curriculum learning was **not enabled** in the primary completed run reported
below. A separate curriculum-enabled training run has since completed and is
documented as an additional evaluation candidate later in this document.

For the subsequent full novelty experiment, curriculum mode is executed as
three ordered, checkpointed phases:

```text
Phase 1: general NER sources
Phase 2: synthetic PII sources
Phase 3: financial PII sources
```

Each phase runs one epoch over its assigned source family, loads that phase's
best validation checkpoint, and passes the resulting model into the next
phase. Curriculum runs can be restarted with `--resume-from-checkpoint`; each
phase resumes its most recent checkpoint when one exists.

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

### Additional Variant: Curriculum Learning

The novelty implementation supports source-aware curriculum learning:

```text
general NER -> synthetic PII -> financial PII
```

The first completed novelty experiment was deliberately run as a single
training phase:

```text
Source conditioning: True
Curriculum learning: False
Hierarchical head: True
```

Therefore, its reported results isolate source conditioning and hierarchical
classification without attributing gains to curriculum scheduling. The later
curriculum-enabled variant is retained as a separately evaluated experiment.

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

## Controlled Held-Out Comparison

The exported source-conditioned hierarchical models, with and without
curriculum, were evaluated alongside the direct baseline and the eight
original PIIBench comparison systems on the corrected held-out
`data/test_5k.jsonl` subset.

| Approach | Evaluation Split | F1 | Precision | Recall |
|---|---|---:|---:|---:|
| Direct DeBERTa fine-tuning baseline | `test_5k` | **0.6476** | **0.6300** | **0.6662** |
| Source-conditioned hierarchical DeBERTa, no curriculum | `test_5k` | 0.5899 | 0.5565 | 0.6274 |
| Source-conditioned hierarchical DeBERTa, with curriculum | `test_5k` | 0.2772 | 0.3491 | 0.2299 |
| Best original comparator: SpanMarker BERT | `test_5k` | 0.1723 | 0.4266 | 0.1080 |

The source-conditioned hierarchical model exceeds the strongest original
comparator by `0.4176` absolute F1. However, it trails direct fine-tuning by
`0.0577` F1 on held-out comparison data, despite outperforming direct
fine-tuning on `val_1p`. The appropriate conclusion is that source
conditioning plus hierarchy alone did not improve controlled held-out
performance in this run. The curriculum-enabled run should be treated as a
separate experiment, not as confirmation of the no-curriculum result.

## Curriculum-Enabled Variant Results

A separate run completed with source conditioning, hierarchical classification,
and three-phase curriculum learning enabled:

```text
Phase 1: general NER sources
Phase 2: synthetic PII sources
Phase 3: gretel_finance and finer_139
```

The run completed and its exported model was saved to:

```text
models/full_novel_curriculum/final_model
```

The training command used `--skip-final-eval`. After export, the final
curriculum model was evaluated locally on the identical corrected
`data/test_5k.jsonl` subset used for all comparison systems.

Validation trajectory across the retained phase-best checkpoints:

| Phase | Best Step | Eval F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| Phase 1: general NER | 2,000 | 0.13074 | 0.10973 | 0.16169 |
| Phase 2: synthetic PII | 6,000 | **0.42983** | 0.37389 | 0.50545 |
| Phase 3: financial PII | 1,000 | 0.30474 | 0.32372 | 0.28787 |

Final curriculum model held-out result:

| Evaluation Split | Records | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| `test_5k` | 5,000 | 0.2772 | 0.3491 | 0.2299 |

The curriculum final model remains above the strongest original external
comparator by `0.1049` absolute F1, but it trails the no-curriculum novelty
model by `0.3127` F1 and direct fine-tuning by `0.3704` F1. This is consistent
with catastrophic forgetting during the source-restricted curriculum:
performance rose after the synthetic PII phase, then deteriorated after the
financial-only phase. Phase 2 may be evaluated as a diagnostic early-stopping
ablation, but it is not the result of the completed three-phase curriculum
model.

## Full Held-Out Test Comparison

After the `test_5k` ranking reversed the validation ordering, the direct
fine-tuned model and the source-conditioned hierarchical model without
curriculum were evaluated on the complete corrected held-out split. The
curriculum model was not included in this final competition because it had
already substantially underperformed both models on `test_5k`.

| Model | Test Records | F1 | Precision | Recall |
|---|---:|---:|---:|---:|
| **Direct Fine-tuned DeBERTa** | 100,002 | **0.6455** | **0.6277** | **0.6645** |
| Source-conditioned Hierarchical DeBERTa | 100,002 | 0.5894 | 0.5560 | 0.6270 |

Both runs used the same test artifact:

```text
SHA-256: 65f8edc86399ba3f9e4ba44591d4583f9271f5d1df20e30a913305049559df77
Metric: seqeval exact span and entity type match
```

Evaluation streamed records in 5,000-record chunks on a local NVIDIA GeForce
RTX 4070 with 8 GB VRAM:

| Model | CUDA Minibatch | Runtime (sec) | Peak Allocated VRAM | Peak Reserved VRAM |
|---|---:|---:|---:|---:|
| Direct Fine-tuned DeBERTa | 8 | 1123.0 | 0.955 GiB | 1.057 GiB |
| Source-conditioned Hierarchical DeBERTa | 6 | 995.5 | 0.891 GiB | 0.977 GiB |

The full test split confirms the `test_5k` ranking: the direct model improves
on SC+H by `0.0561` absolute F1, `0.0717` precision, and `0.0375` recall.

### Entity-Level Analysis

Across the `82` fine-grained entity types, direct DeBERTa has higher F1 on
`54` types and SC+H has higher F1 on `28` types. When fine-entity F1 is
summarized within the ten hierarchical groups using support-weighted means,
direct DeBERTa is better in every group.

| Group | Support | Direct F1 | SC+H F1 | Direct Delta |
|---|---:|---:|---:|---:|
| FINANCIAL_NER | 58,821 | 0.3229 | 0.2412 | +0.0817 |
| LOCATION | 53,111 | 0.7151 | 0.6729 | +0.0422 |
| PERSON_GROUP | 46,789 | 0.8004 | 0.7515 | +0.0489 |
| ORG_ROLE | 30,723 | 0.7422 | 0.7252 | +0.0170 |
| TEMPORAL | 30,683 | 0.5923 | 0.5548 | +0.0376 |
| NETWORK | 24,406 | 0.6611 | 0.5920 | +0.0691 |
| MISC | 23,574 | 0.7318 | 0.6321 | +0.0997 |
| CONTACT | 18,437 | 0.7087 | 0.6593 | +0.0494 |
| CREDENTIAL | 12,882 | 0.8902 | 0.8611 | +0.0290 |
| FINANCIAL_ID | 8,995 | 0.8763 | 0.7305 | +0.1458 |

The most consequential direct-model gains include `CRYPTO_ADDRESS`
(`+0.8629` F1, support `1,569`), `IBAN` (`+0.3743`, support `769`),
`ACCOUNT_NUMBER` (`+0.2590`, support `2,686`), `IP_ADDRESS` (`+0.1330`,
support `6,178`), `USERNAME` (`+0.1099`, support `8,287`), and
`FINANCIAL_ENTITY` (`+0.0817`, support `58,821`).

SC+H remains stronger for selected entities, most notably `HTTP_COOKIE`
(`+0.3940` F1, support `595`), `DATE_TIME` (`+0.0422`, support `1,463`),
`PHONE_NUMBER` (`+0.0132`, support `3,079`), `COORDINATE` (`+0.0195`,
support `857`), and `COMPANY_NAME` (`+0.0073`, support `8,674`). These
localized benefits do not outweigh direct DeBERTa's broader advantages.

The complete entity and group outputs are generated by:

```bash
python analyze_full_test_comparison.py
```

and written under:

```text
benchmark_results/full_test/direct_vs_source_conditioned_hierarchical_analysis.md
benchmark_results/full_test/direct_vs_source_conditioned_hierarchical_analysis.json
benchmark_results/full_test/direct_vs_source_conditioned_hierarchical_entities.csv
benchmark_results/full_test/direct_vs_source_conditioned_hierarchical_groups.csv
```

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

The curriculum archive downloaded for local analysis contains:

```text
cloud_runs/curriculum/models/full_novel_curriculum/final_model/
cloud_runs/curriculum/models/full_novel_curriculum/phase_1/
cloud_runs/curriculum/models/full_novel_curriculum/phase_2/
cloud_runs/curriculum/models/full_novel_curriculum/phase_3/
cloud_runs/curriculum/models/full_novel_curriculum/curriculum_phase_summary.json
```

Its controlled held-out evaluation is stored at:

```text
benchmark_results/corrected_test_5k/curriculum/benchmark_results.json
```

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

- The completed run does not measure the incremental contribution of source
  conditioning versus the hierarchical head separately.
- The three-phase curriculum schedule changes both source composition and
  optimization trajectory; its poor final score does not establish that all
  curriculum-learning designs will fail.
- Phase 2 has better validation F1 than phase 3 but has not been reported as a
  held-out model because it is not the final three-phase curriculum output.

## Reporting Status

The controlled `test_5k` comparison now includes the eight original
comparators and three trained models: direct fine-tuning, source-conditioned
hierarchical fine-tuning, and its curriculum-enabled variant. Report these
held-out scores as the primary experimental comparison and retain the
`val_1p` scores only as model-selection evidence. The benchmark includes
`222` BIO continuation spans that seqeval treats as new gold entities; the
generated summary records this caveat in `data/test_5k_summary.json`.

Remaining optional analyses are a BIO-normalized sensitivity analysis and
evaluation of the retained phase-2 curriculum checkpoint as an early-stopping
ablation.

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
`1.67%` absolute. On the completed held-out `test_5k` comparison, however,
the same model scores `0.5899` F1 versus direct fine-tuning at `0.6476` F1.
The curriculum-enabled variant performs more poorly still, scoring `0.2772`
F1 on `test_5k`, with its degradation appearing after the final
financial-specialization phase. On the final full-test competition, direct
fine-tuning scores `0.6455` F1 while SC+H scores `0.5894`, establishing the
simpler direct model as the strongest completed solution.
