# Corrected Held-Out Benchmark Protocol

## Purpose

The direct DeBERTa baseline and the source-conditioned hierarchical DeBERTa
model were trained on the corrected PIIBench preparation pipeline. The final
paper comparison must therefore use examples that remain held out under that
same corrected split.

This protocol compares all systems on one new, reproducible benchmark:

```text
data/test_5k.jsonl
```

## Why Not Reuse The Original 1% Table


The corrected preparation pipeline rebuilt the
train/validation/test membership and label space. Evaluating the newly trained
models against an old held-out subset risks testing on records that are now in
their training split. Results from the old table remain background context,
but are not a clean head-to-head comparison for the new trained models.

## Benchmark Dataset

`create_evaluation_subset.py` samples only from the corrected held-out
`data/test.jsonl`. It uses source-stratified proportional allocation with
largest remainders, followed by seeded within-source sampling.

| Item | Value |
|---|---:|
| Sampling seed | `42` |
| Corrected held-out source test records | `100,002` |
| Benchmark records | `5,000` |
| Benchmark entity mentions | `15,326` |
| Represented sources | `10` |
| Represented entity types | `82` |
| SHA-256 of `data/test_5k.jsonl` | `46238c6cc12526d4d29e9282e96d2936f77174f5fb0d1be903039b6ab77d2d16` |

Generate the artifact from the prepared split:

```bash
python create_evaluation_subset.py \
  --test-file ./data/test.jsonl \
  --output-file ./data/test_5k.jsonl \
  --summary-file ./data/test_5k_summary.json \
  --size 5000 \
  --seed 42
```

The generated summary records source counts, entity counts, and both the full
test and subset hashes for reporting and reproducibility.

## Systems Evaluated

The comparison contains the eight systems in the original PIIBench table plus
the two models trained in this experiment.

| System | Role |
|---|---|
| Microsoft Presidio | Original rule-based baseline |
| spaCy `en_core_web_lg` | Original general NER baseline |
| `dslim/bert-base-NER` | Original CoNLL-2003 baseline |
| `Davlan/xlm-roberta-base-wikiann-ner` | Original WikiANN baseline |
| `tomaarsen/span-marker-mbert-base-multinerd` | Original MultiNERD baseline |
| `tomaarsen/span-marker-bert-base-fewnerd-fine-super` | Original FewNERD baseline |
| `iiiorg/piiranha-v1-detect-personal-information` | Original PII-specific baseline |
| `nbroad/finer-139-xtremedistil-l12-h384` | Original financial NER baseline |
| Direct fine-tuned DeBERTa | Experiment baseline |
| Source-conditioned hierarchical DeBERTa | Proposed model - v1 (without Curriculum Learning) |

All results use seqeval exact span and entity-type precision, recall, and F1
after mapping external labels into the corrected 82-entity taxonomy. The
benchmark outputs record the dataset hash and the mapping revision
`corrected_canonical_v1`.

The original eight comparators are mandatory for both local and VM
benchmarking. `run_benchmarking.py` supplies Microsoft Presidio and spaCy,
while `run_existing_models_benchmark.py` supplies the remaining six original
models. Neither result file is a complete comparison table by itself.

## Execution

From the repository directory, extract the downloaded novelty best model if
it has not already been extracted:

```bash
tar -xzf ../cloud_runs/novelty/pii-novelty-run.tar.gz \
  -C ../cloud_runs \
  novelty/best_model \
  novelty/novelty_results_summary.json \
  novelty/trainer_state.json
```

Install the evaluation dependencies and the spaCy pipeline used in the
original paper:

```bash
pip install -r requirements.txt span-marker
python -m spacy download en_core_web_lg
```

Run the proposed model, spaCy, and Presidio:

```bash
mkdir -p benchmark_results/corrected_test_5k
python run_benchmarking.py \
  --test-file ./data/test_5k.jsonl \
  --model-path ../cloud_runs/novelty/best_model \
  --system-name "Source-conditioned Hierarchical DeBERTa" \
  --confidence-threshold 0.0 \
  --max-length 256 \
  --device cuda \
  --spacy-model en_core_web_lg \
  --batch-size 8 \
  --output-dir ./benchmark_results/corrected_test_5k/novelty_spacy_presidio
```

Run the direct DeBERTa model without repeating spaCy and Presidio:

```bash
python run_benchmarking.py \
  --test-file ./data/test_5k.jsonl \
  --model-path ../cloud_runs/baseline/models/best_model \
  --system-name "Direct Fine-tuned DeBERTa" \
  --confidence-threshold 0.0 \
  --max-length 256 \
  --device cuda \
  --batch-size 8 \
  --skip-spacy \
  --skip-presidio \
  --output-dir ./benchmark_results/corrected_test_5k/direct_deberta
```

Run the remaining six public neural baselines:

```bash
python run_existing_models_benchmark.py \
  --test-path ./data/test_5k.jsonl \
  --output-path ./benchmark_results/corrected_test_5k/public_models.json \
  --batch-size 8 \
  --device cuda
```

`run_existing_models_benchmark.py` saves after each model and resumes
successful entries only when the input hash and label-mapping revision match.
If a model fails because of memory pressure, reduce `--batch-size` and rerun
the same command; failed entries are retried.

Compile the complete validated table after the three runs finish:

```bash
python compile_comparative_results.py \
  --novelty-results ./benchmark_results/corrected_test_5k/novelty_spacy_presidio/benchmark_results.json \
  --direct-results ./benchmark_results/corrected_test_5k/direct_deberta/benchmark_results.json \
  --public-results ./benchmark_results/corrected_test_5k/public_models.json \
  --output-dir ./benchmark_results/corrected_test_5k
```

The compiler fails if any of the eight original systems or either trained
model is missing, has a failed public-model result, or was evaluated on a
different test-file hash or record count. It also rejects a comparison table
with fewer than `5,000` evaluated records.

## Reporting

Archive these files with the paper experiment record:

```text
data/test_5k_summary.json
benchmark_results/corrected_test_5k/novelty_spacy_presidio/benchmark_results.json
benchmark_results/corrected_test_5k/direct_deberta/benchmark_results.json
benchmark_results/corrected_test_5k/public_models.json
benchmark_results/corrected_test_5k/comparative_results_all_systems.json
benchmark_results/corrected_test_5k/comparative_results_all_systems.csv
benchmark_results/corrected_test_5k/comparative_results_all_systems.md
```

Report the old paper table as a prior benchmark result and the corrected
5,000-record table as the controlled comparison for the new models. Do not
combine numeric rows from the two test subsets into a single head-to-head
ranking.
