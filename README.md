# PII Detection System

Production-ready PII detection system using a fine-tuned DeBERTa-v3-base token classification model.

## Project Structure

```
pii-detector/
├── data/                          # Train/val/test splits + label mapping
│   ├── train.jsonl
│   ├── val.jsonl
│   ├── test.jsonl
│   └── label_mapping.json
├── models/
│   ├── deberta-v3-base/           # Downloaded base model weights
│   ├── checkpoints/               # Training checkpoints
│   └── best_model/                # Final fine-tuned model
├── pii_datasets/                  # Raw downloaded datasets
│   └── consolidated/
│       ├── consolidated.jsonl
│       └── entity_types.json
├── src/
│   ├── download_datasets.py       # Downloads 10 datasets from HuggingFace
│   ├── consolidate_pii_datasets.py # Normalises all sources into BIO format
│   ├── data_preparation.py        # Splits, rare-entity dropping, label mapping
│   ├── download_model.py          # Downloads microsoft/deberta-v3-base
│   ├── train.py                   # Fine-tuning with HuggingFace Trainer
│   ├── inference.py               # PIIDetector and FastPIIDetector classes
│   ├── api.py                     # Flask REST API
│   └── exceptions.py              # Custom exception hierarchy
├── run_data_pipeline.py           # Orchestrates steps 1-3 of data prep
├── run_training_pipeline.py       # Orchestrates model download + fine-tuning
├── run_benchmarking.py            # Benchmarks our model vs spaCy vs Presidio
├── test_detector.py               # Test suite
├── example_client.py              # API client demo
├── Makefile                       # Convenience targets
├── Dockerfile
└── requirements.txt
```

## Datasets

Ten datasets are downloaded from HuggingFace and consolidated into a unified BIO-tagged format:

| Dataset | Rows | Domain |
|---|---|---|
| ai4privacy/pii-masking-400k | ~400k | General, 63 PII classes |
| ai4privacy/pii-masking-300k | ~300k | General + Finance (FinPII-80k) |
| gretelai/synthetic_pii_finance_multilingual | ~56k | Finance (100 doc types) |
| nvidia/Nemotron-PII | ~100k | General (50+ industries) |
| wikiann (en) | ~20k | Wikipedia, PER/ORG/LOC only |
| Babelscape/multinerd (en) | varies | Wikipedia + news, 15 types |
| DFKI-SLT/few-nerd | ~188k | Wikipedia, 66 fine-grained types |
| conll2003 | ~14k | News (Reuters), 4 types |
| nlpaueb/finer-139 | ~1.1M | Finance (SEC filings), 139 XBRL tags |
| Isotonic/pii-masking-200k | ~200k | General, 54 PII classes |

finer-139 is capped at 150k records during data preparation. Entity types with fewer than 500 B- mentions globally are collapsed to O.

## Quick Start

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the data pipeline

```bash
python run_data_pipeline.py
```

This downloads all datasets, normalises them into BIO format, drops rare entity types, and writes stratified 80/10/10 train/val/test splits.

Skip steps you have already run:

```bash
python run_data_pipeline.py --skip-download
python run_data_pipeline.py --skip-download --skip-consolidate
```

### 3. Run the training pipeline

```bash
python run_training_pipeline.py
```

This downloads `microsoft/deberta-v3-base` and fine-tunes it. The best checkpoint is saved to `./models/best_model`.

Skip the model download if already done:

```bash
python run_training_pipeline.py --skip-download
```

### 4. Run tests

```bash
python test_detector.py
```

### 5. Start the API server

```bash
python src/api.py --model-path ./models/best_model
```

Or via Make:

```bash
make api
```

## Training Configuration

Default hyperparameters (`run_training_pipeline.py`):

| Parameter | Default |
|---|---|
| Base model | microsoft/deberta-v3-base |
| Batch size | 16 |
| Gradient accumulation | 4 (effective batch = 64) |
| Learning rate | 2e-5 |
| Epochs | 10 |
| Max sequence length | 512 |
| Warmup ratio | 0.06 |
| Weight decay | 0.01 |
| Early stopping patience | 3 |
| Eval/save every | 2000 steps |

Override any parameter:

```bash
python run_training_pipeline.py \
    --batch-size 8 \
    --grad-accum 8 \
    --epochs 5 \
    --max-length 256 \
    --gradient-checkpointing
```

Resume from a checkpoint:

```bash
python run_training_pipeline.py --skip-download --resume-from-checkpoint
```

## Inference

### PIIDetector (single-text / sequential batch)

```python
from src.inference import PIIDetector

detector = PIIDetector(
    model_path="./models/best_model",
    confidence_threshold=0.5,
)

result = detector.detect("My email is alice@example.com and SSN is 123-45-6789")
print(result.has_pii)        # True
print(result.redacted_text)  # "My email is [REDACTED] and SSN is [REDACTED]"
print(result.pii_types)      # ["EMAIL", "SSN"]
print(result.entities)
# [
#   {"text": "alice@example.com", "type": "EMAIL", "start": 12, "end": 29, "confidence": 0.97},
#   {"text": "123-45-6789", "type": "SSN", "start": 39, "end": 50, "confidence": 0.95},
# ]
```

`detect()` always returns a `PIIResult` dataclass. `has_pii` is False and `error` is set on per-item inference failures, so batch callers can continue processing.

### FastPIIDetector (padded-batch inference)

```python
from src.inference import FastPIIDetector

detector = FastPIIDetector(
    model_path="./models/best_model",
    confidence_threshold=0.5,
    batch_size=32,
)

texts = ["Contact Alice at alice@example.com", "The weather is nice"]
results = detector.batch_detect(texts)
stats = detector.get_pii_statistics(results)
```

`FastPIIDetector` pads inputs and runs a single forward pass per batch. Use this for high-throughput pipelines.

### PIIResult fields

| Field | Type | Description |
|---|---|---|
| `has_pii` | bool | True if any entity found above threshold |
| `redacted_text` | str | Input with PII spans replaced by `[REDACTED]` |
| `pii_types` | list[str] | Sorted unique entity types found |
| `entities` | list[dict] | Per-entity: text, type, start, end, confidence |
| `error` | str or None | Set on per-item failure; None on success |

## REST API

### Endpoints

#### GET /health

```bash
curl http://localhost:5000/health
```

```json
{"status": "healthy", "model_loaded": true}
```

#### GET /info

Returns model path, device, confidence threshold, supported PII types, num_labels, batch_size.

#### POST /detect

```bash
curl -X POST http://localhost:5000/detect \
  -H "Content-Type: application/json" \
  -d '{"text": "Email me at user@example.com"}'
```

```json
{
  "has_pii": true,
  "redacted_text": "Email me at [REDACTED]",
  "pii_types": ["EMAIL"],
  "entities": [{"text": "user@example.com", "type": "EMAIL", "start": 12, "end": 28, "confidence": 0.97}],
  "error": null
}
```

#### POST /detect/batch

```bash
curl -X POST http://localhost:5000/detect/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["My name is John", "SSN: 123-45-6789"], "return_stats": true}'
```

Returns `{"results": [...], "statistics": {...}}`. Every entry in `results` is a PIIResult dict — never null.

#### POST /detect/file

Accepts `.txt`, `.csv`, `.xlsx`, `.xls` uploads (max 50 MB).

```bash
curl -X POST http://localhost:5000/detect/file \
  -F "file=@data.csv" \
  -F "columns=name,email"
```

For CSV/Excel, `columns` is an optional comma-separated list of column names to process. Omit to process all columns. The response includes per-cell results and a fully redacted copy of the records.

For plain text, each non-blank line is processed separately.

All error responses use the shape: `{"error": "<ExceptionClassName>", "message": "...", "details": {...}}`.


### Deployment

```bash
# 4-worker gunicorn
make api-prod

# Docker
make docker-build
make docker-run
make docker-run-gpu
```

### API client demo

```bash
python example_client.py --host http://localhost:5000
```

Demonstrates all four endpoints including file upload for `.txt` and `.csv`.

## Benchmarking

Evaluates our model against spaCy `en_core_web_trf` and Microsoft Presidio on the test split using seqeval span-level F1.

```bash
# Download spaCy models first
make download-deps

# Run benchmark on full test set
make benchmark

# Limit to first 500 records
make benchmark N=500

# Skip individual systems
python run_benchmarking.py --skip-spacy
python run_benchmarking.py --skip-presidio
```

Entity type normalisation maps are defined in `run_benchmarking.py` (`SPACY_LABEL_MAP`, `PRESIDIO_LABEL_MAP`) to align external systems to our label set before evaluation.

Results are written to `./benchmark_results/benchmark_results.json` and per-system report JSONs.

## Makefile targets

```
make install            Install dependencies
make download-deps      Download spaCy models for benchmarking
make data               Full data pipeline (download + consolidate + prepare)
make data SKIP=1        Skip download, re-run consolidation + preparation
make train              Download base model + fine-tune
make train SKIP=1       Skip model download, fine-tune only
make pipeline           data + train
make test               Run test suite
make benchmark          Benchmark all three systems
make benchmark N=500    Benchmark on first 500 records
make api                Start API server (dev mode)
make api-prod           Start API via gunicorn (4 workers)
make docker-build       Build Docker image
make docker-run         Run Docker container
make clean              Remove __pycache__ and .pyc files
make clean-models       Remove trained model checkpoints
make clean-data         Remove data splits
make clean-all          Remove everything generated
```

## Custom exceptions

All exceptions inherit from `PIIDetectionError` and expose a `to_dict()` method used by the API error handler.

```
PIIDetectionError
├── ModelError
│   ├── ModelNotFoundError
│   ├── ModelLoadError
│   └── ModelInferenceError
├── InputError
│   ├── EmptyInputError
│   ├── InputTooLargeError
│   └── InvalidInputTypeError
├── FileParsingError
│   ├── UnsupportedFileTypeError
│   ├── FileReadError
│   ├── TextFileParsingError
│   ├── CSVParsingError
│   └── ExcelParsingError
└── APIError
    ├── MissingFieldError
    └── InvalidFieldError
```

## Author

Pritesh Jha

## License

Apache License version 2.0
