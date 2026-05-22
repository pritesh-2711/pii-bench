"""
PII detection and redaction inference.

Two classes:
    PIIDetector      — single-text and sequential batch inference
    FastPIIDetector  — padded-batch inference for high-throughput pipelines

Both classes return a PIIResult dataclass for every input, which carries:
    has_pii         bool
    redacted_text   str  (same type/format as input; original text if no PII)
    pii_types       list[str]
    entities        list[dict]
    error           str | None  (set on per-item inference failures)
"""

import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from transformers import AutoConfig, AutoTokenizer, AutoModelForTokenClassification

try:
    from train_novel import COARSE_NAMES, DEFAULT_SOURCE_TOKEN, HierarchicalPIIModel
except Exception:  # pragma: no cover - only used when loading novelty models
    COARSE_NAMES = []
    DEFAULT_SOURCE_TOKEN = "[SRC=general]"
    HierarchicalPIIModel = None

from exceptions import (
    EmptyInputError,
    InputTooLargeError,
    InvalidInputTypeError,
    ModelNotFoundError,
    ModelLoadError,
    ModelInferenceError,
)

# Maximum character length accepted per text. Texts longer than this are
# rejected before tokenisation to avoid silent truncation.
MAX_CHARS = 50_000

# Required files that must be present inside the model directory.
_REQUIRED_MODEL_FILES = {"label_mapping.json", "config.json"}


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

@dataclass
class PIIResult:
    """
    Returned by detect() and batch_detect() for every input text.

    Fields:
        has_pii         True if at least one entity was found above threshold.
        redacted_text   Input text with all PII spans replaced by [REDACTED].
                        Identical to the original text when has_pii is False.
        pii_types       Sorted list of unique entity type strings found.
        entities        List of dicts, each with keys:
                            text, type, start, end, confidence
        error           None on success; error message string on per-item failure.
    """
    has_pii: bool = False
    redacted_text: str = ""
    pii_types: List[str] = field(default_factory=list)
    entities: List[Dict] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> Dict:
        return asdict(self)


# ---------------------------------------------------------------------------
# PIIDetector
# ---------------------------------------------------------------------------

class PIIDetector:
    """
    Loads a fine-tuned DeBERTa-v3-base token classification model and runs
    PII detection and redaction on individual texts.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        device: Optional[str] = None,
    ):
        self.model_path           = Path(model_path)
        self.confidence_threshold = confidence_threshold

        self._validate_model_path()

        try:
            with open(self.model_path / "label_mapping.json") as f:
                mapping = json.load(f)
            self.label2id = mapping["label2id"]
            self.id2label = {int(k): v for k, v in mapping["id2label"].items()}
        except (KeyError, json.JSONDecodeError, OSError) as exc:
            raise ModelLoadError(str(self.model_path), str(exc)) from exc

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(str(self.model_path))
            self.config = AutoConfig.from_pretrained(str(self.model_path))
            self.source_conditioned = bool(
                getattr(self.config, "pii_source_conditioned", False)
            )
            self.default_source_token = getattr(
                self.config,
                "pii_default_source_token",
                DEFAULT_SOURCE_TOKEN,
            )
            self.model = self._load_model()
        except Exception as exc:
            raise ModelLoadError(str(self.model_path), str(exc)) from exc

        if device is not None:
            self.device = torch.device(device)
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model.to(self.device)
        self.model.eval()

        self.pii_types = sorted({
            lbl[2:] for lbl in self.id2label.values()
            if lbl.startswith("B-")
        })

        print(
            f"PIIDetector loaded: {self.model_path} | device={self.device} | "
            f"types={len(self.pii_types)} | threshold={self.confidence_threshold}"
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def detect(self, text: str) -> PIIResult:
        """
        Run detection and redaction on a single text string.

        Returns a PIIResult. Sets PIIResult.error (and has_pii=False) instead
        of raising on per-item inference failures so that batch callers can
        continue processing remaining items.

        Raises:
            EmptyInputError         if text is blank.
            InputTooLargeError      if text exceeds MAX_CHARS.
            InvalidInputTypeError   if text is not a str.
        """
        self._validate_text(text)

        try:
            entities = self._run_inference(text)
        except ModelInferenceError:
            raise
        except Exception as exc:
            return PIIResult(
                has_pii=False,
                redacted_text=text,
                error=f"Inference error: {exc}",
            )

        redacted = self._apply_redaction(text, entities)

        return PIIResult(
            has_pii=bool(entities),
            redacted_text=redacted,
            pii_types=sorted({e["type"] for e in entities}),
            entities=entities,
        )

    def batch_detect(self, texts: List[str]) -> List[PIIResult]:
        """
        Sequential single-text inference over a list of strings.
        Per-item errors are captured in PIIResult.error; the list always has
        the same length as the input.

        Raises:
            EmptyInputError  if texts is an empty list.
        """
        if not texts:
            raise EmptyInputError("texts list")

        results = []
        for idx, text in enumerate(texts):
            try:
                self._validate_text(text, position=idx)
                result = self.detect(text)
            except (EmptyInputError, InputTooLargeError, InvalidInputTypeError) as exc:
                result = PIIResult(
                    has_pii=False,
                    redacted_text=str(text) if not isinstance(text, str) else text,
                    error=str(exc),
                )
            results.append(result)
        return results

    def get_pii_statistics(self, results: List[PIIResult]) -> Dict:
        """Aggregate counts across a list of PIIResult objects."""
        total    = len(results)
        with_pii = sum(1 for r in results if r.has_pii)
        type_counts: Dict[str, int] = {}
        for r in results:
            for t in r.pii_types:
                type_counts[t] = type_counts.get(t, 0) + 1
        return {
            "total_texts":           total,
            "texts_with_pii":        with_pii,
            "texts_without_pii":     total - with_pii,
            "pii_rate":              round(with_pii / total, 4) if total > 0 else 0.0,
            "pii_type_distribution": dict(sorted(type_counts.items(), key=lambda x: -x[1])),
            "errors":                sum(1 for r in results if r.error),
        }

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _validate_model_path(self):
        if not self.model_path.exists():
            raise ModelNotFoundError(str(self.model_path))
        missing = _REQUIRED_MODEL_FILES - {f.name for f in self.model_path.iterdir()}
        if missing:
            raise ModelLoadError(
                str(self.model_path),
                f"Missing required files: {', '.join(sorted(missing))}",
            )

    def _validate_text(self, text, position: int = None):
        if not isinstance(text, str):
            raise InvalidInputTypeError(type(text).__name__, position)
        if not text.strip():
            raise EmptyInputError("text")
        if len(text) > MAX_CHARS:
            raise InputTooLargeError(len(text), MAX_CHARS)

    def _load_model(self):
        architecture = getattr(self.config, "pii_model_architecture", "flat")
        if architecture == "hierarchical":
            if HierarchicalPIIModel is None:
                raise ModelLoadError(
                    str(self.model_path),
                    "Hierarchical model class is unavailable.",
                )
            return HierarchicalPIIModel.from_pretrained(
                str(self.model_path),
                config=self.config,
                num_fine_labels=len(self.label2id),
                num_coarse_labels=len(COARSE_NAMES),
                coarse_weight=float(getattr(self.config, "pii_coarse_loss_weight", 0.3)),
            )

        return AutoModelForTokenClassification.from_pretrained(str(self.model_path))

    def _prepare_model_text(self, text: str) -> Tuple[str, int]:
        if not getattr(self, "source_conditioned", False):
            return text, 0
        prefix = f"{self.default_source_token} "
        return prefix + text, len(prefix)

    def _shift_offsets(
        self,
        offset_mapping: List[Tuple[int, int]],
        offset_shift: int,
    ) -> List[Tuple[int, int]]:
        if offset_shift == 0:
            return offset_mapping

        shifted = []
        for char_start, char_end in offset_mapping:
            if char_start == 0 and char_end == 0:
                shifted.append((0, 0))
            elif char_end <= offset_shift:
                shifted.append((0, 0))
            else:
                shifted.append((max(0, char_start - offset_shift),
                                max(0, char_end - offset_shift)))
        return shifted

    @torch.inference_mode()
    def _run_inference(self, text: str) -> List[Dict]:
        try:
            model_text, offset_shift = self._prepare_model_text(text)
            encoding = self.tokenizer(
                model_text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=False,
                return_offsets_mapping=True,
            )
            offset_mapping = self._shift_offsets(
                encoding.pop("offset_mapping")[0].tolist(),
                offset_shift,
            )
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            logits      = self.model(**encoding).logits
            probs       = torch.softmax(logits, dim=-1)[0]
            pred_ids    = torch.argmax(probs, dim=-1).cpu().numpy()
            confidences = probs.max(dim=-1).values.cpu().numpy()
        except Exception as exc:
            raise ModelInferenceError(str(exc)) from exc

        return self._extract_entities(text, pred_ids, confidences, offset_mapping)

    def _extract_entities(
        self,
        text: str,
        pred_ids: np.ndarray,
        confidences: np.ndarray,
        offset_mapping: List[Tuple[int, int]],
    ) -> List[Dict]:
        entities      = []
        current_type  = None
        current_start = None
        current_end   = None
        current_confs: List[float] = []

        for pred_id, conf, (char_start, char_end) in zip(
            pred_ids, confidences, offset_mapping
        ):
            if char_start == 0 and char_end == 0:
                continue

            label = self.id2label.get(int(pred_id), "O")
            if float(conf) < self.confidence_threshold:
                label = "O"

            if label.startswith("B-"):
                if current_type is not None:
                    entities.append(
                        self._make_entity(text, current_type, current_start,
                                          current_end, current_confs)
                    )
                current_type  = label[2:]
                current_start = char_start
                current_end   = char_end
                current_confs = [float(conf)]

            elif label.startswith("I-") and current_type == label[2:]:
                current_end = char_end
                current_confs.append(float(conf))

            else:
                if current_type is not None:
                    entities.append(
                        self._make_entity(text, current_type, current_start,
                                          current_end, current_confs)
                    )
                current_type  = None
                current_start = None
                current_end   = None
                current_confs = []

        if current_type is not None:
            entities.append(
                self._make_entity(text, current_type, current_start,
                                  current_end, current_confs)
            )

        return entities

    def _make_entity(
        self,
        text: str,
        etype: str,
        start: int,
        end: int,
        confs: List[float],
    ) -> Dict:
        return {
            "text":       text[start:end],
            "type":       etype,
            "start":      start,
            "end":        end,
            "confidence": float(np.mean(confs)),
        }

    def _apply_redaction(self, text: str, entities: List[Dict]) -> str:
        if not entities:
            return text
        sorted_entities = sorted(entities, key=lambda e: e["start"], reverse=True)
        out = text
        for entity in sorted_entities:
            out = out[: entity["start"]] + "[REDACTED]" + out[entity["end"]:]
        return out


# ---------------------------------------------------------------------------
# FastPIIDetector — padded-batch inference
# ---------------------------------------------------------------------------

class FastPIIDetector(PIIDetector):
    """
    Batched inference using padded forward passes.
    Significantly faster than sequential detect() when processing large volumes.
    """

    def __init__(
        self,
        model_path: str,
        confidence_threshold: float = 0.5,
        batch_size: int = 32,
        device: Optional[str] = None,
    ):
        super().__init__(model_path, confidence_threshold, device)
        self.batch_size = batch_size

    def batch_detect(self, texts: List[str]) -> List[PIIResult]:
        """
        Batched inference over a list of strings.
        Per-item validation errors and inference errors are captured in
        PIIResult.error; the list always has the same length as the input.

        Raises:
            EmptyInputError  if texts is an empty list.
        """
        if not texts:
            raise EmptyInputError("texts list")

        # Validate all items first; mark invalid ones so we can skip them
        # in the forward pass but still return a result at the correct index.
        validated: List[Optional[str]] = []
        pre_errors: List[Optional[str]] = []
        for idx, text in enumerate(texts):
            try:
                self._validate_text(text, position=idx)
                validated.append(text)
                pre_errors.append(None)
            except (EmptyInputError, InputTooLargeError, InvalidInputTypeError) as exc:
                validated.append(None)
                pre_errors.append(str(exc))

        results: List[Optional[PIIResult]] = [None] * len(texts)

        # Fill pre-error slots immediately
        for idx, err in enumerate(pre_errors):
            if err is not None:
                original = texts[idx]
                results[idx] = PIIResult(
                    has_pii=False,
                    redacted_text=str(original) if not isinstance(original, str) else original,
                    error=err,
                )

        # Collect valid (index, text) pairs for batched inference
        valid_pairs = [(idx, t) for idx, t in enumerate(validated) if t is not None]

        for batch_start in range(0, len(valid_pairs), self.batch_size):
            batch_pairs = valid_pairs[batch_start: batch_start + self.batch_size]
            indices     = [p[0] for p in batch_pairs]
            batch_texts = [p[1] for p in batch_pairs]

            try:
                batch_results = self._run_batch_inference(batch_texts)
            except Exception as exc:
                # Entire batch failed — record error for each item
                for idx, text in zip(indices, batch_texts):
                    results[idx] = PIIResult(
                        has_pii=False,
                        redacted_text=text,
                        error=f"Batch inference error: {exc}",
                    )
                continue

            for idx, text, entities in zip(indices, batch_texts, batch_results):
                redacted = self._apply_redaction(text, entities)
                results[idx] = PIIResult(
                    has_pii=bool(entities),
                    redacted_text=redacted,
                    pii_types=sorted({e["type"] for e in entities}),
                    entities=entities,
                )

        return results

    @torch.inference_mode()
    def _run_batch_inference(self, texts: List[str]) -> List[List[Dict]]:
        try:
            encoding = self.tokenizer(
                [self._prepare_model_text(text)[0] for text in texts],
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True,
                return_offsets_mapping=True,
            )
            raw_offsets = encoding.pop("offset_mapping").tolist()
            offset_mappings = [
                self._shift_offsets(offsets, self._prepare_model_text(text)[1])
                for text, offsets in zip(texts, raw_offsets)
            ]
            encoding = {k: v.to(self.device) for k, v in encoding.items()}

            logits = self.model(**encoding).logits
            probs  = torch.softmax(logits, dim=-1)
        except Exception as exc:
            raise ModelInferenceError(str(exc)) from exc

        all_entities = []
        for j, (text, offsets) in enumerate(zip(texts, offset_mappings)):
            pred_ids = torch.argmax(probs[j], dim=-1).cpu().numpy()
            confs    = probs[j].max(dim=-1).values.cpu().numpy()
            all_entities.append(
                self._extract_entities(text, pred_ids, confs, offsets)
            )
        return all_entities


# ---------------------------------------------------------------------------
# Smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    detector = PIIDetector(model_path="./models/best_model")

    samples = [
        "My name is Pritesh Jha and my email is pritesh.jha@example.com",
        "Call me at +91-98765-43210 or reach John Doe at john.doe@corp.in",
        "The quarterly revenue increased by 12 percent.",
        "SSN: 123-45-6789, DOB: 01/15/1990, card ending 4242",
        "Wire transfer from HDFC account 50100123456789 to JP Morgan Chase",
    ]

    print("=" * 60)
    print("PIIDetector — smoke test")
    print("=" * 60)

    for text in samples:
        result = detector.detect(text)
        print(f"\nText         : {text}")
        print(f"has_pii      : {result.has_pii}")
        print(f"redacted_text: {result.redacted_text}")
        if result.entities:
            for e in result.entities:
                print(f"  [{e['start']}:{e['end']}] {e['text']!r:30s} | "
                      f"{e['type']:25s} | conf={e['confidence']:.3f}")

    print("\n" + "=" * 60)
    print("FastPIIDetector — batch smoke test")
    print("=" * 60)
    fast = FastPIIDetector(model_path="./models/best_model", batch_size=4)
    batch_results = fast.batch_detect(samples)
    stats = fast.get_pii_statistics(batch_results)
    print(f"\nStats: {stats}")
