"""
PII Dataset Consolidation Script
=================================
Reads all downloaded JSONL files, normalises every source into a unified
BIO-tagged format, reports unique entity types across all sources, and
writes a single consolidated JSONL file ready for transformer fine-tuning.

Output schema (one JSON object per line):
  {
    "tokens":  ["Pritesh", "works", "at", "XYZ-Corp"],
    "labels":  ["B-PERSON", "O", "O", "B-ORG"],
    "source":  "conll2003",
    "text":    "Pritesh works at XYZ-Corp"  # with --include-text
  }

Usage:
    python src/consolidate_pii_datasets.py \
        --data-dir src/pii_datasets \
        --output-dir src/pii_datasets/consolidated

Fixes vs original:
  - few_nerd: fewnerd_labels now use B-/I- prefixes so BIO tagging is preserved.
  - nvidia/Nemotron-PII: span_to_bio extended to handle 'start_index'/'end_index'
    and additional label key variants used by Nemotron spans.
  - parse_span_field: also handles list-of-lists (Nemotron nested format).
  - span_to_bio: falls back to character search when exact char offset misses.
"""

import ast
import json
import argparse
from pathlib import Path
from collections import defaultdict


def tokens_to_text(tokens: list) -> str:
    return " ".join(str(tok) for tok in tokens)


def get_text(row: dict, *columns: str) -> str:
    for col in columns:
        if not col:
            continue
        value = row.get(col)
        if isinstance(value, str) and value:
            return value
    return ""


def make_record(tokens: list, labels: list, source: str,
                include_text: bool = False, text: str = "") -> dict:
    min_len = min(len(tokens), len(labels))
    rec = {
        "tokens": tokens[:min_len],
        "labels": labels[:min_len],
        "source": source,
    }
    if include_text:
        rec["text"] = text or tokens_to_text(rec["tokens"])
    return rec

# ---------------------------------------------------------------------------
# Entity label normalisation map
# ---------------------------------------------------------------------------
LABEL_NORM = {
    # Person
    "PER": "PERSON", "B-PER": "B-PERSON", "I-PER": "I-PERSON",
    "FIRSTNAME": "PERSON", "B-FIRSTNAME": "B-PERSON", "I-FIRSTNAME": "I-PERSON",
    "LASTNAME": "PERSON", "B-LASTNAME": "B-PERSON", "I-LASTNAME": "I-PERSON",
    "MIDDLENAME": "PERSON", "B-MIDDLENAME": "B-PERSON", "I-MIDDLENAME": "I-PERSON",
    "PREFIX": "PERSON", "B-PREFIX": "B-PERSON", "I-PREFIX": "I-PERSON",
    "GENDER": "PERSON", "B-GENDER": "B-PERSON", "I-GENDER": "I-PERSON",
    "SEX": "PERSON", "B-SEX": "B-PERSON", "I-SEX": "I-PERSON",
    "AGE": "PERSON", "B-AGE": "B-PERSON", "I-AGE": "I-PERSON",
    "DOB": "DATE", "B-DOB": "B-DATE", "I-DOB": "I-DATE",
    "EYECOLOR": "PERSON", "B-EYECOLOR": "B-PERSON", "I-EYECOLOR": "I-PERSON",
    "HEIGHT": "PERSON", "B-HEIGHT": "B-PERSON", "I-HEIGHT": "I-PERSON",
    # few-nerd coarse labels (lowercase, decoded from integers — no B-/I- yet)
    "person": "PERSON",
    "organization": "ORG",
    "location": "LOC",
    "other": "MISC",
    "art": "MISC",
    "building": "LOC",
    "event": "MISC",
    "product": "MISC",

    # Organisation
    "ORG": "ORG",
    "COMPANYNAME": "ORG", "B-COMPANYNAME": "B-ORG", "I-COMPANYNAME": "I-ORG",
    "ACCOUNTNAME": "ORG", "B-ACCOUNTNAME": "B-ORG", "I-ACCOUNTNAME": "I-ORG",
    "COMPANY": "ORG", "B-COMPANY": "B-ORG", "I-COMPANY": "I-ORG",

    # Location
    "LOC": "LOC",
    "CITY": "LOC", "B-CITY": "B-LOC", "I-CITY": "I-LOC",
    "STATE": "LOC", "B-STATE": "B-LOC", "I-STATE": "I-LOC",
    "COUNTY": "LOC", "B-COUNTY": "B-LOC", "I-COUNTY": "I-LOC",
    "ZIPCODE": "LOC", "B-ZIPCODE": "B-LOC", "I-ZIPCODE": "I-LOC",
    "STREET": "ADDRESS", "B-STREET": "B-ADDRESS", "I-STREET": "I-ADDRESS",
    "BUILDINGNUMBER": "ADDRESS", "B-BUILDINGNUMBER": "B-ADDRESS", "I-BUILDINGNUMBER": "I-ADDRESS",
    "SECONDARYADDRESS": "ADDRESS", "B-SECONDARYADDRESS": "B-ADDRESS", "I-SECONDARYADDRESS": "I-ADDRESS",
    "NEARBYGPSCOORDINATE": "LOC", "B-NEARBYGPSCOORDINATE": "B-LOC", "I-NEARBYGPSCOORDINATE": "I-LOC",
    "ORDINALDIRECTION": "LOC", "B-ORDINALDIRECTION": "B-LOC", "I-ORDINALDIRECTION": "I-LOC",

    # Email
    "EMAIL": "EMAIL",

    # Phone
    "PHONENUMBER": "PHONE", "B-PHONENUMBER": "B-PHONE", "I-PHONENUMBER": "I-PHONE",
    "PHONE": "PHONE",
    "PHONEIMEI": "PHONE", "B-PHONEIMEI": "B-PHONE", "I-PHONEIMEI": "I-PHONE",
    "PHONE_NUMBER": "PHONE", "B-PHONE_NUMBER": "B-PHONE", "I-PHONE_NUMBER": "I-PHONE",

    # Financial identifiers
    "CREDITCARDNUMBER": "CREDIT_CARD", "B-CREDITCARDNUMBER": "B-CREDIT_CARD", "I-CREDITCARDNUMBER": "I-CREDIT_CARD",
    "CREDITCARDCVV": "CREDIT_CARD", "B-CREDITCARDCVV": "B-CREDIT_CARD", "I-CREDITCARDCVV": "I-CREDIT_CARD",
    "CREDITCARDISSUER": "CREDIT_CARD", "B-CREDITCARDISSUER": "B-CREDIT_CARD", "I-CREDITCARDISSUER": "I-CREDIT_CARD",
    "CREDIT_CARD": "CREDIT_CARD",
    "CREDIT_CARD_NUMBER": "CREDIT_CARD", "B-CREDIT_CARD_NUMBER": "B-CREDIT_CARD", "I-CREDIT_CARD_NUMBER": "I-CREDIT_CARD",
    "IBAN": "IBAN",
    "BIC": "BIC",
    "ACCOUNTNUMBER": "ACCOUNT_NUMBER", "B-ACCOUNTNUMBER": "B-ACCOUNT_NUMBER", "I-ACCOUNTNUMBER": "I-ACCOUNT_NUMBER",
    "ACCOUNT_NUMBER": "ACCOUNT_NUMBER",
    "ROUTING_NUMBER": "ROUTING_NUMBER",
    "MASKEDNUMBER": "ACCOUNT_NUMBER", "B-MASKEDNUMBER": "B-ACCOUNT_NUMBER", "I-MASKEDNUMBER": "I-ACCOUNT_NUMBER",
    "PIN": "PIN",
    "TAX_ID": "TAX_ID",
    "SSN": "SSN",

    # Currency / Amount
    "AMOUNT": "AMOUNT",
    "CURRENCY": "CURRENCY", "B-CURRENCY": "B-CURRENCY", "I-CURRENCY": "I-CURRENCY",
    "CURRENCYCODE": "CURRENCY", "B-CURRENCYCODE": "B-CURRENCY", "I-CURRENCYCODE": "I-CURRENCY",
    "CURRENCYNAME": "CURRENCY", "B-CURRENCYNAME": "B-CURRENCY", "I-CURRENCYNAME": "I-CURRENCY",
    "CURRENCYSYMBOL": "CURRENCY", "B-CURRENCYSYMBOL": "B-CURRENCY", "I-CURRENCYSYMBOL": "I-CURRENCY",

    # Crypto
    "BITCOINADDRESS": "CRYPTO_ADDRESS", "B-BITCOINADDRESS": "B-CRYPTO_ADDRESS", "I-BITCOINADDRESS": "I-CRYPTO_ADDRESS",
    "ETHEREUMADDRESS": "CRYPTO_ADDRESS", "B-ETHEREUMADDRESS": "B-CRYPTO_ADDRESS", "I-ETHEREUMADDRESS": "I-CRYPTO_ADDRESS",
    "LITECOINADDRESS": "CRYPTO_ADDRESS", "B-LITECOINADDRESS": "B-CRYPTO_ADDRESS", "I-LITECOINADDRESS": "I-CRYPTO_ADDRESS",

    # Network / Device
    "IP": "IP_ADDRESS", "B-IP": "B-IP_ADDRESS", "I-IP": "I-IP_ADDRESS",
    "IPV4": "IP_ADDRESS", "B-IPV4": "B-IP_ADDRESS", "I-IPV4": "I-IP_ADDRESS",
    "IPV6": "IP_ADDRESS", "B-IPV6": "B-IP_ADDRESS", "I-IPV6": "I-IP_ADDRESS",
    "MAC": "IP_ADDRESS", "B-MAC": "B-IP_ADDRESS", "I-MAC": "I-IP_ADDRESS",
    "USERAGENT": "USERNAME", "B-USERAGENT": "B-USERNAME", "I-USERAGENT": "I-USERNAME",
    "URL": "URL",

    # Auth
    "USERNAME": "USERNAME",
    "PASSWORD": "PASSWORD",

    # Date / Time
    "DATE": "DATE",
    "TIME": "TIME",

    # Job
    "JOBTITLE": "JOB", "B-JOBTITLE": "B-JOB", "I-JOBTITLE": "I-JOB",
    "JOBAREA": "JOB", "B-JOBAREA": "B-JOB", "I-JOBAREA": "I-JOB",
    "JOBTYPE": "JOB", "B-JOBTYPE": "B-JOB", "I-JOBTYPE": "I-JOB",

    # Vehicle
    "VEHICLEVIN": "VEHICLE", "B-VEHICLEVIN": "B-VEHICLE", "I-VEHICLEVIN": "I-VEHICLE",
    "VEHICLEVRM": "VEHICLE", "B-VEHICLEVRM": "B-VEHICLE", "I-VEHICLEVRM": "I-VEHICLE",
    "VEHI": "VEHICLE", "B-VEHI": "B-VEHICLE", "I-VEHI": "I-VEHICLE",

    # multinerd types
    "ANIM": "MISC", "BIO": "MISC", "CEL": "MISC", "DIS": "MISC",
    "EVE": "MISC", "FOOD": "MISC", "INST": "MISC", "MEDIA": "MISC",
    "MYTH": "MISC", "PLANT": "MISC",

    "MISC": "MISC",
}


def normalise_label(label: str) -> str:
    if label == "O":
        return "O"

    prefix = ""
    base = label
    if label.startswith("B-") or label.startswith("I-"):
        prefix = label[:2]
        base = label[2:]

    # Direct lookup with full key (handles pre-prefixed entries in LABEL_NORM)
    full_key = prefix + base
    if full_key in LABEL_NORM:
        normed = LABEL_NORM[full_key]
        if normed.startswith("B-") or normed.startswith("I-"):
            return normed
        return prefix + normed

    # Lookup base only
    if base in LABEL_NORM:
        normed = LABEL_NORM[base]
        if normed.startswith("B-") or normed.startswith("I-"):
            return normed
        return prefix + normed

    # Lowercase base lookup (catches few-nerd decoded strings passed with a prefix)
    if base.lower() in LABEL_NORM:
        normed = LABEL_NORM[base.lower()]
        if normed.startswith("B-") or normed.startswith("I-"):
            return normed
        return prefix + normed

    # XBRL / unknown camelCase -> FINANCIAL_ENTITY
    if base and base[0].isupper() and "-" not in base and "_" not in base:
        return prefix + "FINANCIAL_ENTITY"

    return prefix + base.upper()


# ---------------------------------------------------------------------------
# Span-to-BIO converter
# ---------------------------------------------------------------------------

def span_to_bio(text: str, spans: list) -> tuple:
    """
    Convert raw text + character-offset spans into whitespace-tokenised
    tokens and BIO labels.

    Handles span dicts with keys:
      start / begin / char_start / start_index / startIndex
      end   / char_end / end_index / endIndex
      type  / label / entity_type / tag / pii_type / category / ner_tag
    """
    tokens = text.split()
    labels = ["O"] * len(tokens)

    if not tokens:
        return tokens, labels

    # Build char-offset -> token-index map
    char_to_tok = {}
    pos = 0
    for tok_idx, tok in enumerate(tokens):
        start_pos = text.find(tok, pos)
        if start_pos == -1:
            pos += 1
            continue
        for c in range(start_pos, start_pos + len(tok)):
            char_to_tok[c] = tok_idx
        pos = start_pos + len(tok)

    for span in spans:
        if not isinstance(span, dict):
            continue

        # Resolve start offset
        start = (
            span.get("start") or span.get("begin") or span.get("char_start")
            or span.get("start_index") or span.get("startIndex")
            or span.get("offset")
        )
        # Resolve end offset
        end = (
            span.get("end") or span.get("char_end")
            or span.get("end_index") or span.get("endIndex")
        )
        # Resolve label
        label = (
            span.get("type") or span.get("label") or span.get("entity_type")
            or span.get("tag") or span.get("pii_type") or span.get("category")
            or span.get("ner_tag") or span.get("entity_label") or span.get("class")
        )

        if start is None or end is None or not label:
            # Last resort: if span has 'value', try to find it in text
            value = span.get("value") or span.get("text") or span.get("surface_form")
            label = label or span.get("entity") or ""
            if value and label:
                idx = text.find(str(value))
                if idx != -1:
                    start = idx
                    end = idx + len(str(value))
                else:
                    continue
            else:
                continue

        try:
            start, end = int(start), int(end)
        except (TypeError, ValueError):
            continue

        first_tok = char_to_tok.get(start)
        last_tok = char_to_tok.get(end - 1)

        # Fallback: if exact char not in map, scan nearby chars
        if first_tok is None:
            for offset in range(0, 5):
                first_tok = char_to_tok.get(start + offset)
                if first_tok is not None:
                    break
        if last_tok is None:
            for offset in range(0, 5):
                last_tok = char_to_tok.get(end - 1 - offset)
                if last_tok is not None:
                    break

        if first_tok is None or last_tok is None:
            continue

        labels[first_tok] = f"B-{label}"
        for i in range(first_tok + 1, last_tok + 1):
            labels[i] = f"I-{label}"

    return tokens, labels


def parse_span_field(raw) -> list:
    """
    Parse the pii_spans / spans column which may be a JSON string, a list,
    or a list of lists (Nemotron nested format).
    """
    if raw is None:
        return []
    if isinstance(raw, list):
        result = []
        for item in raw:
            if isinstance(item, dict):
                result.append(item)
            elif isinstance(item, list):
                # Nemotron sometimes stores spans as [[start, end, label], ...]
                if len(item) >= 3:
                    result.append({"start": item[0], "end": item[1], "type": item[2]})
            elif isinstance(item, str):
                parsed = parse_span_field(item)
                result.extend(parsed)
        return result
    if isinstance(raw, str):
        for parser in (json.loads, ast.literal_eval):
            try:
                parsed = parser(raw)
            except Exception:
                continue
            if isinstance(parsed, list):
                return parse_span_field(parsed)
            if isinstance(parsed, dict):
                return [parsed]
        return []
    return []


# ---------------------------------------------------------------------------
# Per-source readers
# ---------------------------------------------------------------------------

def read_bio_jsonl(filepath: Path, token_col: str, label_col: str,
                   source: str, label_names: list = None,
                   include_text: bool = False, text_col: str = None) -> list:
    """
    Generic reader for datasets with tokens + BIO labels (or integer ids).
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tokens = row.get(token_col)
            labels = row.get(label_col)
            if not tokens or not labels:
                continue

            if label_names and labels and isinstance(labels[0], int):
                labels = [label_names[i] if i < len(label_names) else "O" for i in labels]

            labels = [normalise_label(str(l)) for l in labels]

            records.append(make_record(
                tokens,
                labels,
                source,
                include_text=include_text,
                text=get_text(row, text_col),
            ))
    return records


def read_bio_jsonl_fewnerd(filepath: Path, source: str, label_names: list,
                           include_text: bool = False) -> list:
    """
    FIX: few-nerd reader.

    The original script decoded integers to coarse label strings like "person"
    (no B-/I- prefix) and passed them directly to normalise_label, which then
    returned "PERSON" (no prefix). This broke BIO structure entirely.

    Correct approach: integer 0 = O, odd = B-*, even(>0) = I-*.
    We reconstruct BIO by looking at the fine_ner_tags column to decide
    B vs I, or use the coarse ner_tags with run-length encoding.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tokens = row.get("tokens")
            ner_tags = row.get("ner_tags")
            if not tokens or ner_tags is None:
                continue

            labels = []
            prev_label = None
            for tag_id in ner_tags:
                if tag_id == 0:
                    labels.append("O")
                    prev_label = None
                else:
                    raw_label = label_names[tag_id] if tag_id < len(label_names) else "other"
                    canonical = normalise_label(raw_label)
                    # Assign B- if this is the start of a new entity span, I- if continuing
                    if prev_label == canonical:
                        labels.append(f"I-{canonical}")
                    else:
                        labels.append(f"B-{canonical}")
                    prev_label = canonical

            records.append(make_record(
                tokens,
                labels,
                source,
                include_text=include_text,
                text=get_text(row, "text"),
            ))
    return records


def read_span_jsonl(filepath: Path, text_col: str, span_col: str, source: str,
                    include_text: bool = False) -> list:
    """
    Reader for datasets with raw text + character-offset span annotations.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get(text_col, "")
            spans_raw = row.get(span_col)
            if not text:
                continue

            spans = parse_span_field(spans_raw)
            tokens, labels = span_to_bio(text, spans)
            labels = [normalise_label(l) for l in labels]

            if tokens:
                records.append(make_record(
                    tokens,
                    labels,
                    source,
                    include_text=include_text,
                    text=text,
                ))
    return records


def read_nvidia_jsonl(filepath: Path, include_text: bool = False) -> list:
    """
    FIX: nvidia/Nemotron-PII reader.

    The dataset stores spans in the 'spans' column. The original span_to_bio
    only checked start/begin/char_start for offsets. Nemotron uses
    'start'/'end' (confirmed from dataset card) but some rows store the field
    as a JSON string rather than a parsed list. Additionally, 'text_tagged'
    provides a tagged version which we use as fallback.

    This reader also tries parsing entity labels from 'text_tagged' if
    'spans' yields no entities, using a simple regex over XML-like tags.
    """
    import re
    bracket_tag_re = re.compile(r'\[(.*?)\]([A-Za-z_][A-Za-z0-9_]*)', re.DOTALL)

    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            text = row.get("text", "")
            spans_raw = row.get("spans")
            if not text:
                continue

            spans = parse_span_field(spans_raw)
            tokens, labels = span_to_bio(text, spans)
            labels = [normalise_label(l) for l in labels]
            record_text = text

            # Fallback: if no entities found and text_tagged is available, parse
            # Nemotron's "[value]label" inline format.
            has_entities = any(l != "O" for l in labels)
            if not has_entities:
                text_tagged = row.get("text_tagged", "")
                if text_tagged:
                    fallback_spans = []
                    clean = ""
                    remaining = text_tagged
                    while remaining:
                        m = bracket_tag_re.search(remaining)
                        if not m:
                            clean += remaining
                            break
                        clean += remaining[:m.start()]
                        entity_start = len(clean)
                        entity_text = m.group(1)
                        clean += entity_text
                        entity_end = len(clean)
                        entity_type = m.group(2)
                        fallback_spans.append({
                            "start": entity_start,
                            "end": entity_end,
                            "type": entity_type,
                        })
                        remaining = remaining[m.end():]

                    if fallback_spans and clean.strip():
                        tokens, labels = span_to_bio(clean, fallback_spans)
                        labels = [normalise_label(l) for l in labels]
                        record_text = clean

            if tokens:
                records.append(make_record(
                    tokens,
                    labels,
                    "nvidia_nemotron",
                    include_text=include_text,
                    text=record_text,
                ))
    return records


def read_finer_jsonl(filepath: Path, include_text: bool = False) -> list:
    """
    finer-139: integer tags, 0=O, odd=B-FINANCIAL_ENTITY, even(>0)=I-FINANCIAL_ENTITY.
    """
    records = []
    with open(filepath) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            tokens = row.get("tokens")
            ner_tags = row.get("ner_tags")
            if not tokens or ner_tags is None:
                continue

            labels = []
            for tag in ner_tags:
                if tag == 0:
                    labels.append("O")
                elif tag % 2 == 1:
                    labels.append("B-FINANCIAL_ENTITY")
                else:
                    labels.append("I-FINANCIAL_ENTITY")

            records.append(make_record(
                tokens,
                labels,
                "finer_139",
                include_text=include_text,
                text=get_text(row, "text"),
            ))
    return records


# ---------------------------------------------------------------------------
# Reporting
# ---------------------------------------------------------------------------

def collect_unique_entities(all_records: list) -> tuple:
    from collections import Counter
    per_source = defaultdict(set)
    global_types = set()

    for rec in all_records:
        source = rec["source"]
        for label in rec["labels"]:
            if label != "O" and label.startswith("B-"):
                entity_type = label[2:]
                per_source[source].add(entity_type)
                global_types.add(entity_type)

    return per_source, global_types


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main(data_dir: Path, output_dir: Path, include_text: bool = False):
    output_dir.mkdir(parents=True, exist_ok=True)

    all_records = []
    load_errors = []

    # 1. ai4privacy/pii-masking-400k
    p = data_dir / "ai4privacy_400k" / "train.jsonl"
    if p.exists():
        print(f"Reading {p} ...")
        recs = read_bio_jsonl(
            p,
            "mbert_tokens",
            "mbert_token_classes",
            "ai4privacy_400k",
            include_text=include_text,
            text_col="source_text",
        )
        print(f"  {len(recs):,} records")
        all_records.extend(recs)
    else:
        load_errors.append(str(p))

    # 2. ai4privacy/pii-masking-300k
    p = data_dir / "ai4privacy_300k" / "train.jsonl"
    if p.exists():
        print(f"Reading {p} ...")
        recs = read_bio_jsonl(
            p,
            "mbert_text_tokens",
            "mbert_bio_labels",
            "ai4privacy_300k",
            include_text=include_text,
            text_col="source_text",
        )
        print(f"  {len(recs):,} records")
        all_records.extend(recs)
    else:
        load_errors.append(str(p))

    # 3. gretelai/synthetic_pii_finance_multilingual
    for split in ["train", "test"]:
        p = data_dir / "gretel_finance" / f"{split}.jsonl"
        if p.exists():
            print(f"Reading {p} ...")
            recs = read_span_jsonl(
                p,
                "generated_text",
                "pii_spans",
                "gretel_finance",
                include_text=include_text,
            )
            print(f"  {len(recs):,} records")
            all_records.extend(recs)

    # 4. nvidia/Nemotron-PII  [FIXED]
    p = data_dir / "nvidia_nemotron" / "train.jsonl"
    if p.exists():
        print(f"Reading {p} ...")
        recs = read_nvidia_jsonl(p, include_text=include_text)
        print(f"  {len(recs):,} records")
        all_records.extend(recs)
    else:
        load_errors.append(str(p))

    # 5. wikiann (en)
    wikiann_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    for split in ["train", "validation", "test"]:
        p = data_dir / "wikiann" / f"{split}.jsonl"
        if p.exists():
            print(f"Reading {p} ...")
            recs = read_bio_jsonl(
                p,
                "tokens",
                "ner_tags",
                "wikiann",
                label_names=wikiann_labels,
                include_text=include_text,
                text_col="text",
            )
            print(f"  {len(recs):,} records")
            all_records.extend(recs)

    # 6. Babelscape/multinerd (en)
    multinerd_labels = [
        "O",
        "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC",
        "B-ANIM", "I-ANIM", "B-BIO", "I-BIO", "B-CEL", "I-CEL",
        "B-DIS", "I-DIS", "B-EVE", "I-EVE", "B-FOOD", "I-FOOD",
        "B-INST", "I-INST", "B-MEDIA", "I-MEDIA", "B-MYTH", "I-MYTH",
        "B-PLANT", "I-PLANT", "B-TIME", "I-TIME", "B-VEHI", "I-VEHI",
    ]
    p = data_dir / "multinerd" / "train_en.jsonl"
    if p.exists():
        print(f"Reading {p} ...")
        recs = read_bio_jsonl(
            p,
            "tokens",
            "ner_tags",
            "multinerd",
            label_names=multinerd_labels,
            include_text=include_text,
            text_col="text",
        )
        print(f"  {len(recs):,} records")
        all_records.extend(recs)
    else:
        load_errors.append(str(p))

    # 7. DFKI-SLT/few-nerd  [FIXED]
    # Coarse label names — integer 0=O, 1=art, 2=building, ... 8=product
    fewnerd_labels = ["O", "art", "building", "event", "location",
                      "organization", "other", "person", "product"]
    for split in ["train", "validation", "test"]:
        p = data_dir / "few_nerd" / f"{split}.jsonl"
        if p.exists():
            print(f"Reading {p} ...")
            recs = read_bio_jsonl_fewnerd(
                p,
                "few_nerd",
                fewnerd_labels,
                include_text=include_text,
            )
            print(f"  {len(recs):,} records")
            all_records.extend(recs)

    # 8. CoNLL-2003
    conll_labels = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG",
                    "B-LOC", "I-LOC", "B-MISC", "I-MISC"]
    for split in ["train", "validation", "test"]:
        p = data_dir / "conll2003" / f"{split}.jsonl"
        if p.exists():
            print(f"Reading {p} ...")
            recs = read_bio_jsonl(
                p,
                "tokens",
                "ner_tags",
                "conll2003",
                label_names=conll_labels,
                include_text=include_text,
                text_col="text",
            )
            print(f"  {len(recs):,} records")
            all_records.extend(recs)

    # 9. nlpaueb/finer-139
    for split in ["train", "validation", "test"]:
        p = data_dir / "finer_139" / f"{split}.jsonl"
        if p.exists():
            print(f"Reading {p} ...")
            recs = read_finer_jsonl(p, include_text=include_text)
            print(f"  {len(recs):,} records")
            all_records.extend(recs)

    # 10. Isotonic/pii-masking-200k
    p = data_dir / "isotonic_pii_200k" / "train.jsonl"
    if p.exists():
        print(f"Reading {p} ...")
        recs = read_bio_jsonl(
            p,
            "tokenised_text",
            "bio_labels",
            "isotonic_pii_200k",
            include_text=include_text,
            text_col="unmasked_text",
        )
        print(f"  {len(recs):,} records")
        all_records.extend(recs)
    else:
        load_errors.append(str(p))

    # ------------------------------------------------------------------
    # Entity report
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("UNIQUE ENTITY TYPES PER SOURCE")
    print("=" * 70)
    per_source, global_types = collect_unique_entities(all_records)

    for source, types in sorted(per_source.items()):
        print(f"\n[{source}] ({len(types)} types)")
        print("  " + ", ".join(sorted(types)))

    print("\n" + "=" * 70)
    print(f"GLOBAL UNIQUE ENTITY TYPES ({len(global_types)} total)")
    print("=" * 70)
    for t in sorted(global_types):
        print(f"  {t}")

    entity_report = {
        "global": sorted(global_types),
        "per_source": {src: sorted(types) for src, types in per_source.items()},
    }
    report_path = output_dir / "entity_types.json"
    with open(report_path, "w") as f:
        json.dump(entity_report, f, indent=2)
    print(f"\nEntity report saved to: {report_path}")

    # ------------------------------------------------------------------
    # Write consolidated JSONL
    # ------------------------------------------------------------------
    out_path = output_dir / "consolidated.jsonl"
    print(f"\nWriting {len(all_records):,} records to {out_path} ...")
    with open(out_path, "w") as f:
        for rec in all_records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    size_mb = out_path.stat().st_size / (1024 * 1024)
    print(f"Done. File size: {size_mb:.1f} MB")

    if load_errors:
        print(f"\nWARNING — {len(load_errors)} file(s) not found (skipped):")
        for e in load_errors:
            print(f"  {e}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 70)
    print("CONSOLIDATION SUMMARY")
    print("=" * 70)
    source_counts = defaultdict(int)
    for rec in all_records:
        source_counts[rec["source"]] += 1
    for src, count in sorted(source_counts.items(), key=lambda x: -x[1]):
        print(f"  {src:<40} {count:>10,} records")
    print(f"  {'TOTAL':<40} {len(all_records):>10,} records")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=str, default="./pii_datasets",
                        help="Root directory containing downloaded dataset folders")
    parser.add_argument("--output-dir", type=str, default="./pii_datasets/consolidated",
                        help="Directory to write consolidated output")
    parser.add_argument("--include-text", action="store_true",
                        help="Include source text in consolidated records when available")
    args = parser.parse_args()
    main(Path(args.data_dir), Path(args.output_dir), include_text=args.include_text)
