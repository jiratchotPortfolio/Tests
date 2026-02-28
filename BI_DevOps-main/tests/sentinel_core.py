"""
sentinel_core.py
Core logic extracted from the Sentinel pipeline.
Depends only on Python stdlib + beautifulsoup4.
"""

import re
import hashlib
import unicodedata
import json
from html.parser import HTMLParser
from typing import Optional


# ---------------------------------------------------------------------------
# PII / Boilerplate patterns
# ---------------------------------------------------------------------------
_EMAIL_RE  = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE  = re.compile(r"\+?[\d\s\-().]{7,15}\d")
_CARD_RE   = re.compile(r"\b(?:\d[ -]?){13,16}\b")

_BOILERPLATE = [
    re.compile(r"This (email|message|communication) (is |may be )?confidential.*",
               re.IGNORECASE | re.DOTALL),
    re.compile(r"Sent from my (iPhone|Galaxy|Pixel|Android).*", re.IGNORECASE),
    re.compile(r"Thank you for contacting .{0,50} support\.", re.IGNORECASE),
    re.compile(r"(\[Ticket #\d+\]|\[Case #\d+\])", re.IGNORECASE),
]

REQUIRED_FIELDS = {"external_ref_id", "agent_id", "channel", "started_at",
                   "resolution_status", "turns"}
VALID_CHANNELS  = {"live_chat", "email", "phone", "in_app"}
VALID_STATUS    = {"resolved", "escalated", "abandoned", "pending"}
VALID_ROLES     = {"customer", "agent", "bot"}


# ---------------------------------------------------------------------------
# HTML stripper
# ---------------------------------------------------------------------------
class _HTMLStripper(HTMLParser):
    def __init__(self):
        super().__init__()
        self.parts = []

    def handle_data(self, data):
        self.parts.append(data)


def strip_html(text: str) -> str:
    s = _HTMLStripper()
    s.feed(text)
    return " ".join(s.parts)


# ---------------------------------------------------------------------------
# Pre-processing steps
# ---------------------------------------------------------------------------
def strip_boilerplate(text: str) -> str:
    for pattern in _BOILERPLATE:
        text = pattern.sub("", text)
    return text


def redact_pii(text: str) -> str:
    text = _EMAIL_RE.sub("[EMAIL]", text)
    text = _CARD_RE.sub("[CARD]", text)   # card before phone: phone regex is greedy over digits
    text = _PHONE_RE.sub("[PHONE]", text)
    return text


def normalise_unicode(text: str) -> str:
    return unicodedata.normalize("NFKC", text)


def estimate_tokens(text: str) -> int:
    return max(1, len(text) // 4)


# ---------------------------------------------------------------------------
# Schema validation (no Pydantic — pure stdlib)
# ---------------------------------------------------------------------------
class ValidationError(Exception):
    pass


def validate_record(raw: dict) -> dict:
    missing = REQUIRED_FIELDS - raw.keys()
    if missing:
        raise ValidationError(f"Missing fields: {missing}")
    if raw["channel"] not in VALID_CHANNELS:
        raise ValidationError(f"Invalid channel: {raw['channel']}")
    if raw["resolution_status"] not in VALID_STATUS:
        raise ValidationError(f"Invalid status: {raw['resolution_status']}")
    for turn in raw.get("turns", []):
        if turn.get("speaker_role", "").lower() not in VALID_ROLES:
            raise ValidationError(f"Invalid speaker_role: {turn.get('speaker_role')}")
    return raw


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------
def process_record(raw: dict) -> Optional[dict]:
    try:
        validate_record(raw)
    except ValidationError:
        return None

    cleaned_turns = []
    for turn in raw["turns"]:
        text = turn["message_text"]
        text = strip_html(text)
        text = strip_boilerplate(text)
        text = redact_pii(text)
        text = normalise_unicode(text)
        text = text.strip()
        if not text:
            continue
        cleaned_turns.append({
            "speaker_role": turn["speaker_role"].lower(),
            "message_text": text,
            "message_tokens": estimate_tokens(text),
            "sent_at": turn["sent_at"],
        })

    if not cleaned_turns:
        return None

    full_text = " ".join(t["message_text"] for t in cleaned_turns)
    checksum  = hashlib.sha256(full_text.encode()).hexdigest()

    return {**raw, "turns": cleaned_turns,
            "full_text": full_text, "checksum": checksum}


# ---------------------------------------------------------------------------
# Embedding cache simulation
# ---------------------------------------------------------------------------
_embed_cache: dict[str, list[float]] = {}


def embed_text(text: str) -> list[float]:
    """Deterministic mock embedding: SHA-256 → 8-dim float vector."""
    h = hashlib.sha256(text.encode()).digest()
    vec = [b / 255.0 for b in h[:8]]
    return vec


def embed_with_cache(text: str) -> tuple[list[float], bool]:
    """Returns (vector, cache_hit)."""
    key = hashlib.sha256(text.encode()).hexdigest()
    if key in _embed_cache:
        return _embed_cache[key], True
    vec = embed_text(text)
    _embed_cache[key] = vec
    return vec, False


def clear_embed_cache():
    _embed_cache.clear()


# ---------------------------------------------------------------------------
# Cosine similarity (for retrieval tests)
# ---------------------------------------------------------------------------
def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot  = sum(x * y for x, y in zip(a, b))
    na   = sum(x * x for x in a) ** 0.5
    nb   = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)


def top_k(query_vec: list[float], corpus: list[tuple[str, list[float]]], k: int) -> list[str]:
    scored = [(doc_id, cosine_similarity(query_vec, vec)) for doc_id, vec in corpus]
    scored.sort(key=lambda x: x[1], reverse=True)
    return [doc_id for doc_id, _ in scored[:k]]


# ---------------------------------------------------------------------------
# SQL query builder (no DB — just validates query structure)
# ---------------------------------------------------------------------------
ALLOWED_FILTER_KEYS = {"date_from", "date_to", "channel", "topic_id", "resolution_status"}


def build_metadata_query(filters: dict) -> tuple[str, list]:
    unknown = set(filters.keys()) - ALLOWED_FILTER_KEYS
    if unknown:
        raise ValueError(f"Unknown filter keys: {unknown}")

    clauses, params = [], []
    if "date_from" in filters:
        clauses.append("c.started_at >= %s"); params.append(filters["date_from"])
    if "date_to" in filters:
        clauses.append("c.started_at <= %s"); params.append(filters["date_to"])
    if "channel" in filters:
        clauses.append("c.channel = %s"); params.append(filters["channel"])
    if "topic_id" in filters:
        clauses.append("c.topic_id = %s"); params.append(filters["topic_id"])
    if "resolution_status" in filters:
        clauses.append("c.resolution_status = %s"); params.append(filters["resolution_status"])

    where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
    sql = f"SELECT * FROM conversations c {where} ORDER BY c.started_at DESC LIMIT 100"
    return sql, params
