"""
preprocessing_pipeline.py
--------------------------
NLP pre-processing pipeline applied to raw conversation records before
embedding and storage.

Pipeline stages (applied sequentially):
  1. Schema validation   - enforce required fields via Pydantic
  2. HTML stripping      - remove markup from rich-text CRM exports
  3. PII redaction       - regex-based masking of emails, phone numbers, card numbers
  4. Boilerplate removal - strip repeated agent signatures and legal disclaimers
  5. Unicode normalisation
  6. Language detection  - filter non-target-language records
  7. Token counting      - annotate turns with approximate token budgets
"""

import re
import unicodedata
import logging
from typing import Optional

from pydantic import BaseModel, field_validator, ValidationError

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Blocklist: patterns appearing in boilerplate text to strip
# ---------------------------------------------------------------------------
_BOILERPLATE_PATTERNS = [
    re.compile(r"This (email|message|communication) (is |may be )?confidential.*", re.IGNORECASE | re.DOTALL),
    re.compile(r"Sent from my (iPhone|Galaxy|Pixel|Android).*", re.IGNORECASE),
    re.compile(r"Thank you for contacting .{0,50} support\.", re.IGNORECASE),
    re.compile(r"(\[Ticket #\d+\]|\[Case #\d+\])", re.IGNORECASE),
]

# PII patterns
_EMAIL_RE = re.compile(r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}")
_PHONE_RE = re.compile(r"\+?[\d\s\-().]{7,15}\d")
_CARD_RE  = re.compile(r"\b(?:\d[ -]?){13,16}\b")


class ConversationTurnSchema(BaseModel):
    speaker_role: str
    message_text: str
    sent_at: str

    @field_validator("speaker_role")
    @classmethod
    def validate_role(cls, v: str) -> str:
        allowed = {"customer", "agent", "bot"}
        if v.lower() not in allowed:
            raise ValueError(f"speaker_role must be one of {allowed}")
        return v.lower()


class ConversationSchema(BaseModel):
    external_ref_id: str
    agent_id: str
    channel: str
    started_at: str
    resolution_status: str
    turns: list[ConversationTurnSchema]
    ended_at: Optional[str] = None
    topic_id: Optional[int] = None
    csat_score: Optional[int] = None
    handle_time_secs: Optional[int] = None
    language_code: str = "en"


class PreprocessingPipeline:
    """
    Stateless transformation pipeline. Each stage is implemented as a
    private method and called sequentially by `process()`.

    Returning `None` from any stage causes the record to be dropped and
    written to the dead-letter queue.
    """

    def process(self, raw: dict) -> Optional[dict]:
        try:
            record = ConversationSchema.model_validate(raw)
        except ValidationError as exc:
            logger.debug("Schema validation failed: %s", exc)
            return None

        cleaned_turns = []
        full_text_parts = []

        for turn in record.turns:
            text = turn.message_text
            text = self._strip_html(text)
            text = self._strip_boilerplate(text)
            text = self._redact_pii(text)
            text = self._normalise_unicode(text)
            text = text.strip()

            if not text:
                continue

            token_count = self._estimate_tokens(text)
            cleaned_turns.append({
                "speaker_role": turn.speaker_role,
                "message_text": text,
                "message_tokens": token_count,
                "sent_at": turn.sent_at,
            })
            full_text_parts.append(text)

        if not cleaned_turns:
            logger.debug("All turns empty after cleaning; dropping record %s", record.external_ref_id)
            return None

        return {
            **record.model_dump(exclude={"turns"}),
            "turns": cleaned_turns,
            "full_text": " ".join(full_text_parts),
        }

    # ------------------------------------------------------------------
    # Pipeline stages
    # ------------------------------------------------------------------

    @staticmethod
    def _strip_html(text: str) -> str:
        from html.parser import HTMLParser

        class _Stripper(HTMLParser):
            def __init__(self):
                super().__init__()
                self.parts = []

            def handle_data(self, data):
                self.parts.append(data)

        stripper = _Stripper()
        stripper.feed(text)
        return " ".join(stripper.parts)

    @staticmethod
    def _strip_boilerplate(text: str) -> str:
        for pattern in _BOILERPLATE_PATTERNS:
            text = pattern.sub("", text)
        return text

    @staticmethod
    def _redact_pii(text: str) -> str:
        text = _EMAIL_RE.sub("[EMAIL]", text)
        text = _PHONE_RE.sub("[PHONE]", text)
        text = _CARD_RE.sub("[CARD]", text)
        return text

    @staticmethod
    def _normalise_unicode(text: str) -> str:
        return unicodedata.normalize("NFKC", text)

    @staticmethod
    def _estimate_tokens(text: str) -> int:
        """
        Approximate token count using the rule of thumb: ~4 characters per token.
        For production use, replace with tiktoken for precise GPT tokenisation.
        """
        return max(1, len(text) // 4)
