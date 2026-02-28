"""
data_ingestion.py
-----------------
Batch ingestion pipeline for raw customer conversation logs.

Reads JSONL or CSV log files from a source directory, applies the NLP
pre-processing pipeline, persists structured metadata to PostgreSQL, and
dispatches text content to the VectorEmbeddingManager for indexing.

Usage:
    python data_ingestion.py --source data/raw_logs/ --batch-size 512
"""

import argparse
import hashlib
import json
import logging
import sys
import uuid
from pathlib import Path
from typing import Generator, Iterator

import psycopg2
from psycopg2.extras import execute_values

from config import settings
from preprocessing_pipeline import PreprocessingPipeline
from vector_embedding_manager import VectorEmbeddingManager

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger(__name__)


class ConversationIngestionPipeline:
    """
    Orchestrates the end-to-end ingestion of raw conversation logs into the
    Sentinel hybrid storage layer (PostgreSQL + vector index).
    """

    def __init__(self, batch_size: int = 512) -> None:
        self.batch_size = batch_size
        self.preprocessor = PreprocessingPipeline()
        self.embedder = VectorEmbeddingManager()
        self._conn = self._connect()
        self._batch_id: str | None = None

    # ------------------------------------------------------------------
    # Database connectivity
    # ------------------------------------------------------------------

    def _connect(self) -> psycopg2.extensions.connection:
        conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
        )
        conn.autocommit = False
        return conn

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def run(self, source_dir: Path) -> dict:
        """
        Execute the full ingestion pipeline for all log files in source_dir.

        Returns a summary dict with counts of processed and failed records.
        """
        self._batch_id = self._create_batch_record(str(source_dir))
        processed, failed = 0, 0

        log_files = list(source_dir.glob("*.jsonl")) + list(source_dir.glob("*.csv"))
        logger.info("Discovered %d log file(s) in %s", len(log_files), source_dir)

        for file_path in log_files:
            logger.info("Processing file: %s", file_path.name)
            for batch in self._read_batches(file_path):
                p, f = self._process_batch(batch)
                processed += p
                failed += f

        self._finalize_batch_record(processed, failed, status="completed")
        logger.info("Ingestion complete. Processed: %d | Failed: %d", processed, failed)
        return {"processed": processed, "failed": failed}

    # ------------------------------------------------------------------
    # File reading
    # ------------------------------------------------------------------

    def _read_batches(self, file_path: Path) -> Generator[list[dict], None, None]:
        """Yield successive batches of raw records from a log file."""
        suffix = file_path.suffix.lower()
        if suffix == ".jsonl":
            records = self._parse_jsonl(file_path)
        elif suffix == ".csv":
            records = self._parse_csv(file_path)
        else:
            logger.warning("Unsupported file type: %s. Skipping.", file_path.suffix)
            return

        batch: list[dict] = []
        for record in records:
            batch.append(record)
            if len(batch) >= self.batch_size:
                yield batch
                batch = []
        if batch:
            yield batch

    @staticmethod
    def _parse_jsonl(path: Path) -> Iterator[dict]:
        with path.open("r", encoding="utf-8") as fh:
            for line_no, line in enumerate(fh, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    yield json.loads(line)
                except json.JSONDecodeError as exc:
                    logger.warning("Malformed JSON at line %d: %s", line_no, exc)

    @staticmethod
    def _parse_csv(path: Path) -> Iterator[dict]:
        import csv
        with path.open("r", encoding="utf-8", newline="") as fh:
            reader = csv.DictReader(fh)
            for row in reader:
                yield dict(row)

    # ------------------------------------------------------------------
    # Batch processing
    # ------------------------------------------------------------------

    def _process_batch(self, raw_records: list[dict]) -> tuple[int, int]:
        """
        Pre-process, embed, and persist a batch of records.

        Wraps the entire batch in a single database transaction to ensure
        atomicity: either all records in a batch are committed, or none are.
        """
        processed, failed = 0, 0
        conversation_rows, turn_rows = [], []
        text_chunks, chunk_conv_map = [], {}

        for raw in raw_records:
            try:
                cleaned = self.preprocessor.process(raw)
                if cleaned is None:
                    failed += 1
                    continue

                checksum = hashlib.sha256(cleaned["full_text"].encode()).hexdigest()
                conv_id = str(uuid.uuid4())

                conversation_rows.append((
                    conv_id,
                    cleaned["external_ref_id"],
                    cleaned["agent_id"],
                    cleaned.get("topic_id"),
                    cleaned["channel"],
                    cleaned.get("language_code", "en"),
                    cleaned["started_at"],
                    cleaned.get("ended_at"),
                    cleaned["resolution_status"],
                    cleaned.get("csat_score"),
                    cleaned.get("handle_time_secs"),
                    checksum,
                ))

                for idx, turn in enumerate(cleaned["turns"]):
                    turn_rows.append((
                        conv_id,
                        idx,
                        turn["speaker_role"],
                        turn["message_text"],
                        turn.get("message_tokens"),
                        turn["sent_at"],
                    ))
                    chunk_key = len(text_chunks)
                    text_chunks.append(turn["message_text"])
                    chunk_conv_map[chunk_key] = (conv_id, idx)

                processed += 1

            except Exception as exc:  # noqa: BLE001
                logger.error("Record processing error: %s", exc, exc_info=False)
                failed += 1

        if not conversation_rows:
            return processed, failed

        try:
            embedding_ids = self.embedder.embed_and_index(text_chunks)
            with self._conn:
                with self._conn.cursor() as cur:
                    execute_values(
                        cur,
                        """
                        INSERT INTO conversations (
                            conversation_id, external_ref_id, agent_id, topic_id,
                            channel, language_code, started_at, ended_at,
                            resolution_status, csat_score, handle_time_secs, raw_log_checksum
                        ) VALUES %s
                        ON CONFLICT (external_ref_id) DO NOTHING
                        """,
                        conversation_rows,
                    )
                    for i, turn_row in enumerate(turn_rows):
                        embedding_chunk_id = embedding_ids.get(i)
                        cur.execute(
                            """
                            INSERT INTO conversation_turns (
                                conversation_id, turn_index, speaker_role,
                                message_text, message_tokens, sent_at, embedding_chunk_id
                            ) VALUES (%s, %s, %s, %s, %s, %s, %s)
                            ON CONFLICT (conversation_id, turn_index) DO NOTHING
                            """,
                            (*turn_row, embedding_chunk_id),
                        )
        except Exception as exc:
            logger.error("Batch commit failed: %s", exc, exc_info=True)
            self._conn.rollback()
            failed += processed
            processed = 0

        return processed, failed

    # ------------------------------------------------------------------
    # Batch tracking
    # ------------------------------------------------------------------

    def _create_batch_record(self, source_path: str) -> str:
        batch_id = str(uuid.uuid4())
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    "INSERT INTO ingestion_batches (batch_id, source_path) VALUES (%s, %s)",
                    (batch_id, source_path),
                )
        return batch_id

    def _finalize_batch_record(self, processed: int, failed: int, status: str) -> None:
        with self._conn:
            with self._conn.cursor() as cur:
                cur.execute(
                    """
                    UPDATE ingestion_batches
                    SET completed_at = NOW(), records_processed = %s,
                        records_failed = %s, status = %s
                    WHERE batch_id = %s
                    """,
                    (processed, failed, status, self._batch_id),
                )


def main() -> None:
    parser = argparse.ArgumentParser(description="Sentinel conversation log ingestion pipeline")
    parser.add_argument("--source", type=Path, required=True, help="Directory containing raw log files")
    parser.add_argument("--batch-size", type=int, default=512, help="Records per processing batch")
    args = parser.parse_args()

    if not args.source.is_dir():
        logger.error("Source path does not exist or is not a directory: %s", args.source)
        sys.exit(1)

    pipeline = ConversationIngestionPipeline(batch_size=args.batch_size)
    summary = pipeline.run(args.source)
    sys.exit(0 if summary["failed"] == 0 else 1)


if __name__ == "__main__":
    main()
