"""
db_client.py
------------
PostgreSQL connection and query abstraction layer.
"""

import logging
from contextlib import contextmanager
from typing import Any, Generator, Optional

import psycopg2
import psycopg2.extras

from config import settings

logger = logging.getLogger(__name__)


class DatabaseClient:
    def __init__(self) -> None:
        self._conn = psycopg2.connect(
            host=settings.POSTGRES_HOST,
            port=settings.POSTGRES_PORT,
            dbname=settings.POSTGRES_DB,
            user=settings.POSTGRES_USER,
            password=settings.POSTGRES_PASSWORD,
        )

    @contextmanager
    def cursor(self) -> Generator:
        with self._conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            yield cur

    def fetch_conversation_metadata(self, filters: dict) -> list[dict[str, Any]]:
        clauses, params = [], []
        if "date_from" in filters:
            clauses.append("c.started_at >= %s")
            params.append(filters["date_from"])
        if "date_to" in filters:
            clauses.append("c.started_at <= %s")
            params.append(filters["date_to"])
        if "channel" in filters:
            clauses.append("c.channel = %s")
            params.append(filters["channel"])
        if "topic_id" in filters:
            clauses.append("c.topic_id = %s")
            params.append(filters["topic_id"])

        where = ("WHERE " + " AND ".join(clauses)) if clauses else ""
        sql = f"""
            SELECT
                c.conversation_id,
                c.external_ref_id,
                c.resolution_status,
                c.csat_score,
                c.channel,
                a.full_name AS agent_name,
                t.topic_name
            FROM conversations c
            LEFT JOIN agents a ON c.agent_id = a.agent_id
            LEFT JOIN topics t ON c.topic_id = t.topic_id
            {where}
            ORDER BY c.started_at DESC
            LIMIT 100
        """
        with self.cursor() as cur:
            cur.execute(sql, params)
            return [dict(row) for row in cur.fetchall()]
