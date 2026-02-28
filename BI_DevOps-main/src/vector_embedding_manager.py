"""
vector_embedding_manager.py
---------------------------
Manages the generation, indexing, and retrieval of dense vector embeddings
for conversation turn text.

Supports two backends:
  - OpenAI text-embedding-ada-002 (default; requires OPENAI_API_KEY)
  - HuggingFace sentence-transformers (offline/air-gapped deployments)

The FAISS index is persisted to disk and reloaded on startup to avoid
re-embedding on application restart.
"""

import logging
import pickle
from pathlib import Path
from typing import Optional

import faiss
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


class VectorEmbeddingManager:
    """
    Wraps FAISS index operations and embedding model calls.

    Design rationale: Centralising vector operations in a single class
    allows the FAISS index and embedding model to be swapped without
    affecting the upstream ingestion pipeline.
    """

    EMBEDDING_DIM = 1536  # OpenAI ada-002; adjust to 768 for most HuggingFace models

    def __init__(self) -> None:
        self._index: faiss.Index = self._load_or_create_index()
        self._id_map: dict[int, str] = self._load_id_map()
        self._embed_fn = self._build_embed_function()

    # ------------------------------------------------------------------
    # Index persistence
    # ------------------------------------------------------------------

    def _index_path(self) -> Path:
        return Path(settings.VECTOR_DB_PATH) / "sentinel.faiss"

    def _id_map_path(self) -> Path:
        return Path(settings.VECTOR_DB_PATH) / "id_map.pkl"

    def _load_or_create_index(self) -> faiss.Index:
        path = self._index_path()
        if path.exists():
            logger.info("Loading existing FAISS index from %s", path)
            return faiss.read_index(str(path))
        logger.info("Creating new FAISS IVF index (dim=%d)", self.EMBEDDING_DIM)
        quantizer = faiss.IndexFlatIP(self.EMBEDDING_DIM)
        index = faiss.IndexIVFFlat(quantizer, self.EMBEDDING_DIM, 100, faiss.METRIC_INNER_PRODUCT)
        return index

    def _load_id_map(self) -> dict[int, str]:
        path = self._id_map_path()
        if path.exists():
            with path.open("rb") as fh:
                return pickle.load(fh)
        return {}

    def _persist(self) -> None:
        index_path = self._index_path()
        index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self._index, str(index_path))
        with self._id_map_path().open("wb") as fh:
            pickle.dump(self._id_map, fh)

    # ------------------------------------------------------------------
    # Embedding backend selection
    # ------------------------------------------------------------------

    def _build_embed_function(self):
        if settings.EMBEDDING_BACKEND == "openai":
            return self._embed_openai
        return self._embed_huggingface

    def _embed_openai(self, texts: list[str]) -> np.ndarray:
        from openai import OpenAI
        client = OpenAI(api_key=settings.OPENAI_API_KEY)
        response = client.embeddings.create(
            model="text-embedding-ada-002",
            input=texts,
        )
        vectors = [item.embedding for item in response.data]
        arr = np.array(vectors, dtype=np.float32)
        faiss.normalize_L2(arr)
        return arr

    def _embed_huggingface(self, texts: list[str]) -> np.ndarray:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer(settings.HF_MODEL_NAME)
        arr = model.encode(texts, normalize_embeddings=True, convert_to_numpy=True)
        return arr.astype(np.float32)

    # ------------------------------------------------------------------
    # Public interface
    # ------------------------------------------------------------------

    def embed_and_index(self, texts: list[str]) -> dict[int, str]:
        """
        Embed a list of text strings and add them to the FAISS index.

        Returns a mapping from list index -> FAISS internal ID (as string),
        which is stored alongside the PostgreSQL record for cross-referencing.
        """
        if not texts:
            return {}

        vectors = self._embed_fn(texts)
        start_id = self._index.ntotal

        if not self._index.is_trained:
            logger.info("Training FAISS IVF index on %d vectors", len(vectors))
            self._index.train(vectors)

        self._index.add(vectors)
        id_map: dict[int, str] = {}
        for i in range(len(texts)):
            internal_id = start_id + i
            chunk_id = f"chunk_{internal_id}"
            self._id_map[internal_id] = chunk_id
            id_map[i] = chunk_id

        self._persist()
        logger.debug("Indexed %d vectors. Total index size: %d", len(texts), self._index.ntotal)
        return id_map

    def search(self, query: str, top_k: int = 20) -> list[tuple[str, float]]:
        """
        Perform ANN search for a query string.

        Returns a list of (chunk_id, similarity_score) tuples ranked by
        descending cosine similarity.
        """
        query_vec = self._embed_fn([query])
        distances, indices = self._index.search(query_vec, top_k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1:
                continue
            chunk_id = self._id_map.get(int(idx))
            if chunk_id:
                results.append((chunk_id, float(dist)))
        return results
