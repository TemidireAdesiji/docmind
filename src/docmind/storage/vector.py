"""Qdrant-backed hybrid vector index.

Combines dense (sentence-transformer) and sparse (BM25) embeddings
for hybrid retrieval.
"""

from __future__ import annotations

from langchain_huggingface import HuggingFaceEmbeddings
from langchain_qdrant import (
    FastEmbedSparse,
    QdrantVectorStore,
    RetrievalMode,
)
from qdrant_client import QdrantClient
from qdrant_client.http import models as qmodels

from docmind.log import logger
from docmind.settings import Settings, get_settings


class VectorIndex:
    """Manages Qdrant collections with hybrid embeddings."""

    def __init__(self, cfg: Settings | None = None) -> None:
        self._cfg = cfg or get_settings()
        self._cfg.paths.ensure_dirs()

        self._client = QdrantClient(path=str(self._cfg.paths.vector_data))
        self._dense = HuggingFaceEmbeddings(
            model_name=self._cfg.dense_embedding
        )
        self._sparse = FastEmbedSparse(model_name=self._cfg.sparse_embedding)

    # -- collection lifecycle ----------------------------------------

    def ensure_collection(self, name: str | None = None) -> None:
        """Create the collection if it does not already exist."""
        label = name or self._cfg.collection_label
        if self._client.collection_exists(label):
            logger.debug("Collection '{}' already present", label)
            return

        vec_size = len(self._dense.embed_query("dimension probe"))
        self._client.create_collection(
            collection_name=label,
            vectors_config=qmodels.VectorParams(
                size=vec_size,
                distance=qmodels.Distance.COSINE,
            ),
            sparse_vectors_config={
                self._cfg.sparse_field: (qmodels.SparseVectorParams())
            },
        )
        logger.info("Created collection '{}'", label)

    def drop_collection(self, name: str | None = None) -> None:
        """Delete a collection if it exists."""
        label = name or self._cfg.collection_label
        if not self._client.collection_exists(label):
            return
        self._client.delete_collection(label)
        logger.info("Dropped collection '{}'", label)

    # -- store handle ------------------------------------------------

    def get_store(self, name: str | None = None) -> QdrantVectorStore:
        """Return a LangChain vector-store handle.

        Raises
        ------
        RuntimeError
            If the collection does not exist.
        """
        label = name or self._cfg.collection_label
        if not self._client.collection_exists(label):
            raise RuntimeError(
                f"Collection '{label}' does not exist. "
                "Call ensure_collection() first."
            )
        return QdrantVectorStore(
            client=self._client,
            collection_name=label,
            embedding=self._dense,
            sparse_embedding=self._sparse,
            retrieval_mode=RetrievalMode.HYBRID,
            sparse_vector_name=self._cfg.sparse_field,
        )
