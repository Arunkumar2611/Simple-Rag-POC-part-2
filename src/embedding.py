"""Shared, normalized sentence embedding pipeline."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from sentence_transformers import SentenceTransformer
from langchain_core.documents import Document

from .chunking import chunk_documents
from .document_loader import load_all_documents


class EmbeddingPipeline:
    def __init__(self, model_name: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size: int = 900, chunk_overlap: int = 120):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)

    def chunk_documents(self, documents: Sequence[Document]) -> list[Document]:
        return chunk_documents(list(documents), self.chunk_size, self.chunk_overlap)

    def embed_texts(self, texts: Sequence[str], batch_size: int = 64) -> np.ndarray:
        if not texts:
            return np.empty((0, self.model.get_sentence_embedding_dimension()), dtype=np.float32)
        vectors = self.model.encode(list(texts), batch_size=batch_size, convert_to_numpy=True,
                                    normalize_embeddings=True, show_progress_bar=False)
        return np.asarray(vectors, dtype=np.float32)

    def embed_chunks(self, chunks: Sequence[Document], batch_size: int = 64) -> np.ndarray:
        return self.embed_texts([chunk.page_content for chunk in chunks], batch_size)


# Compatibility with the original class name.
EmbeddingPipelines = EmbeddingPipeline

if __name__ == "__main__":
    docs = load_all_documents("data")
    pipeline = EmbeddingPipeline()
    chunks = pipeline.chunk_documents(docs)
    vectors = pipeline.embed_chunks(chunks)
    print(f"Documents: {len(docs)} | chunks: {len(chunks)} | vectors: {vectors.shape}")
