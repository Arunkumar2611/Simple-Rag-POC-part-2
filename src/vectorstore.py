"""Persistent FAISS cosine-similarity store for text chunks."""
from __future__ import annotations

import json
import math
import os
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence

import faiss
import numpy as np
from langchain_core.documents import Document

from .embedding import EmbeddingPipeline


class FaissVectorStore:
    def __init__(self, persist_dir: str | None = None, embedding_model: str = "sentence-transformers/all-MiniLM-L6-v2", chunk_size: int = 900, chunk_overlap: int = 120):
        project_root = Path(__file__).resolve().parent.parent
        self.persist_dir = Path(persist_dir) if persist_dir else project_root / "faiss_store"
        if not self.persist_dir.is_absolute():
            self.persist_dir = project_root / self.persist_dir
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.index: faiss.Index | None = None
        self.metadata: list[dict] = []
        self.pipeline = EmbeddingPipeline(embedding_model, chunk_size, chunk_overlap)
        self.min_relevance_score = float(os.getenv("RAG_MIN_RELEVANCE_SCORE", "0.22"))

    @property
    def index_path(self) -> Path:
        return self.persist_dir / "faiss.index"

    @property
    def metadata_path(self) -> Path:
        return self.persist_dir / "metadata.json"

    def build_from_documents(self, documents: Sequence[Document]) -> int:
        chunks = self.pipeline.chunk_documents(documents)
        if not chunks:
            raise ValueError("No searchable chunks were produced from the documents")
        vectors = self.pipeline.embed_chunks(chunks)
        self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.metadata = [{"text": d.page_content, **d.metadata} for d in chunks]
        self.save()
        counts = Counter(item.get("relative_source", item.get("file_name", "unknown")) for item in self.metadata)
        for source, count in sorted(counts.items()):
            source_chunks = [item for item in self.metadata
                             if item.get("relative_source", item.get("file_name", "unknown")) == source]
            self._emit({
                "event": "ingestion_file_chunks",
                "source": source,
                "chunk_count": count,
                "chunk_ids": [item["chunk_id"] for item in source_chunks],
            })
        self._emit({"event": "index_saved", "chunk_count": len(chunks), "index_path": str(self.index_path),
                    "metadata_path": str(self.metadata_path)})
        return len(chunks)

    @staticmethod
    def _emit(event: dict) -> None:
        print(json.dumps({"timestamp": datetime.now(timezone.utc).isoformat(), **event}, ensure_ascii=False))

    def save(self) -> None:
        if self.index is None:
            raise RuntimeError("Cannot save an empty vector store")
        index_tmp = self.index_path.with_suffix(".index.tmp")
        meta_tmp = self.metadata_path.with_suffix(".json.tmp")
        faiss.write_index(self.index, str(index_tmp))
        meta_tmp.write_text(json.dumps(self.metadata, ensure_ascii=False), encoding="utf-8")
        os.replace(index_tmp, self.index_path)
        os.replace(meta_tmp, self.metadata_path)

    def load(self) -> None:
        if not (self.index_path.is_file() and self.metadata_path.is_file()):
            raise FileNotFoundError(f"Vector store is incomplete in {self.persist_dir}")
        index = faiss.read_index(str(self.index_path))
        metadata = json.loads(self.metadata_path.read_text(encoding="utf-8"))
        if index.ntotal != len(metadata):
            raise ValueError("FAISS index and metadata are out of sync; rebuild the store")
        self.index, self.metadata = index, metadata

    def query(self, query_text: str, top_k: int = 5) -> list[dict]:
        if not query_text.strip():
            return []
        if self.index is None:
            raise RuntimeError("Vector store is not loaded; call load() or build_from_documents()")
        if top_k < 1:
            raise ValueError("top_k must be positive")
        vector = self.pipeline.embed_texts([query_text])
        candidate_k = min(max(top_k * 10, 50), self.index.ntotal)
        scores, indices = self.index.search(vector, candidate_k)
        candidates = [
            {"vector_score": float(score), "metadata": self.metadata[int(i)]}
            for score, i in zip(scores[0], indices[0]) if i >= 0
        ]
        lexical_scores = self._bm25_scores(query_text, candidates)
        results = []
        for item, lexical_score in zip(candidates, lexical_scores):
            vector_score = max(0.0, item["vector_score"])
            combined_score = 0.65 * vector_score + 0.35 * lexical_score
            if combined_score >= self.min_relevance_score:
                results.append({
                    "score": combined_score,
                    "vector_score": item["vector_score"],
                    "lexical_score": lexical_score,
                    "metadata": item["metadata"],
                })
        results.sort(key=lambda item: item["score"], reverse=True)
        results = results[:top_k]
        self._log_retrieval(query_text, results, len(candidates) - len(results))
        return results

    @staticmethod
    def _tokens(text: str) -> list[str]:
        stop_words = {"what", "which", "when", "where", "who", "why", "how", "is", "are", "was", "were", "the", "a", "an", "of", "in", "on", "to", "for", "and", "or", "with", "do", "does", "did", "it", "this", "that", "can", "could", "please", "explain", "tell"}
        tokens = []
        for token in re.findall(r"[a-z0-9]+", text.casefold()):
            if token in stop_words or len(token) < 2:
                continue
            if token.endswith("ies") and len(token) > 5:
                token = token[:-3] + "y"
            elif token.endswith("s") and len(token) > 4:
                token = token[:-1]
            tokens.append(token)
        return tokens

    @classmethod
    def _bm25_scores(cls, query: str, candidates: list[dict]) -> list[float]:
        terms = set(cls._tokens(query))
        if not terms or not candidates:
            return [0.0] * len(candidates)
        tokenized = [cls._tokens(item["metadata"].get("text", "")) for item in candidates]
        average_length = sum(map(len, tokenized)) / max(1, len(tokenized))
        document_frequency = {term: sum(term in tokens for tokens in tokenized) for term in terms}
        raw_scores = []
        for tokens in tokenized:
            length = len(tokens)
            score = 0.0
            for term in terms:
                frequency = tokens.count(term)
                if not frequency:
                    continue
                idf = max(0.0, math.log(1 + (len(tokenized) - document_frequency[term] + 0.5) / (document_frequency[term] + 0.5)))
                score += idf * frequency * 2.2 / (frequency + 1.2 * (0.25 + 0.75 * length / max(1, average_length)))
            raw_scores.append(score)
        maximum = max(raw_scores, default=0.0)
        return [score / maximum if maximum else 0.0 for score in raw_scores]

    def _log_retrieval(self, query: str, results: list[dict], filtered_count: int) -> None:
        self._emit({
            "event": "retrieval_results",
            "query": query,
            "result_count": len(results),
            "filtered_candidate_count": filtered_count,
            "results": [
                {
                    "rank": rank,
                    "score": item["score"],
                    "vector_score": item["vector_score"],
                    "lexical_score": item["lexical_score"],
                    "source": item["metadata"].get("relative_source", item["metadata"].get("file_name")),
                    "page": item["metadata"].get("page_number"),
                    "row": item["metadata"].get("row_number"),
                    "sheet": item["metadata"].get("sheet_name"),
                    "chunk_id": item["metadata"].get("chunk_id"),
                    "citation": self._citation(item["metadata"]),
                    "retrieved_text": item["metadata"].get("text", ""),
                }
                for rank, item in enumerate(results, start=1)
            ],
        })

    @staticmethod
    def _citation(metadata: dict) -> str:
        source = metadata.get("relative_source", metadata.get("file_name", "unknown"))
        location = []
        if metadata.get("page_number") is not None:
            location.append(f"page {metadata['page_number']}")
        if metadata.get("sheet_name"):
            location.append(f"sheet {metadata['sheet_name']}")
        if metadata.get("row_number") is not None:
            location.append(f"row {metadata['row_number']}")
        location.append(f"chunk {metadata.get('chunk_id', 'unknown')}")
        return f"{source} ({', '.join(location)})"

    def search(self, query_embedding: np.ndarray, top_k: int = 5) -> list[dict]:
        if self.index is None:
            raise RuntimeError("Vector store is not loaded")
        q = np.asarray(query_embedding, dtype=np.float32)
        if q.ndim == 1:
            q = q[None, :]
        faiss.normalize_L2(q)
        scores, indices = self.index.search(q, min(top_k, self.index.ntotal))
        return [{"score": float(score), "metadata": self.metadata[int(i)]}
                for score, i in zip(scores[0], indices[0]) if i >= 0]


if __name__ == "__main__":
    from .document_loader import load_all_documents
    docs = load_all_documents("data")
    store = FaissVectorStore()
    count = store.build_from_documents(docs)
    print(f"Indexed {count} chunks")
