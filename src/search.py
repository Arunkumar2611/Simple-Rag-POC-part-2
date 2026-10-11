"""Retrieval helpers used by the ADK tool and command-line entry point."""
from __future__ import annotations

from pathlib import Path

from .document_loader import SUPPORTED_EXTENSIONS, _ignored_paths, load_all_documents, sha256_file
from .vectorstore import FaissVectorStore


class RAGSearch:
    def __init__(self, persist_dir: str | None = None, data_dir: str | None = None):
        self.vectorstore = FaissVectorStore(persist_dir)
        project_root = Path(__file__).resolve().parent.parent
        root = Path(data_dir) if data_dir else project_root / "data"
        root = root.resolve()
        expected = self._source_fingerprints(root)
        if (self.vectorstore.index_path.exists() and self.vectorstore.metadata_path.exists()
                and self._index_matches_sources(expected)):
            self.vectorstore.load()
        else:
            documents = load_all_documents(str(root))
            self.vectorstore.build_from_documents(documents)

    @staticmethod
    def _source_fingerprints(root: Path) -> dict[str, str]:
        ignored = _ignored_paths(root)
        return {
            path.relative_to(root).as_posix(): sha256_file(path)
            for path in sorted(root.rglob("*"))
            if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
            and path.relative_to(root).as_posix().casefold() not in ignored
        }

    def _index_matches_sources(self, expected: dict[str, str]) -> bool:
        try:
            import json
            indexed = json.loads(self.vectorstore.metadata_path.read_text(encoding="utf-8"))
            actual = {
                str(item.get("relative_source", "")).replace("\\", "/"): item.get("file_hash")
                for item in indexed
            }
            return actual == expected
        except (OSError, ValueError, TypeError):
            return False

    def retrieve(self, query: str, top_k: int = 5) -> list[dict]:
        return self.vectorstore.query(query, top_k)

    def search_and_summarize(self, query: str, top_k: int = 5) -> str:
        results = self.retrieve(query, top_k)
        if not results:
            return "No relevant document chunks were found."
        return "\n\n".join(
            f"[{item['metadata'].get('file_name', 'document')} | "
            f"{item['metadata'].get('chunk_id', '')} | score={item['score']:.3f}]\n"
            f"{item['metadata']['text']}" for item in results
        )
