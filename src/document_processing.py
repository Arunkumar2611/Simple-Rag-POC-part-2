import logging
from pathlib import Path

from src.document_loader import load_all_documents
from src.vectorstore import FaissVectorStore

logging.basicConfig(level=logging.WARNING)

def main() -> None:
    data_dir = Path(__file__).resolve().parent.parent / "data"
    documents = load_all_documents(str(data_dir))
    store = FaissVectorStore()
    chunk_count = store.build_from_documents(documents)
    store._emit({"event": "ingestion_complete", "source_record_count": len(documents), "chunk_count": chunk_count, "index_path": str(store.index_path)})


if __name__ == "__main__":
    logging.basicConfig(level=logging.WARNING)
    main()
