# Local document RAG POC

This project ingests PDF, TXT/Markdown, JSON, CSV, Excel, and DOCX files under `data/`, splits them into source-aware chunks, embeds them with a normalized Sentence Transformers model, and retrieves them with a persistent FAISS cosine-similarity index. Google ADK uses the retrieval function as its only document evidence source.

## Setup

Use Python 3.10 or newer, install `requirements.txt`, and set `GOOGLE_API_KEY` in the environment for Gemini. The first run downloads the embedding model if it is not already cached.

## Run

From the repository root:

```powershell
python -m src.document_processing
python main.py
```

For the ADK web development UI, run `adk web` from the repository root and select the `src` agent directory. The index is stored in the project root at `faiss_store/faiss.index`, with searchable chunk text and source metadata in `faiss_store/metadata.json`. Remove that directory when you want a clean rebuild after changing the embedding model or chunking settings.

PDFs are extracted page by page with pypdf, with local RapidOCR fallback for scanned pages. Install the updated dependencies after adding OCR support with `python -m pip install -r requirements.txt`; OCR models are fetched on first use. Retrieval combines normalized vector similarity with BM25-style lexical matching and filters weak matches using `RAG_MIN_RELEVANCE_SCORE` (default `0.22`). This helps exact terms such as `semaphore` surface even when the surrounding chunk is short. Scores are ranking signals, not calibrated probabilities. The ADK prompt requires citations from returned filenames and available page/row metadata, and instructs the agent to abstain when retrieved passages do not support an answer.

Place relative paths under `data/.ragignore` only when you want to exclude files from the index. By default, all supported files in nested folders are ingested. The app compares indexed file hashes to current source files and rebuilds a stale index automatically; you can also force a rebuild with `python -m src.document_processing`.

## Tracking ingestion and retrieval

`python -m src.document_processing` emits JSON Lines events. Each `ingestion_file_chunks` event reports the source file, chunk count, and chunk IDs. `index_saved` and `ingestion_complete` report the persisted paths and totals. During a query, `retrieval_results` lists each ranked match with its cosine score, source, page/row/sheet where available, chunk ID, citation, and retrieved text. This makes it possible to redirect stdout to a log file for later review.

The interactive answer is printed as JSON with `status`, `answer`, and `citations`. Each citation includes the source, location, exact chunk ID, and a short supporting quote. For example:

```json
{
  "status": "answered",
  "answer": "A semaphore limits access to a shared resource.",
  "citations": [
    {
      "source": "text_files/python_intro.txt",
      "location": "chunk",
      "chunk_id": "...",
      "quote": "..."
    }
  ]
}
```
