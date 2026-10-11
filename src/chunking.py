from __future__ import annotations

import hashlib
import re

from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


def is_table_block(block: str) -> bool:
    lines = block.strip().splitlines()

    return (
        len(lines) >= 2
        and lines[0].lstrip().startswith("|")
        and any(
            re.match(r"^\s*\|?\s*:?-{3,}", line)
            for line in lines[1:2]
        )
    )


def split_table(table: str, max_chars: int) -> list[str]:
    lines = table.strip().splitlines()

    if len(lines) < 3:
        return [table]

    header = lines[0]
    separator = lines[1]
    rows = lines[2:]

    chunks = []
    current_rows = [header, separator]

    for row in rows:
        candidate = "\n".join(current_rows + [row])
        if len(candidate) > max_chars and len(current_rows) > 2:
            chunks.append("\n".join(current_rows))
            current_rows = [header, separator, row]
        else:
            current_rows.append(row)
        if len(current_rows) == 3 and len(
            "\n".join(current_rows)
        ) > max_chars:
            chunks.append("\n".join(current_rows))
            current_rows = [header, separator]

    if len(current_rows) > 2:
        chunks.append("\n".join(current_rows))

    return chunks or [table]


def split_markdown_blocks(text: str) -> list[str]:
    return [
        block.strip()
        for block in re.split(r"\n\s*\n", text)
        if block.strip()
    ]


def stable_chunk_id(source: str, file_hash: str, index: int, content: str) -> str:
    digest = hashlib.sha256(f"{source}|{file_hash}|{index}|{content}".encode("utf-8")).hexdigest()
    return digest


def chunk_documents(documents: list[Document], chunk_size: int = 1200, chunk_overlap: int = 150) -> list[Document]:
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    if chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be >= 0 and < chunk_size")

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n## ","\n### ","\n\n","\n",". "," ","",],
        length_function=len,
        keep_separator=True,
        add_start_index=True,
    )

    result: list[Document] = []

    for document in documents:
        text = document.page_content
        blocks = split_markdown_blocks(text)

        expanded_blocks = []

        for block in blocks:
            if is_table_block(block) and len(block) > chunk_size:
                expanded_blocks.extend(
                    split_table(block, chunk_size)
                )
            elif len(block) > chunk_size:
                expanded_blocks.append(block)
            else:
                expanded_blocks.append(block)

        grouped = []
        current = ""

        for block in expanded_blocks:
            candidate = (f"{current}\n\n{block}" if current else block)

            if len(candidate) <= chunk_size:
                current = candidate
                continue

            if current:
                grouped.append(current)

            current = block

        if current:
            grouped.append(current)

        final_texts = []

        for group in grouped:
            if len(group) <= chunk_size:
                final_texts.append(group)
            else:
                final_texts.extend(
                    text_splitter.split_text(group)
                )

        for index, chunk in enumerate(final_texts):
            metadata = dict(document.metadata)

            metadata.update({
                "chunk_index": index,
                "chunk_id": stable_chunk_id(
                    metadata.get("source", ""),
                    metadata.get("file_hash", ""),
                    index,
                    chunk,
                ),
                "chunk_char_count": len(chunk),
            })

            result.append(Document(page_content=chunk,metadata=metadata))

    return result