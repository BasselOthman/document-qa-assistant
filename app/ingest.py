"""Document Ingestion Script.

Loads all PDF files in data/, splits them into chunks with configurable
size and overlap, attaches metadata (filename, 1-based page number),
embeds them with FastEmbed (bge-small-en-v1.5), and stores them idempotently
in a local persistent Chroma collection and BM25 index.

Usage:
    python -m app.ingest
"""

import glob
import hashlib
import os
import pickle
import sys
from typing import List, Tuple
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import get_settings
from app.vectorstore import get_vectorstore


def generate_chunk_id(doc: Document, chunk_index: int) -> str:
    """Generate a deterministic SHA-256 ID for a document chunk.

    Guarantees idempotency so repeated ingestion runs never insert duplicate vectors.
    """
    source = doc.metadata.get("source", "unknown")
    page = doc.metadata.get("page", 1)
    content_hash = hashlib.sha256(doc.page_content.encode("utf-8")).hexdigest()[:16]
    raw_key = f"{source}::p{page}::c{chunk_index}::{content_hash}"
    return hashlib.sha256(raw_key.encode("utf-8")).hexdigest()


def load_pdf_pages(file_path: str) -> List[Document]:
    """Load a single PDF file page-by-page, attaching clean metadata."""
    documents: List[Document] = []
    filename = os.path.basename(file_path)

    try:
        from langchain_community.document_loaders import PyPDFLoader

        loader = PyPDFLoader(file_path)
        pages = loader.load()
        for idx, page_doc in enumerate(pages):
            page_num = page_doc.metadata.get("page", idx)
            # Normalize to 1-based index
            if isinstance(page_num, int):
                page_num = page_num + 1 if page_num == idx else page_num
            else:
                page_num = idx + 1

            clean_text = page_doc.page_content.strip()
            if clean_text:
                documents.append(
                    Document(
                        page_content=clean_text,
                        metadata={"source": filename, "page": page_num},
                    )
                )
    except Exception as e:
        # Fallback to direct pypdf reading if community loader faces issues
        try:
            import pypdf

            reader = pypdf.PdfReader(file_path)
            for idx, page in enumerate(reader.pages):
                text = page.extract_text() or ""
                clean_text = text.strip()
                if clean_text:
                    documents.append(
                        Document(
                            page_content=clean_text,
                            metadata={"source": filename, "page": idx + 1},
                        )
                    )
        except Exception as fallback_err:
            print(f"[Warning] Failed to load {file_path}: {e} / {fallback_err}", file=sys.stderr)

    return documents


def split_documents(
    docs: List[Document],
    chunk_size: int = 800,
    chunk_overlap: int = 150,
) -> List[Tuple[str, Document]]:
    """Split page documents into overlapping chunks with deterministic IDs."""
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunk_records: List[Tuple[str, Document]] = []
    chunk_counter = 0

    for doc in docs:
        sub_chunks = text_splitter.split_text(doc.page_content)
        for i, chunk_text in enumerate(sub_chunks):
            chunk_doc = Document(
                page_content=chunk_text,
                metadata={
                    "source": doc.metadata.get("source", "unknown"),
                    "page": doc.metadata.get("page", 1),
                },
            )
            chunk_id = generate_chunk_id(chunk_doc, chunk_counter)
            chunk_records.append((chunk_id, chunk_doc))
            chunk_counter += 1

    return chunk_records


def run_ingestion() -> Tuple[int, int]:
    """Run full ingestion pipeline over data/ PDFs and build indices.

    Returns:
        Tuple of (number of new chunks added, total chunks now indexed).
    """
    settings = get_settings()
    pdf_dir = os.path.abspath(settings.pdf_data_dir)

    if not os.path.exists(pdf_dir):
        os.makedirs(pdf_dir, exist_ok=True)
        print(f"[Ingest] Created missing directory: {pdf_dir}")

    pdf_files = sorted(glob.glob(os.path.join(pdf_dir, "*.pdf")))
    if not pdf_files:
        print(f"[Ingest] No PDF files found in {pdf_dir}. Add PDFs to begin.")
        return 0, 0

    print(f"[Ingest] Found {len(pdf_files)} PDF file(s) in {pdf_dir}:")
    for p in pdf_files:
        print(f"  - {os.path.basename(p)}")

    all_page_docs: List[Document] = []
    for pdf_path in pdf_files:
        page_docs = load_pdf_pages(pdf_path)
        print(f"[Ingest] Loaded {len(page_docs)} page(s) from {os.path.basename(pdf_path)}")
        all_page_docs.extend(page_docs)

    if not all_page_docs:
        print("[Ingest] No text could be extracted from PDFs.")
        return 0, 0

    # Split into chunks
    chunk_records = split_documents(
        all_page_docs,
        chunk_size=settings.chunk_size,
        chunk_overlap=settings.chunk_overlap,
    )
    print(f"[Ingest] Generated {len(chunk_records)} chunks (size={settings.chunk_size}, overlap={settings.chunk_overlap})")

    # Connect to persistent Chroma vector store
    vectorstore = get_vectorstore()

    # Query existing IDs to avoid re-inserting duplicates
    try:
        existing_data = vectorstore.get()
        existing_ids = set(existing_data.get("ids", []))
    except Exception:
        existing_ids = set()

    new_ids = []
    new_docs = []
    for chunk_id, doc in chunk_records:
        if chunk_id not in existing_ids:
            new_ids.append(chunk_id)
            new_docs.append(doc)

    if new_docs:
        print(f"[Ingest] Adding {len(new_docs)} new chunks into Chroma collection '{settings.chroma_collection}'...")
        vectorstore.add_documents(documents=new_docs, ids=new_ids)
        print("[Ingest] Chroma embedding and insertion complete.")
    else:
        print(f"[Ingest] All {len(chunk_records)} chunks are already present in Chroma. No duplicates added.")

    # Rebuild and persist BM25 index from all current chunks
    all_chunks = [doc for _, doc in chunk_records]
    os.makedirs(os.path.dirname(os.path.abspath(settings.bm25_index_path)), exist_ok=True)
    with open(settings.bm25_index_path, "wb") as f:
        pickle.dump(all_chunks, f)
    print(f"[Ingest] Saved BM25 keyword index ({len(all_chunks)} chunks) to {settings.bm25_index_path}")

    total_count = vectorstore._collection.count()
    print(f"[Ingest] Done! Total chunks in Chroma vector store: {total_count}")
    return len(new_docs), total_count


if __name__ == "__main__":
    new_count, total_count = run_ingestion()
    print(f"Ingestion finished successfully: +{new_count} chunks (Total: {total_count})")
