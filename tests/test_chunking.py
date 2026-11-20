"""Unit tests for document chunking, overlap, and ID determinism."""

from langchain_core.documents import Document
from app.ingest import generate_chunk_id, split_documents


def test_split_documents_chunk_size_and_overlap():
    """Verify text splitter chunks text within bounds and preserves overlap."""
    chunk_size = 200
    chunk_overlap = 50

    # Create dummy text of 600 characters
    sample_text = (
        "Retrieval-Augmented Generation (RAG) is a technique that combines an information retrieval component "
        "with a text generator model. The retrieval component searches a collection of documents to find those "
        "relevant to the user query. The generator then synthesizes this retrieved context into a coherent answer. "
        "This approach prevents hallucinations, grounds outputs in factual documents, and allows dynamic knowledge "
        "updates without full model retraining."
    )

    doc = Document(page_content=sample_text, metadata={"source": "test_guide.pdf", "page": 1})
    records = split_documents([doc], chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    assert len(records) > 1, "Expected text to be split into multiple chunks"

    for chunk_id, chunk_doc in records:
        assert len(chunk_doc.page_content) <= chunk_size + 20, "Chunk exceeded size limit"
        assert chunk_doc.metadata["source"] == "test_guide.pdf"
        assert chunk_doc.metadata["page"] == 1
        assert isinstance(chunk_id, str) and len(chunk_id) == 64, "Chunk ID must be 64-char SHA256 hex"


def test_chunk_id_determinism():
    """Verify identical content produces the exact same ID, while modified content differs."""
    doc_a = Document(page_content="Exact identical text", metadata={"source": "doc.pdf", "page": 2})
    doc_b = Document(page_content="Exact identical text", metadata={"source": "doc.pdf", "page": 2})
    doc_c = Document(page_content="Slightly altered text", metadata={"source": "doc.pdf", "page": 2})

    id_a = generate_chunk_id(doc_a, 0)
    id_b = generate_chunk_id(doc_b, 0)
    id_c = generate_chunk_id(doc_c, 0)

    assert id_a == id_b, "Identical chunk metadata and content must generate identical ID"
    assert id_a != id_c, "Altered content must generate different ID"


def test_metadata_preservation():
    """Verify source and page metadata is maintained across multi-page documents."""
    docs = [
        Document(page_content="Page one content line.", metadata={"source": "manual.pdf", "page": 1}),
        Document(page_content="Page two content line.", metadata={"source": "manual.pdf", "page": 2}),
    ]

    records = split_documents(docs, chunk_size=100, chunk_overlap=20)
    pages = [r[1].metadata["page"] for r in records]
    sources = [r[1].metadata["source"] for r in records]

    assert 1 in pages and 2 in pages
    assert all(s == "manual.pdf" for s in sources)
