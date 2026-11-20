"""API endpoint tests for GET /health and POST /ask across modes and error scenarios."""

from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.llm import LLMConfigurationError
from app.main import app
from app.schemas import DocumentGrade
from app.vectorstore import EmptyVectorStoreError

client = TestClient(app)

SAMPLE_DOC = Document(
    page_content="RAG architectures connect models to external knowledge.",
    metadata={"source": "rag_handbook.pdf", "page": 1},
)


@patch("app.main.get_document_count")
def test_get_health(mock_doc_count):
    """Verify /health returns 200 with vector store document count."""
    mock_doc_count.return_value = 42

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "ok"
    assert data["total_documents"] == 42
    assert "chroma_collection" in data


@patch("app.chains.retrieve_documents")
@patch("app.chains.get_chat_model")
def test_post_ask_simple_mode(mock_get_llm, mock_retrieve):
    """Verify POST /ask executes simple mode and returns answer with citations."""
    mock_retrieve.return_value = [SAMPLE_DOC]

    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(
        content="Based on rag_handbook.pdf (page 1), RAG connects models to external knowledge."
    )
    mock_get_llm.return_value = mock_llm

    payload = {
        "question": "What does RAG connect models to?",
        "top_k": 2,
        "mode": "simple",
    }
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert "Based on rag_handbook.pdf" in data["answer"]
    assert data["mode"] == "simple"
    assert data["steps"] == ["retrieve", "generate"]
    assert len(data["sources"]) == 1
    assert data["sources"][0]["file"] == "rag_handbook.pdf"
    assert data["sources"][0]["page"] == 1


@patch("app.graph.retrieve_documents")
@patch("app.graph.get_chat_model")
def test_post_ask_graph_mode(mock_get_llm, mock_retrieve):
    """Verify POST /ask executes LangGraph CRAG mode and returns steps trace."""
    mock_retrieve.return_value = [SAMPLE_DOC]

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = DocumentGrade(relevant=True, reason="Context is sufficient.")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="Generated answer via CRAG.")
    mock_get_llm.return_value = mock_llm

    payload = {
        "question": "Explain CRAG workflow",
        "top_k": 2,
        "mode": "graph",
    }
    response = client.post("/ask", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["mode"] == "graph"
    assert data["steps"] == ["retrieve", "grade_documents", "generate"]
    assert len(data["sources"]) >= 1


@patch("app.chains.retrieve_documents")
def test_empty_vector_store_error_handling(mock_retrieve):
    """Verify 400 error is returned when vector store has not been indexed."""
    mock_retrieve.side_effect = EmptyVectorStoreError(
        "Vector store is empty. Please run 'python -m app.ingest' first."
    )

    payload = {"question": "What is RAG?", "mode": "simple"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 400
    data = response.json()
    assert "Vector store is empty" in data["detail"]


@patch("app.chains.retrieve_documents")
@patch("app.chains.get_chat_model")
def test_missing_api_key_error_handling(mock_get_llm, mock_retrieve):
    """Verify 503 error is returned when OPENAI_API_KEY is missing."""
    mock_retrieve.return_value = [SAMPLE_DOC]
    mock_get_llm.side_effect = LLMConfigurationError("API key is not configured.")

    payload = {"question": "What is RAG?", "mode": "simple"}
    response = client.post("/ask", json=payload)

    assert response.status_code == 503
    data = response.json()
    assert "API key is not configured" in data["detail"]
