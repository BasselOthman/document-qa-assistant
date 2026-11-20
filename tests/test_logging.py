"""Tests for structured request logging middleware."""

import json
import logging
from unittest.mock import MagicMock, patch
from fastapi.testclient import TestClient
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.main import app

client = TestClient(app)


class LogCaptureHandler(logging.Handler):
    """Custom in-memory log record capture handler."""

    def __init__(self):
        super().__init__()
        self.records = []

    def emit(self, record):
        self.records.append(self.format(record))


@patch("app.main.get_document_count", return_value=10)
def test_structured_logging_on_health_request(mock_doc_count):
    """Verify GET /health logs a compliant structured JSON entry without leaking secrets."""
    test_logger = logging.getLogger("rag_request_logger")
    capture_handler = LogCaptureHandler()
    capture_handler.setFormatter(logging.Formatter("%(message)s"))
    test_logger.addHandler(capture_handler)

    try:
        response = client.get("/health")
        assert response.status_code == 200

        # Find JSON log line
        json_logs = []
        for line in capture_handler.records:
            try:
                parsed = json.loads(line)
                if "endpoint" in parsed:
                    json_logs.append(parsed)
            except Exception:
                continue

        assert len(json_logs) >= 1, "Expected at least one structured JSON log line"
        entry = json_logs[-1]

        # Check required schema fields
        assert "request_id" in entry
        assert entry["endpoint"] == "/health"
        assert "latency_ms" in entry
        assert entry["status"] == 200
        assert "retrieved_chunks" in entry
        assert "llm_calls" in entry
        assert "steps" in entry

        # Verify no secret fields exist
        assert "api_key" not in entry
        assert "openai_api_key" not in entry
        assert "authorization" not in str(entry).lower()

    finally:
        test_logger.removeHandler(capture_handler)


@patch("app.chains.retrieve_documents")
@patch("app.chains.get_chat_model")
def test_structured_logging_on_ask_request(mock_get_llm, mock_retrieve):
    """Verify POST /ask captures telemetry: mode, strategy, chunks count, and status."""
    mock_retrieve.return_value = [
        Document(page_content="Snippet 1", metadata={"source": "doc.pdf", "page": 1}),
        Document(page_content="Snippet 2", metadata={"source": "doc.pdf", "page": 2}),
    ]
    mock_llm = MagicMock()
    mock_llm.invoke.return_value = AIMessage(content="Grounded answer text.")
    mock_get_llm.return_value = mock_llm

    test_logger = logging.getLogger("rag_request_logger")
    capture_handler = LogCaptureHandler()
    capture_handler.setFormatter(logging.Formatter("%(message)s"))
    test_logger.addHandler(capture_handler)

    try:
        payload = {
            "question": "What is the logging format?",
            "mode": "simple",
            "strategy": "hybrid",
        }
        response = client.post("/ask", json=payload)
        assert response.status_code == 200

        json_logs = []
        for line in capture_handler.records:
            try:
                parsed = json.loads(line)
                if parsed.get("endpoint") == "/ask":
                    json_logs.append(parsed)
            except Exception:
                continue

        assert len(json_logs) >= 1
        entry = json_logs[-1]

        assert entry["endpoint"] == "/ask"
        assert entry["mode"] == "simple"
        assert entry["strategy"] == "hybrid"
        assert entry["retrieved_chunks"] == 2
        assert entry["llm_calls"] == 1
        assert entry["status"] == 200
        assert entry["steps"] == ["retrieve", "generate"]

    finally:
        test_logger.removeHandler(capture_handler)
