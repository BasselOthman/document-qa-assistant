"""Tests for LangGraph Corrective RAG state machine and all 3 routing paths."""

from unittest.mock import MagicMock, patch
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from app.graph import FALLBACK_ANSWER, GraphState, build_graph
from app.schemas import DocumentGrade, QueryRewrite

DUMMY_DOC = Document(
    page_content="RAG systems decouple parametric memory from external vector knowledge.",
    metadata={"source": "handbook.pdf", "page": 1},
)


@patch("app.graph.retrieve_documents")
def test_path_1_relevant_on_first_try(mock_retrieve):
    """Path 1: retrieve -> grade (relevant=True) -> generate -> END."""
    mock_retrieve.return_value = [DUMMY_DOC]

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    mock_structured.invoke.return_value = DocumentGrade(relevant=True, reason="Passage is sufficient.")
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="RAG decouples parametric and non-parametric memory.")

    graph = build_graph(chat_model=mock_llm, max_rewrites=1)

    initial_state: GraphState = {
        "question": "What does RAG decouple?",
        "current_query": "What does RAG decouple?",
        "documents": [],
        "relevant": False,
        "answer": "",
        "rewrites": 0,
        "steps": [],
        "top_k": 2,
        "strategy": "dense",
    }

    final_state = graph.invoke(initial_state)

    assert final_state["steps"] == ["retrieve", "grade_documents", "generate"]
    assert "parametric" in final_state["answer"]
    assert final_state["relevant"] is True
    assert final_state["rewrites"] == 0


@patch("app.graph.retrieve_documents")
def test_path_2_rewrite_then_relevant(mock_retrieve):
    """Path 2: retrieve -> grade (False) -> rewrite_query -> retrieve -> grade (True) -> generate -> END."""
    mock_retrieve.return_value = [DUMMY_DOC]

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    # 1. First grading: False
    # 2. Query rewrite call
    # 3. Second grading: True
    mock_structured.invoke.side_effect = [
        DocumentGrade(relevant=False, reason="Initial passage not relevant."),
        QueryRewrite(improved_query="RAG memory decoupling technical details"),
        DocumentGrade(relevant=True, reason="Rewritten retrieval provided relevant context."),
    ]
    mock_llm.with_structured_output.return_value = mock_structured
    mock_llm.invoke.return_value = AIMessage(content="Generated answer after successful query rewrite.")

    graph = build_graph(chat_model=mock_llm, max_rewrites=1)

    initial_state: GraphState = {
        "question": "decouple memory?",
        "current_query": "decouple memory?",
        "documents": [],
        "relevant": False,
        "answer": "",
        "rewrites": 0,
        "steps": [],
        "top_k": 2,
        "strategy": "dense",
    }

    final_state = graph.invoke(initial_state)

    assert final_state["steps"] == [
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "retrieve",
        "grade_documents",
        "generate",
    ]
    assert final_state["rewrites"] == 1
    assert final_state["relevant"] is True
    assert "after successful query rewrite" in final_state["answer"]


@patch("app.graph.retrieve_documents")
def test_path_3_rewrite_then_fallback(mock_retrieve):
    """Path 3: retrieve -> grade (False) -> rewrite_query -> retrieve -> grade (False) -> fallback -> END."""
    mock_retrieve.return_value = [DUMMY_DOC]

    mock_llm = MagicMock()
    mock_structured = MagicMock()
    # 1. First grading: False
    # 2. Query rewrite
    # 3. Second grading: False (exhausts max_rewrites=1)
    mock_structured.invoke.side_effect = [
        DocumentGrade(relevant=False, reason="Passage does not contain answer."),
        QueryRewrite(improved_query="unanswerable question rewritten"),
        DocumentGrade(relevant=False, reason="Still insufficient information."),
    ]
    mock_llm.with_structured_output.return_value = mock_structured

    graph = build_graph(chat_model=mock_llm, max_rewrites=1)

    initial_state: GraphState = {
        "question": "What is the secret chef name?",
        "current_query": "What is the secret chef name?",
        "documents": [],
        "relevant": False,
        "answer": "",
        "rewrites": 0,
        "steps": [],
        "top_k": 2,
        "strategy": "dense",
    }

    final_state = graph.invoke(initial_state)

    assert final_state["steps"] == [
        "retrieve",
        "grade_documents",
        "rewrite_query",
        "retrieve",
        "grade_documents",
        "fallback",
    ]
    assert final_state["rewrites"] == 1
    assert final_state["relevant"] is False
    assert final_state["answer"] == FALLBACK_ANSWER
