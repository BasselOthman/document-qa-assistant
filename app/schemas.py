"""Pydantic schemas for request/response models and structured outputs."""

from typing import List, Literal, Optional
from pydantic import BaseModel, Field


class SourceItem(BaseModel):
    """Metadata and excerpt for a retrieved context passage."""

    file: str = Field(description="Filename of the source document")
    page: int = Field(description="1-based page number where the chunk originates")
    snippet: str = Field(description="Extracted snippet or excerpt text")


class AskRequest(BaseModel):
    """Request payload for the /ask endpoint."""

    question: str = Field(..., min_length=1, description="Question to answer from the documents")
    top_k: int = Field(default=4, ge=1, le=20, description="Number of document chunks to retrieve")
    mode: Literal["graph", "simple"] = Field(
        default="graph", description="Execution pipeline mode: 'graph' (Corrective RAG) or 'simple' (Chain)"
    )
    strategy: Optional[Literal["dense", "hybrid", "hybrid_rerank"]] = Field(
        default=None, description="Retrieval strategy override: 'dense', 'hybrid', or 'hybrid_rerank'"
    )


class AskResponse(BaseModel):
    """Response payload for the /ask endpoint."""

    answer: str = Field(..., description="Generated answer or refusal statement")
    sources: List[SourceItem] = Field(default_factory=list, description="List of cited source chunks")
    mode: str = Field(..., description="Mode executed ('graph' or 'simple')")
    steps: List[str] = Field(
        default_factory=list,
        description="List of pipeline steps/nodes executed for operational transparency",
    )


class HealthResponse(BaseModel):
    """Health status and vector store statistics."""

    status: str = "ok"
    chroma_collection: str
    total_documents: int
    retrieval_strategy: str
    llm_model: str


class DocumentGrade(BaseModel):
    """Structured assessment of whether retrieved context is sufficient."""

    relevant: bool = Field(
        description="True if the retrieved passages contain sufficient information to answer the question, False otherwise."
    )
    reason: str = Field(
        description="One-line explanation supporting the grading decision."
    )


class QueryRewrite(BaseModel):
    """Structured output for question rewriting."""

    improved_query: str = Field(
        description="An optimized search query rewritten to retrieve better passages for the user's intent."
    )
