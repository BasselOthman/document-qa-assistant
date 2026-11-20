"""FastAPI Application for Document Q&A Assistant.

Endpoints:
- GET /health: Health check, document count, and configuration stats.
- POST /ask: Query the system using Simple RAG or Corrective RAG (CRAG) workflow.

Features:
- Structured JSON request logging middleware (auditable, secret-safe).
- Robust error handling for empty vector store, missing keys, and LLM failures.
"""

import json
import logging
import sys
import time
import uuid
from typing import Any, Dict

from fastapi import FastAPI, HTTPException, Request, Response, status
from fastapi.responses import JSONResponse
from starlette.middleware.base import BaseHTTPMiddleware

from app.chains import run_simple_rag
from app.config import get_settings
from app.graph import run_graph_rag
from app.llm import LLMConfigurationError
from app.schemas import AskRequest, AskResponse, HealthResponse
from app.vectorstore import EmptyVectorStoreError, get_document_count

# Configure dedicated structured logger
logger = logging.getLogger("rag_request_logger")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(logging.Formatter("%(message)s"))
if not logger.handlers:
    logger.addHandler(handler)


class StructuredLoggingMiddleware(BaseHTTPMiddleware):
    """Middleware logging one JSON line per request with operational telemetry."""

    async def dispatch(self, request: Request, call_next):
        request_id = str(uuid.uuid4())
        request.state.request_id = request_id
        start_time = time.perf_counter()

        # Placeholders populated during route handling
        request.state.rag_mode = "-"
        request.state.retrieval_strategy = "-"
        request.state.retrieved_chunks = 0
        request.state.llm_calls = 0
        request.state.graph_steps = []

        try:
            response = await call_next(request)
            status_code = response.status_code
        except Exception as exc:
            status_code = 500
            latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
            log_record = {
                "request_id": request_id,
                "endpoint": request.url.path,
                "mode": getattr(request.state, "rag_mode", "-"),
                "strategy": getattr(request.state, "retrieval_strategy", "-"),
                "latency_ms": latency_ms,
                "retrieved_chunks": getattr(request.state, "retrieved_chunks", 0),
                "llm_calls": getattr(request.state, "llm_calls", 0),
                "steps": getattr(request.state, "graph_steps", []),
                "status": status_code,
                "error": str(exc),
            }
            logger.info(json.dumps(log_record))
            raise exc

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        log_record = {
            "request_id": request_id,
            "endpoint": request.url.path,
            "mode": getattr(request.state, "rag_mode", "-"),
            "strategy": getattr(request.state, "retrieval_strategy", "-"),
            "latency_ms": latency_ms,
            "retrieved_chunks": getattr(request.state, "retrieved_chunks", 0),
            "llm_calls": getattr(request.state, "llm_calls", 0),
            "steps": getattr(request.state, "graph_steps", []),
            "status": status_code,
        }
        logger.info(json.dumps(log_record))
        return response


def create_app() -> FastAPI:
    """Application factory for FastAPI Document Q&A service."""
    app = FastAPI(
        title="Document Q&A Assistant",
        description="Production-grade RAG service featuring LangChain building blocks, LangGraph Corrective RAG, and multi-strategy retrieval.",
        version="1.0.0",
    )

    app.add_middleware(StructuredLoggingMiddleware)

    @app.exception_handler(EmptyVectorStoreError)
    async def empty_store_handler(request: Request, exc: EmptyVectorStoreError):
        return JSONResponse(
            status_code=status.HTTP_400_BAD_REQUEST,
            content={"detail": str(exc)},
        )

    @app.exception_handler(LLMConfigurationError)
    async def missing_key_handler(request: Request, exc: LLMConfigurationError):
        return JSONResponse(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            content={"detail": str(exc)},
        )

    @app.get("/health", response_model=HealthResponse, tags=["Observability"])
    async def health_check():
        """Report system health, indexed vector count, and runtime configuration."""
        settings = get_settings()
        total_docs = get_document_count()
        return HealthResponse(
            status="ok",
            chroma_collection=settings.chroma_collection,
            total_documents=total_docs,
            retrieval_strategy=settings.retrieval_strategy,
            llm_model=settings.openai_model,
        )

    @app.post("/ask", response_model=AskResponse, tags=["Document Q&A"])
    async def ask(payload: AskRequest, request: Request):
        """Answer questions from ingested documents using Simple RAG or LangGraph CRAG."""
        settings = get_settings()
        active_strategy = payload.strategy or settings.retrieval_strategy

        # Populate request state for structured logging
        request.state.rag_mode = payload.mode
        request.state.retrieval_strategy = active_strategy

        try:
            if payload.mode == "simple":
                result = run_simple_rag(
                    question=payload.question,
                    top_k=payload.top_k,
                    strategy=active_strategy,
                )
            else:
                result = run_graph_rag(
                    question=payload.question,
                    top_k=payload.top_k,
                    strategy=active_strategy,
                )

            # Compute telemetry for logging
            steps = result.get("steps", [])
            sources = result.get("sources", [])
            request.state.retrieved_chunks = len(sources)
            request.state.graph_steps = steps

            # LLM call accounting:
            # - grade_documents: 1 call
            # - rewrite_query: 1 call per rewrite
            # - generate: 1 call
            # - simple mode generate: 1 call if sources > 0
            if payload.mode == "simple":
                request.state.llm_calls = 1 if len(sources) > 0 else 0
            else:
                llm_nodes = {"grade_documents", "rewrite_query", "generate"}
                request.state.llm_calls = sum(1 for step in steps if step in llm_nodes)

            return AskResponse(
                answer=result["answer"],
                sources=sources,
                mode=result["mode"],
                steps=steps,
            )

        except (EmptyVectorStoreError, LLMConfigurationError):
            raise
        except Exception as err:
            logger.error(f"Error handling question: {err}", exc_info=True)
            raise HTTPException(
                status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                detail=f"An error occurred while processing the request: {str(err)}",
            )

    return app


app = create_app()

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
