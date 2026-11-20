"""Simple RAG Chain: Sequential retrieve-then-generate pipeline."""

from typing import Any, Dict, List, Optional
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from app.llm import get_chat_model
from app.schemas import SourceItem
from app.vectorstore import retrieve_documents

SIMPLE_RAG_SYSTEM_PROMPT = """You are a precise, document-grounded Question Answering assistant.
Your instructions:
1. Answer the user's question ONLY using the factual information directly stated in the provided context passages below.
2. If the context does not contain enough information to answer the question, state clearly: "I cannot find the answer to this question in the provided documents."
3. Do not invent, extrapolate, or assume any information not present in the context.
4. When stating facts, explicitly reference the source filename and page number from the context metadata.
"""

SIMPLE_RAG_USER_TEMPLATE = """Context Passages:
{context}

Question: {question}

Helpful and strictly grounded answer:"""


def format_context(docs: List[Document]) -> str:
    """Format retrieved Document chunks into a readable string for prompt context."""
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 1)
        formatted.append(f"--- Document {i} (Source: {source}, Page: {page}) ---\n{doc.page_content}")
    return "\n\n".join(formatted)


def extract_sources(docs: List[Document]) -> List[SourceItem]:
    """Convert Document objects into structured SourceItem schemas."""
    sources = []
    for doc in docs:
        file = doc.metadata.get("source", "unknown")
        page = int(doc.metadata.get("page", 1))
        # Keep snippet clean and concise (first 250 characters)
        snippet = doc.page_content.strip()[:250].replace("\n", " ") + "..."
        sources.append(SourceItem(file=file, page=page, snippet=snippet))
    return sources


def run_simple_rag(
    question: str,
    top_k: int = 4,
    strategy: Optional[str] = None,
    chat_model: Any = None,
) -> Dict[str, Any]:
    """Execute the simple retrieve-then-generate RAG chain.

    Args:
        question: User query.
        top_k: Number of chunks to retrieve.
        strategy: Retrieval strategy ('dense', 'hybrid', 'hybrid_rerank').
        chat_model: Optional pre-configured chat model (useful for unit testing).

    Returns:
        Dictionary conforming to AskResponse structure.
    """
    steps = ["retrieve"]

    # 1. Retrieve context chunks
    docs = retrieve_documents(query=question, top_k=top_k, strategy=strategy)

    steps.append("generate")

    # If no documents were retrieved at all, return immediate refusal
    if not docs:
        return {
            "answer": "I cannot find the answer to this question in the provided documents.",
            "sources": [],
            "mode": "simple",
            "steps": steps,
        }

    # 2. Build prompt
    context_str = format_context(docs)
    user_prompt = SIMPLE_RAG_USER_TEMPLATE.format(context=context_str, question=question)

    messages = [
        SystemMessage(content=SIMPLE_RAG_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ]

    # 3. Invoke LLM
    llm = chat_model if chat_model is not None else get_chat_model(temperature=0.0)
    response = llm.invoke(messages)
    answer_text = response.content if hasattr(response, "content") else str(response)

    return {
        "answer": answer_text,
        "sources": extract_sources(docs),
        "mode": "simple",
        "steps": steps,
    }
