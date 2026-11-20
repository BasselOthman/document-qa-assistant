"""LangGraph Corrective RAG (CRAG) workflow.

Implements a stateful routing graph:
1. retrieve: fetch top-k chunks for current_query
2. grade_documents: ONE structured LLM call evaluating context relevance
3. conditional routing:
   - if relevant -> generate
   - elif rewrites < max_rewrites -> rewrite_query (loops back to retrieve)
   - else -> fallback
4. generate: answer strictly from context with file & page citations
5. fallback: zero-cost refusal without LLM call
"""

from typing import Any, Dict, List, Literal, Optional, TypedDict
from langchain_core.documents import Document
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph

from app.config import get_settings
from app.llm import get_chat_model
from app.schemas import DocumentGrade, QueryRewrite, SourceItem
from app.vectorstore import retrieve_documents


class GraphState(TypedDict):
    """Workflow state persisted and updated across graph nodes."""

    question: str
    current_query: str
    documents: List[Document]
    relevant: bool
    answer: str
    rewrites: int
    steps: List[str]
    top_k: int
    strategy: Optional[str]


GRADE_PROMPT = """You are a strict relevance grader assessing whether retrieved context documents are sufficient to answer a user's question.

Retrieved Context:
{context}

User Question:
{question}

Evaluate whether the context contains the necessary factual information to formulate a reliable, accurate answer without guessing.
Return your decision as structured output with:
- relevant: true if context is sufficient; false if context is missing, irrelevant, or insufficient.
- reason: a concise one-line explanation.
"""

REWRITE_PROMPT = """You are an expert search query optimizer for a technical document retrieval system.
The previous search query failed to retrieve relevant documents for the user's question.

Initial User Question: {question}
Previous Query: {current_query}

Formulate an improved, concise search query focusing on technical keywords, synonyms, and specific terms that are likely to appear in the source documents.
Do not answer the question; only return the improved query.
"""

GENERATE_SYSTEM_PROMPT = """You are a precise, document-grounded Question Answering assistant.
Your instructions:
1. Answer the user's question ONLY using the factual information directly stated in the provided context passages below.
2. If the context does not contain enough information to answer the question, state clearly: "I cannot find the answer to this question in the provided documents."
3. Do not invent, extrapolate, or assume any information not present in the context.
4. When stating facts, explicitly reference the source filename and page number from the context metadata.
"""

GENERATE_USER_TEMPLATE = """Context Passages:
{context}

Question: {question}

Helpful and strictly grounded answer:"""

FALLBACK_ANSWER = "I cannot find the answer to this question in the provided documents."


def format_context_for_llm(docs: List[Document]) -> str:
    """Format Document chunks with source metadata for LLM ingestion."""
    if not docs:
        return "No documents retrieved."
    formatted = []
    for i, doc in enumerate(docs, 1):
        source = doc.metadata.get("source", "unknown")
        page = doc.metadata.get("page", 1)
        formatted.append(f"--- Document {i} (Source: {source}, Page: {page}) ---\n{doc.page_content}")
    return "\n\n".join(formatted)


def extract_sources_from_docs(docs: List[Document]) -> List[SourceItem]:
    """Convert Document objects into serialized SourceItem schemas."""
    sources = []
    for doc in docs:
        file = doc.metadata.get("source", "unknown")
        page = int(doc.metadata.get("page", 1))
        snippet = doc.page_content.strip()[:250].replace("\n", " ") + "..."
        sources.append(SourceItem(file=file, page=page, snippet=snippet))
    return sources


# --- Node Definitions ---

def make_retrieve_node(chat_model: Any = None):
    """Factory returning retrieve node function."""
    def retrieve(state: GraphState) -> Dict[str, Any]:
        query = state.get("current_query") or state["question"]
        top_k = state.get("top_k", 4)
        strategy = state.get("strategy")

        docs = retrieve_documents(query=query, top_k=top_k, strategy=strategy)
        steps = list(state.get("steps", [])) + ["retrieve"]
        return {"documents": docs, "steps": steps}

    return retrieve


def make_grade_node(chat_model: Any = None):
    """Factory returning grade_documents node function."""
    def grade_documents(state: GraphState) -> Dict[str, Any]:
        docs = state.get("documents", [])
        question = state["question"]
        steps = list(state.get("steps", [])) + ["grade_documents"]

        if not docs:
            return {"relevant": False, "steps": steps}

        context_str = format_context_for_llm(docs)
        prompt = GRADE_PROMPT.format(context=context_str, question=question)

        llm = chat_model if chat_model is not None else get_chat_model(temperature=0.0)

        try:
            structured_llm = llm.with_structured_output(DocumentGrade)
            grade: DocumentGrade = structured_llm.invoke(prompt)
            relevant = grade.relevant
        except Exception:
            # Fallback if structured output is not supported by mock/custom model
            raw_response = llm.invoke(prompt)
            content = raw_response.content if hasattr(raw_response, "content") else str(raw_response)
            relevant = "true" in content.lower() or "relevant" in content.lower()

        return {"relevant": relevant, "steps": steps}

    return grade_documents


def make_rewrite_node(chat_model: Any = None):
    """Factory returning rewrite_query node function."""
    def rewrite_query(state: GraphState) -> Dict[str, Any]:
        question = state["question"]
        current_query = state.get("current_query") or question
        rewrites = state.get("rewrites", 0) + 1
        steps = list(state.get("steps", [])) + ["rewrite_query"]

        prompt = REWRITE_PROMPT.format(question=question, current_query=current_query)
        llm = chat_model if chat_model is not None else get_chat_model(temperature=0.0)

        try:
            structured_llm = llm.with_structured_output(QueryRewrite)
            result = structured_llm.invoke(prompt)
            improved = result.improved_query.strip()
        except Exception:
            raw_response = llm.invoke(prompt)
            improved = (raw_response.content if hasattr(raw_response, "content") else str(raw_response)).strip()

        return {
            "current_query": improved,
            "rewrites": rewrites,
            "steps": steps,
        }

    return rewrite_query


def make_generate_node(chat_model: Any = None):
    """Factory returning generate node function."""
    def generate(state: GraphState) -> Dict[str, Any]:
        docs = state.get("documents", [])
        question = state["question"]
        steps = list(state.get("steps", [])) + ["generate"]

        context_str = format_context_for_llm(docs)
        user_prompt = GENERATE_USER_TEMPLATE.format(context=context_str, question=question)
        messages = [
            SystemMessage(content=GENERATE_SYSTEM_PROMPT),
            HumanMessage(content=user_prompt),
        ]

        llm = chat_model if chat_model is not None else get_chat_model(temperature=0.0)
        response = llm.invoke(messages)
        answer = (response.content if hasattr(response, "content") else str(response)).strip()

        return {"answer": answer, "steps": steps}

    return generate


def fallback_node(state: GraphState) -> Dict[str, Any]:
    """Zero-cost fallback node returned when no context is sufficient."""
    steps = list(state.get("steps", [])) + ["fallback"]
    return {"answer": FALLBACK_ANSWER, "steps": steps}


# --- Routing Logic ---

def make_router(max_rewrites: int):
    """Return routing function after document grading."""
    def route_after_grading(state: GraphState) -> Literal["generate", "rewrite_query", "fallback"]:
        if state.get("relevant", False):
            return "generate"
        if state.get("rewrites", 0) < max_rewrites:
            return "rewrite_query"
        return "fallback"

    return route_after_grading


def build_graph(chat_model: Any = None, max_rewrites: Optional[int] = None):
    """Construct and compile the LangGraph Corrective RAG StateGraph workflow.

    Args:
        chat_model: Optional chat model override (useful for testing).
        max_rewrites: Maximum query rewrite attempts (default from settings).

    Returns:
        Compiled StateGraph instance ready for invocation.
    """
    settings = get_settings()
    effective_max_rewrites = max_rewrites if max_rewrites is not None else settings.max_rewrites

    workflow = StateGraph(GraphState)

    # Register nodes
    workflow.add_node("retrieve", make_retrieve_node(chat_model))
    workflow.add_node("grade_documents", make_grade_node(chat_model))
    workflow.add_node("rewrite_query", make_rewrite_node(chat_model))
    workflow.add_node("generate", make_generate_node(chat_model))
    workflow.add_node("fallback", fallback_node)

    # Register edges
    workflow.add_edge(START, "retrieve")
    workflow.add_edge("retrieve", "grade_documents")

    workflow.add_conditional_edges(
        "grade_documents",
        make_router(effective_max_rewrites),
        {
            "generate": "generate",
            "rewrite_query": "rewrite_query",
            "fallback": "fallback",
        },
    )

    workflow.add_edge("rewrite_query", "retrieve")
    workflow.add_edge("generate", END)
    workflow.add_edge("fallback", END)

    return workflow.compile()


def get_mermaid_diagram() -> str:
    """Return Mermaid diagram string representing the compiled LangGraph workflow."""
    app = build_graph()
    try:
        return app.get_graph().draw_mermaid()
    except Exception:
        # Fallback accurate Mermaid diagram string
        return (
            "graph TD\n"
            "    __start__([Start]) --> retrieve[retrieve]\n"
            "    retrieve --> grade_documents[grade_documents]\n"
            "    grade_documents -->|relevant| generate[generate]\n"
            "    grade_documents -->|not relevant & rewrites < max| rewrite_query[rewrite_query]\n"
            "    grade_documents -->|not relevant & rewrites >= max| fallback[fallback]\n"
            "    rewrite_query --> retrieve\n"
            "    generate --> __end__([End])\n"
            "    fallback --> __end__([End])"
        )


def run_graph_rag(
    question: str,
    top_k: int = 4,
    strategy: Optional[str] = None,
    chat_model: Any = None,
) -> Dict[str, Any]:
    """Execute the Corrective RAG LangGraph workflow.

    Args:
        question: User query.
        top_k: Number of chunks to retrieve.
        strategy: Retrieval strategy ('dense', 'hybrid', 'hybrid_rerank').
        chat_model: Optional chat model override.

    Returns:
        Dictionary conforming to AskResponse structure.
    """
    app = build_graph(chat_model=chat_model)

    initial_state: GraphState = {
        "question": question,
        "current_query": question,
        "documents": [],
        "relevant": False,
        "answer": "",
        "rewrites": 0,
        "steps": [],
        "top_k": top_k,
        "strategy": strategy,
    }

    final_state = app.invoke(initial_state)

    docs = final_state.get("documents", [])
    # If the workflow routed to fallback, do not report sources that were deemed irrelevant
    sources = extract_sources_from_docs(docs) if final_state.get("relevant", False) else []

    return {
        "answer": final_state.get("answer", FALLBACK_ANSWER),
        "sources": sources,
        "mode": "graph",
        "steps": final_state.get("steps", []),
    }
