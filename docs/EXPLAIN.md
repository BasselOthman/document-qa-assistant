# Document Q&A Assistant: Comprehensive Technical Guide & Interview Prep

This document is written for engineers who need to understand and explain every component of this Retrieval-Augmented Generation (RAG) system during technical interviews, portfolio presentations, and architectural reviews.

---

## 1. What is RAG and Why Use It?

### The Core Problem
Large Language Models (LLMs) like GPT-4o, Claude, or Llama are trained on massive public datasets. However, in production they suffer from two critical limitations:
1. **Knowledge Cutoffs:** They have no knowledge of events or literature published after their training date.
2. **Lack of Private / Domain Knowledge:** They cannot see internal company documents, proprietary source code, or private databases.
3. **Hallucinations:** When prompted about unknown facts, models tend to generate plausible-sounding but factually fabricated answers.

### The RAG Solution
**Retrieval-Augmented Generation (RAG)** decouples **parametric memory** (the fixed weights learned during model training) from **non-parametric memory** (an external, dynamically updatable knowledge index such as Chroma or BM25).

Instead of asking the LLM to memorize facts:
1. When a question arrives, we first **retrieve** the most relevant passages from our local document collection.
2. We inject those passages into the prompt as factual context.
3. We instruct the LLM: *"Answer strictly and only from the provided passages. If not present, state that you cannot find the answer."*

This dramatically reduces hallucination, enables citations (source file and page number), and lets you update system knowledge in seconds simply by indexing new files without re-training models.

---

## 2. Repository Architecture & File Walkthrough

| File | Purpose & Responsibilities |
| :--- | :--- |
| `app/config.py` | Centralized typed settings powered by `pydantic-settings`. Loads parameters (paths, chunk sizes, top_k, model names, API keys) from environment variables or `.env`. |
| `app/schemas.py` | Pydantic v2 data models for HTTP request/response payloads (`AskRequest`, `AskResponse`, `HealthResponse`) and structured LLM outputs (`DocumentGrade`, `QueryRewrite`). |
| `app/llm.py` | Factory returning configured `ChatOpenAI` instances. Adheres to the OpenAI-compatible standard, making it drop-in compatible with OpenAI, Groq, Together AI, or local Ollama. |
| `app/vectorstore.py` | Vector store management. Initializes local `FastEmbedEmbeddings` (BAAI/bge-small-en-v1.5), connects to persistent Chroma, manages BM25 serialization, and implements Dense, Hybrid, and Hybrid+Rerank strategies. |
| `app/ingest.py` | CLI tool (`python -m app.ingest`). Loads PDFs page-by-page, performs recursive character chunking, computes deterministic SHA-256 chunk IDs to prevent duplicate vectors, and builds Chroma + BM25 indices. |
| `app/chains.py` | Mode 1: A straightforward, single-turn RAG chain (Retrieve context -> Format prompt -> Generate answer with citations). |
| `app/graph.py` | Mode 2: Corrective RAG (CRAG) workflow implemented using LangGraph `StateGraph`. Implements document relevance grading, query rewriting loops, and zero-cost fallback refusal. |
| `app/main.py` | FastAPI application exposing `GET /health` and `POST /ask`. Houses non-leaking structured JSON request logging middleware and custom HTTP exception handlers. |
| `eval/questions.json` | Curated evaluation dataset containing 20 answerable questions with ground-truth sources and keywords, plus 5 unanswerable questions. |
| `eval/run_eval.py` | Automated evaluation harness measuring retrieval hit rate, keyword match correctness, refusal accuracy, latency, and LLM call counts across modes and strategies, logging runs to MLflow. |
| `Dockerfile` | Multi-stage, security-hardened container definition running on Python 3.11-slim as a non-root user (`appuser`), with build-time embedding model pre-caching. |
| `docker-compose.yml` | Container orchestration service defining local persistent volume mounts for the Chroma database. |

---

## 3. Chunk Size (800) and Overlap (150): The Engineering Rationale

When processing raw text, we chunk documents using `RecursiveCharacterTextSplitter`:
- **`chunk_size = 800` characters (~150 to 200 English words):**
  - If chunks are too small (e.g., 100 characters), they lose semantic meaning. An answer often requires a complete sentence or short paragraph explaining a concept.
  - If chunks are too large (e.g., 3,000 characters), each chunk contains multiple disparate topics. When embedded into a single vector, the specific detail gets diluted in vector space. Furthermore, large chunks consume excessive context window tokens and increase LLM inference cost and latency.
  - 800 characters provides the sweet spot: large enough to capture complete technical explanations, compact enough to maintain sharp embedding vectors.
- **`chunk_overlap = 150` characters (~25 to 35 words):**
  - Natural human language does not split neatly at fixed character counts. If an important sentence spans across a chunk boundary, cutting it in half would ruin both chunks.
  - An overlap of 150 characters guarantees that the end of chunk $N$ repeats at the start of chunk $N+1$, ensuring thoughts and key phrases remain intact across boundaries.

---

## 4. Embeddings and Similarity Search

### What is an Embedding?
An embedding model converts a piece of text into a high-dimensional vector of floating-point numbers (for `BAAI/bge-small-en-v1.5`, this vector has 384 dimensions). Texts with similar semantic meanings are placed close together in this 384-dimensional space.

### Why FastEmbed?
Instead of using PyTorch or requiring an external GPU/API, FastEmbed runs lightweight ONNX runtime quantization on CPU. It is extremely fast, uses minimal RAM, and requires no GPU drivers.

### How Similarity Search Works
When a user asks: *"What does RAG decouple?"*:
1. The question is embedded into a 384-dimensional query vector $\vec{q}$.
2. Chroma performs Approximate Nearest Neighbor (ANN) search against all stored chunk vectors $\vec{d}_i$ using Cosine Similarity:
   $$\text{Cosine Similarity}(\vec{q}, \vec{d}_i) = \frac{\vec{q} \cdot \vec{d}_i}{\|\vec{q}\| \|\vec{d}_i\|}$$
3. Chunks with the highest cosine similarity are retrieved.

---

## 5. LangChain vs. LangGraph: Why Both?

| Aspect | LangChain | LangGraph |
| :--- | :--- | :--- |
| **Primary Focus** | Modular building blocks & integrations. | Cyclic state machines & complex agentic workflows. |
| **Control Flow** | Linear Directed Acyclic Graphs (chains: $A \to B \to C$). | Cycles, loops, conditional branching, state persistence. |
| **State Handling** | Implicit string inputs and outputs passed through pipes (`\|`). | Explicit, typed state dictionary (`TypedDict`) persisted and audited at each node. |
| **Role in this Project** | Document loaders (`PyPDFLoader`), splitters (`RecursiveCharacterTextSplitter`), vector store abstraction (`Chroma`), embeddings (`FastEmbedEmbeddings`), BM25 (`BM25Retriever`). | The Corrective RAG (CRAG) decision machine: coordinating retrieval, grading, query rewrites, and fallback exits. |

**Key Takeaway:** LangChain provides the tools and wrappers; LangGraph orchestrates the multi-step reasoning and corrective recovery flow.

---

## 6. The LangGraph Corrective RAG (CRAG) Workflow

```
   [START]
      │
      ▼
  [retrieve]  ◄───────────┐
      │                   │ (rewrites < max_rewrites)
      ▼                   │
[grade_documents]         │
      │                   │
      ├─(not relevant)──►[rewrite_query]
      │
      ├─(relevant)──────►[generate] ──► [END]
      │
      └─(not relevant & rewrites >= max)──►[fallback] ──► [END]
```

### State Definition (`GraphState`)
Our state is a Python `TypedDict` containing:
- `question`: The user's original inquiry.
- `current_query`: The current search string (updated if rewritten).
- `documents`: List of retrieved `Document` chunks.
- `relevant`: Boolean flag indicating if retrieved passages are sufficient.
- `answer`: The final synthesized answer or refusal message.
- `rewrites`: Counter tracking how many query rewrites have occurred.
- `steps`: List of string names recording every node executed (e.g., `["retrieve", "grade_documents", "generate"]`).

### The 5 Nodes
1. **`retrieve`**: Fetches top-$k$ chunks for `current_query` using the chosen retrieval strategy.
2. **`grade_documents`**: Invokes ONE structured LLM call (`DocumentGrade` model) to determine whether the passages contain enough factual evidence to answer the question.
3. **`rewrite_query`**: If passages are insufficient, an LLM rewrites the query into a more specific, keyword-dense search term and increments `rewrites`.
4. **`generate`**: If context is relevant, synthesizes the answer with strict grounding and file/page citations at `temperature=0.0`.
5. **`fallback`**: If context is missing and rewrite limits are exhausted, returns an immediate, deterministic refusal without calling an LLM.

### Why Loop At Most Once (`MAX_REWRITES = 1`)?
Unconstrained agent loops risk **infinite cycles**, explosive token costs, and high latency. Setting `MAX_REWRITES = 1` allows one intelligent self-correction attempt. If a second retrieval still yields no relevant context, the question is deemed out-of-scope, and the system exits cleanly via `fallback`.

---

## 7. Reducing Hallucination via Grading & Fallback

In a naive RAG chain:
- User asks: *"What is the password to the company vault?"*
- Retriever finds irrelevant chunks about system architecture.
- LLM is forced to generate an answer and often hallucinates or makes speculative guesses.

In our Corrective RAG system:
- **Active Verification:** The `grade_documents` node acts as an automated judge before generation ever takes place.
- **Circuit Breaker:** If the retrieved text lacks the answer, the graph refuses to call the generator.
- **Zero-Cost Fallback:** The `fallback` node returns a canned refusal: *"I cannot find the answer to this question in the provided documents."* This guarantees 0% hallucination on out-of-scope questions while saving generation tokens.

---

## 8. BM25 vs. Dense Embeddings & Why Hybrid Retrieval Wins

### The Limitations of Dense Retrieval Alone
Dense embeddings represent text semantics. However:
- Dense models compress words into a fixed vector and often miss exact alphanumeric sequences (e.g., `ERR-9042`, SKU `B08N5WRWNW`, variable names like `max_retries`).
- High semantic similarity does not always mean exact lexical relevance.

### The Power of BM25 (Sparse Lexical Search)
BM25 (Best Matching 25) calculates Term Frequency (TF) and Inverse Document Frequency (IDF).
- If a query asks for `ERR-9042`, BM25 immediately zeroes in on documents containing that exact rare token.
- Weakness: BM25 has no understanding of synonyms (e.g., searching for "automobile" will miss a document saying "car").

### Why Hybrid (Ensemble) Retrieval Wins
The `hybrid` strategy uses LangChain's `EnsembleRetriever` with Reciprocal Rank Fusion (RRF):
$$RRF\_Score(d) = \sum_{m \in \{\text{dense}, \text{bm25}\}} \frac{w_m}{60 + \text{rank}_m(d)}$$
By combining BM25 ($w=0.5$) and Dense ($w=0.5$), the system achieves the best of both worlds: conceptual semantic understanding and exact keyword precision.

---

## 9. What a Cross-Encoder Reranker Does and Why It is Slower

### Bi-Encoders vs. Cross-Encoders
- **Bi-Encoder (First-Stage / Dense):** Embeds the query $\vec{q}$ and document $\vec{d}$ independently into vectors. Token-to-token interaction across the two texts is never computed. Fast ($O(1)$ vector lookup), but lower precision.
- **Cross-Encoder (Second-Stage / Reranker):** Takes both the query and document concatenated as a single sequence `[CLS] Query [SEP] Document [SEP]` and feeds them into multi-head self-attention. Every token in the query attends to every token in the document.

### Why is it slower?
The cross-encoder must run full neural network forward passes for every candidate pair at query time. Evaluating 15 candidates adds ~30-50 milliseconds of CPU computation.

### Why use `hybrid_rerank`?
By retrieving 15 candidates with hybrid search and reranking them down to the top 4, irrelevant chunks are pushed to the bottom and the most pertinent facts are placed right at the top of the prompt context, maximizing answer quality.

---

## 10. Experiment Tracking with MLflow

### What is MLflow?
MLflow is an open-source platform for managing the machine learning lifecycle. In this project, it is used for local experiment tracking in `./mlruns`.

### What We Track
For every combination of mode (`simple`, `graph`) and retrieval strategy (`dense`, `hybrid`, `hybrid_rerank`), `eval/run_eval.py` logs:
- **Parameters:** `mode`, `strategy`, `chunk_size`, `chunk_overlap`, `top_k`, `model_name`, `max_rewrites`.
- **Metrics:** `retrieval_hit_rate`, `answer_correctness`, `refusal_rate`, `avg_latency_seconds`, `avg_llm_calls`.
- **Artifacts:** `eval/results.json`.

This allows data-driven engineering decisions: you can objectively prove to an employer or stakeholder whether a reranker is worth the extra latency.

---

## 11. Dockerfile Explained Line by Line

```dockerfile
# 1. Official minimal Debian base with Python 3.11
FROM python:3.11-slim

# 2. Optimization flags: disable bytecode files (.pyc) and disable stdout buffering
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8000 \
    HOME=/home/appuser

# 3. Install curl for Docker health check and clean package lists to minimize image size
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# 4. Set container working directory
WORKDIR /app

# 5. Create a non-root system user and home directory for least-privilege security
RUN groupadd -r appuser && useradd -r -g appuser -d /home/appuser -m -s /bin/bash appuser

# 6. Copy requirements separately to leverage Docker layer caching
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 7. Pre-download FastEmbed models at build time so the container boots instantly and works offline
RUN python -c "from fastembed import TextEmbedding; TextEmbedding(model_name='BAAI/bge-small-en-v1.5'); from fastembed.rerank.cross_encoder import TextCrossEncoder; TextCrossEncoder(model_name='Xenova/ms-marco-MiniLM-L-6-v2')"

# 8. Copy application code and PDF assets
COPY app/ ./app/
COPY data/ ./data/

# 9. Ensure the non-root user has full permissions over the working directory
RUN chown -R appuser:appuser /app /home/appuser

# 10. Switch from root to appuser
USER appuser

# 11. Define Docker container healthcheck
HEALTHCHECK --interval=30s --timeout=5s --start-period=15s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# 12. Declare container port
EXPOSE 8000

# 13. Launch production ASGI server
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
```

---

## 12. Interview Questions and Answers (Part 1: Core System)

### Q1: What happens if a user submits a question completely unrelated to the indexed documents?
**Answer:** In simple mode, the prompt instructs the model to refuse if facts are missing. In graph mode, the `grade_documents` node grades the context as not relevant; after one rewrite fails to find relevant text, it routes to `fallback` which immediately returns a clean refusal without making a final generator LLM call.

### Q2: Why did you choose Chroma over a hosted vector database like Pinecone or Weaviate?
**Answer:** Chroma runs entirely embedded and local on disk, requiring zero external infrastructure or network roundtrips. For single-node document Q&A, it keeps deployment simple and cost-free while maintaining sub-millisecond retrieval.

### Q3: Why use FastEmbed instead of Hugging Face `sentence-transformers` with PyTorch?
**Answer:** `sentence-transformers` requires PyTorch, which bloats the Docker image by 2GB+ and consumes substantial RAM. FastEmbed uses ONNX runtime, producing a small Docker image that runs fast on commodity CPUs with minimal dependencies.

### Q4: How do you prevent duplicate chunks when running `python -m app.ingest` multiple times?
**Answer:** Each chunk is assigned a deterministic SHA-256 hash computed over `source`, `page`, `chunk_index`, and `content`. Ingestion queries existing IDs in Chroma and only inserts chunks whose IDs are not already present.

### Q5: What is the purpose of `temperature=0.0`?
**Answer:** Setting temperature to 0.0 minimizes randomness and makes generation deterministic, ensuring the model adheres strictly to the retrieved facts rather than inventing creative completions.

### Q6: How does the `/ask` response demonstrate graph execution in a demo?
**Answer:** The response JSON includes a `steps` list (e.g. `["retrieve", "grade_documents", "rewrite_query", "retrieve", "grade_documents", "generate"]`), showing the audience exactly which decision path the state machine traversed.

### Q7: What are the main limitations of this RAG pipeline?
**Answer:** 
1. Complex tables and charts inside PDFs are converted to raw text and lose two-dimensional layout context.
2. In-memory BM25 index scales well up to hundreds of thousands of chunks, but enterprise multi-terabyte search would require a distributed inverted index like OpenSearch or Elasticsearch.
3. Multi-turn conversational memory is not implemented; each query is treated as an independent state execution.

### Q8: How would you handle table extraction from PDFs?
**Answer:** I would integrate specialized layout-aware parsers such as `pdfplumber`, `unstructured`, or a visual vision-language model (e.g., Table-Transformer or Gemini Vision) to convert tables into Markdown or HTML syntax before chunking.

### Q9: Why did you implement structured output for the document grading step?
**Answer:** Parsing raw text strings for "yes" or "no" is fragile and prone to parsing errors. Using Pydantic models (`DocumentGrade` with `relevant: bool`) leverages tool/function calling for guaranteed JSON schema conformity.

### Q10: How are secrets protected in this application?
**Answer:** API keys are never hardcoded. They are loaded at runtime via environment variables or `.env`. Furthermore, our FastAPI structured logging middleware explicitly logs only telemetry metrics and never records auth headers or keys.

### Q11: What is the difference between Reciprocal Rank Fusion (RRF) and simple score addition?
**Answer:** BM25 scores are unbounded positive numbers ($0$ to $\infty$), while dense similarity scores range from $-1$ to $1$. Directly adding them is invalid because their distributions differ. RRF normalizes across systems by ranking positions rather than raw scores.

### Q12: Why is the fallback node designed to make zero LLM calls?
**Answer:** If the grading node has already confirmed that the context is insufficient, calling the LLM again would simply waste money, add latency, and risk hallucination. Returning a deterministic refusal is both faster and 100% reliable.

### Q13: What does the CI pipeline in GitHub Actions test?
**Answer:** On every push and pull request, GitHub Actions launches a clean Ubuntu environment with Python 3.11, installs `requirements.txt`, and runs the complete `pytest` suite including mock routing and API checks.

### Q14: How does the application handle an empty vector store request?
**Answer:** `app/vectorstore.py` checks `vectorstore._collection.count()`. If zero, it raises `EmptyVectorStoreError`, which FastAPI's exception handler translates to an explicit HTTP 400 Bad Request instructing the user to run ingestion first.

### Q15: How would you scale this application to millions of documents?
**Answer:**
1. Migrate the Chroma storage to a clustered vector database (e.g., Qdrant or Milvus).
2. Offload BM25 to an OpenSearch/Elasticsearch cluster.
3. Decouple document ingestion into an asynchronous Celery/Kafka worker queue.
4. Put the FastAPI service behind a load balancer with horizontal pod autoscaling.

---

## 13. Interview Questions and Answers (Part 2: Advanced Retrieval & Tracking)

### Q16: Why can't we just use a cross-encoder for the entire retrieval process?
**Answer:** Computational complexity. Cross-encoders require computing all-to-all attention for every candidate text pair. Searching 100,000 documents with a cross-encoder would take minutes per query. A two-stage pipeline uses bi-encoders/BM25 for high recall in milliseconds, then uses the cross-encoder only on the top 15 candidates.

### Q17: What is the risk of setting `MAX_REWRITES` too high (e.g., 5)?
**Answer:** Each rewrite adds at least two LLM calls (rewrite + grade) and a vector retrieval roundtrip. A limit of 5 could add 10+ seconds of latency and multiply token costs tenfold, while likely failing anyway if the knowledge is truly absent.

### Q18: What is the purpose of structured JSON logging in production?
**Answer:** Plain text logs cannot be efficiently filtered, aggregated, or queried in log analytics systems like Datadog, CloudWatch, or Elasticsearch. Structured JSON allows engineers to instantly build dashboards tracking p99 latency, refusal percentages, and retrieval error rates.

### Q19: In `eval/run_eval.py`, why do you evaluate retrieval hit rate separately from generation?
**Answer:** A RAG failure can occur in two distinct places: retrieval failure (the search engine didn't find the right page) or generation failure (the context had the answer, but the LLM failed to extract it). Measuring retrieval hit rate independently isolates search quality from language model reasoning.

### Q20: What is the difference between local file tracking in MLflow versus a remote MLflow server?
**Answer:** Local tracking writes runs and artifacts to `./mlruns` on the local filesystem, requiring zero cloud setup. A remote MLflow tracking server connects to a shared Postgres database and S3 bucket, allowing teams to compare experiment runs collaboratively.

### Q21: What happens if the BM25 index and Chroma vector database get out of sync?
**Answer:** In hybrid search, documents found in BM25 might not match Chroma metadata. In this project, `app/ingest.py` guarantees synchronization by rebuilding both the Chroma embeddings and the serialized BM25 pickle file simultaneously from the exact same list of chunk objects.

### Q22: Why did you choose `BAAI/bge-small-en-v1.5` over OpenAI `text-embedding-3-small`?
**Answer:**
1. **Privacy & Independence:** Embeddings are generated locally on CPU without external API dependencies or data transmission.
2. **Zero Ingestion Cost:** Indexing thousands of pages incurs zero API costs.
3. **MTEB Benchmark Performance:** BGE-small is one of the top-ranking lightweight models on the Massive Text Embedding Benchmark.

### Q23: How would you evaluate answer quality if keyword matching is not sufficient?
**Answer:** For advanced evaluation, I would implement **RAG Triad metrics** using LLM-as-a-judge (e.g., TruLens, Ragas, or DeepEval):
1. **Context Relevance:** Did the retriever find information pertinent to the question?
2. **Groundedness / Faithfulness:** Is every claim in the answer supported by the retrieved context?
3. **Answer Relevance:** Does the answer directly address the user's question?
