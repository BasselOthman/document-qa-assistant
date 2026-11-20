# Step-by-Step Container Deployment Guide: Render Free Tier

This guide provides clear, beginner-friendly instructions for deploying the Document Q&A Assistant Docker container to **Render**, a cloud platform that offers a free container hosting tier.

> **Disclaimer:** These cloud deployment steps were researched and documented according to Render's official documentation. They could not be executed directly in this local terminal environment.

---

## 1. Cloud Provider Overview: Render Free Tier

Render offers a Free Web Service tier suitable for hosting containerized APIs:
- **Pricing:** $0 / month (Free tier).
- **Resources:** 512 MB RAM, 0.1 shared vCPU.
- **Inactivity Spin-Down:** Free services spin down after 15 minutes of inactivity. Incoming requests wake the container up within 30-50 seconds.
- **Port:** Render automatically injects a `PORT` environment variable. Our Dockerfile and entrypoint adhere to `$PORT` (defaulting to 8000).

---

## 2. Prerequisites

Before starting, ensure you have:
1. A **GitHub account** with this repository pushed to your personal account (`https://github.com/<your-username>/document-qa-assistant`).
2. A free account on **Render** ([render.com](https://render.com)).
3. An OpenAI-compatible API key (e.g. from OpenAI or Groq).

---

## 3. Preparing Ingestion for Ephemeral Free Hosting

On Render's free tier, local disk storage is ephemeral (resets on container restart). To ensure the vector store and BM25 index are always ready when the container boots, we run ingestion on container startup if the database is not yet built.

You can customize the Docker start command in Render's dashboard:
```bash
sh -c "python -m app.ingest && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
```

---

## 4. Step-by-Step Deployment Walkthrough

### Step 1: Connect Your GitHub Repository
1. Log into your [Render Dashboard](https://dashboard.render.com).
2. Click **New +** in the top right corner and select **Web Service**.
3. Under *Connect a repository*, choose **GitHub**.
4. Select your `document-qa-assistant` repository and click **Connect**.

### Step 2: Configure Service Details
Fill in the basic settings:
- **Name:** `document-qa-assistant`
- **Region:** Choose the region closest to you (e.g., `Frankfurt (EU Central)` or `Oregon (US West)`).
- **Branch:** `main` (or `master`).
- **Language / Runtime:** Select **Docker**.

### Step 3: Configure Instance Type
- Under **Instance Type**, select **Free (512 MB RAM, 0.1 CPU)**.

### Step 4: Configure Secrets (Never Put Keys in the Docker Image!)
Scroll down to the **Environment Variables** section and click **Add Environment Variable** for each secret:

| Key | Example Value | Description |
| :--- | :--- | :--- |
| `OPENAI_API_KEY` | `sk-proj-...` or `gsk_...` | Your secret API key (kept encrypted by Render). |
| `OPENAI_API_BASE` | `https://api.openai.com/v1` | LLM API base URL (or `https://api.groq.com/openai/v1` for Groq). |
| `OPENAI_MODEL` | `gpt-4o-mini` | Model name. |
| `RETRIEVAL_STRATEGY` | `dense` | Default strategy (`dense`, `hybrid`, `hybrid_rerank`). |

### Step 5: Start Command Override (Optional)
Under **Docker Command Override**, enter:
```bash
sh -c "python -m app.ingest && uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"
```

### Step 6: Deploy
Click **Create Web Service**.
Render will:
1. Clone your repository.
2. Build the Dockerfile (installing Python dependencies and pre-caching the FastEmbed embedding model).
3. Start the container and execute health checks.

---

## 5. Verifying Your Live Deployment

Once the build logs show `==> Your service is live at https://document-qa-assistant.onrender.com`, test it using `curl`:

### 1. Test Health Endpoint
```bash
curl -X GET https://document-qa-assistant.onrender.com/health
```
**Expected Response:**
```json
{
  "status": "ok",
  "chroma_collection": "document_qa",
  "total_documents": 25,
  "retrieval_strategy": "dense",
  "llm_model": "gpt-4o-mini"
}
```

### 2. Test Ask Endpoint (LangGraph Corrective RAG)
```bash
curl -X POST https://document-qa-assistant.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the chunk overlap used in document splitting?",
    "mode": "graph",
    "top_k": 4
  }'
```

### 3. Test Fallback Refusal (Unanswerable Question)
```bash
curl -X POST https://document-qa-assistant.onrender.com/ask \
  -H "Content-Type: application/json" \
  -d '{
    "question": "What is the secret root API token for the AWS production cluster?",
    "mode": "graph"
  }'
```
**Expected Response:**
```json
{
  "answer": "I cannot find the answer to this question in the provided documents.",
  "sources": [],
  "mode": "graph",
  "steps": ["retrieve", "grade_documents", "rewrite_query", "retrieve", "grade_documents", "fallback"]
}
```
Notice how `steps` transparently indicates the graph routed through query rewrite and landed on `fallback`!

---

## 6. Updating Documents in Production

When you add new PDF manuals to `data/`:
1. Commit and push the new files to your GitHub repository:
   ```bash
   git add data/
   git commit -m "feat: add new technical manuals"
   git push origin main
   ```
2. Render will automatically detect the commit, trigger a fresh container build, execute `python -m app.ingest`, and redeploy with zero downtime.
