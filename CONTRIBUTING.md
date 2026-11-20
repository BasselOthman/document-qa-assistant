# Contributing to Document Q&A Assistant

Thank you for your interest in contributing to the Document Q&A Assistant project! This document outlines guidelines and development practices for contributors.

---

## 1. Local Development Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/BasselOthman/document-qa-assistant.git
   cd document-qa-assistant
   ```

2. **Create and activate a virtual environment:**
   ```bash
   python -m venv .venv
   # Windows PowerShell
   .venv\Scripts\Activate.ps1
   # Linux / macOS
   source .venv/bin/activate
   ```

3. **Install dependencies:**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Configure environment variables:**
   ```bash
   cp .env.example .env
   ```
   Add your OpenAI or Groq API key in `.env` if testing live LLM generation.

---

## 2. Document Ingestion

Whenever adding new test PDF documents to `data/`, run ingestion to update the persistent Chroma database and serialized BM25 index:

```bash
python -m app.ingest
```

The ingestion process computes deterministic SHA-256 chunk IDs to guarantee zero duplicate vectors.

---

## 3. Running Tests & Benchmarks

Before opening a pull request, verify that all automated tests pass:

```bash
pytest -v
```

To run the complete evaluation benchmark across all retrieval strategies and modes:

```bash
python eval/run_eval.py --mode both --strategy all
```

---

## 4. Code Style & Commit Conventions

- Follow PEP 8 guidelines.
- Use Conventional Commits for all commit messages:
  - `feat:` for new capabilities or graph nodes.
  - `fix:` for bug fixes.
  - `docs:` for documentation updates.
  - `test:` for new or modified test suites.
  - `refactor:` for code restructurings without behavior changes.
