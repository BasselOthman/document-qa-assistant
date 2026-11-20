---
name: Bug Report
about: Create a report to help improve the Document Q&A Assistant
title: '[BUG] '
labels: bug
assignees: ''
---

### Bug Description
A clear and concise description of what the bug is.

### Reproduction Steps
1. Ingest documents: `python -m app.ingest`
2. Run server or evaluation: `...`
3. Request payload sent to `/ask`:
   ```json
   {
     "question": "...",
     "mode": "graph",
     "strategy": "hybrid"
   }
   ```
4. Observe error or unexpected behavior.

### Expected Behavior
A concise description of what you expected to happen.

### Execution Telemetry / Logs
If applicable, paste the structured JSON log line or terminal output:
```json
{}
```

### Environment Details
- OS: [e.g. Windows 11, Ubuntu 22.04, macOS]
- Python Version: [e.g. 3.11.8]
- Model Provider: [e.g. OpenAI, Groq, Ollama]
