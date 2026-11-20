"""Evaluation Harness for Document Q&A Assistant.

Measures:
1. Retrieval Hit Rate (Answerable questions whose source page was retrieved)
2. Answer Correctness (Keyword containment in generated answers)
3. Correct Refusal Rate (Accurate refusal on unanswerable questions)
4. Average Latency (seconds)
5. Average LLM Calls per question

Logs all experiment runs and parameters into MLflow (./mlruns) and outputs
a summary comparison table and eval/results.json.

Usage:
    python eval/run_eval.py --mode both --strategy all
"""

import argparse
import json
import os
import sys
import time
from typing import Any, Dict, List, Optional

# Enable MLflow local filesystem tracking (./mlruns) without deprecation aborts
os.environ["MLFLOW_ALLOW_FILE_STORE"] = "true"
os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"

# Ensure repository root is on sys.path for direct script execution
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.config import get_settings
from app.vectorstore import get_document_count, retrieve_documents

REFUSAL_PHRASES = [
    "cannot find the answer",
    "not found in the provided documents",
    "not mentioned in the provided",
    "does not contain information",
    "no information",
    "unable to find",
    "not present in the context",
    "i cannot find",
]


def is_refusal(answer: str, steps: List[str]) -> bool:
    """Check if the response represents an honest refusal."""
    if "fallback" in steps:
        return True
    lower = answer.lower()
    return any(phrase in lower for phrase in REFUSAL_PHRASES)


def check_keywords(answer: str, keywords: List[str]) -> bool:
    """Return True if answer matches the expected keywords (case-insensitive)."""
    if not keywords:
        return False
    lower = answer.lower()
    matches = sum(1 for kw in keywords if kw.lower() in lower)
    # Require at least 60% of keywords matched
    return (matches / len(keywords)) >= 0.6


def evaluate_configuration(
    questions: List[Dict[str, Any]],
    mode: str,
    strategy: str,
    has_api_key: bool,
    top_k: int = 4,
) -> Dict[str, Any]:
    """Run evaluation for a single (mode, strategy) configuration."""
    print(f"\n=======================================================")
    print(f"Evaluating: Mode='{mode}' | Strategy='{strategy}'")
    print(f"=======================================================")

    answerable_questions = [q for q in questions if not q.get("expect_refusal", False)]
    unanswerable_questions = [q for q in questions if q.get("expect_refusal", False)]

    # 1. Evaluate Retrieval Hit Rate
    retrieval_hits = 0
    for q in answerable_questions:
        try:
            docs = retrieve_documents(query=q["question"], top_k=top_k, strategy=strategy)
            expected_src = q.get("expected_source")
            expected_page = q.get("expected_page")

            hit = False
            for doc in docs:
                src_match = (not expected_src) or (expected_src in doc.metadata.get("source", ""))
                page_match = (not expected_page) or (doc.metadata.get("page") == expected_page)
                if src_match and page_match:
                    hit = True
                    break
            if hit:
                retrieval_hits += 1
        except Exception as e:
            print(f"Retrieval error for '{q['id']}': {e}", file=sys.stderr)

    retrieval_hit_rate = round(retrieval_hits / len(answerable_questions), 4) if answerable_questions else 0.0

    if not has_api_key:
        print("[Notice] No LLM API key configured. Generation metrics marked 'not yet run'.")
        return {
            "mode": mode,
            "strategy": strategy,
            "status": "not yet run (requires OPENAI_API_KEY for generation)",
            "retrieval_hit_rate": retrieval_hit_rate,
            "answer_correctness": "not yet run",
            "refusal_rate": "not yet run",
            "avg_latency_seconds": "not yet run",
            "avg_llm_calls": "not yet run",
            "total_questions": len(questions),
        }

    # 2. Evaluate Generation with Live LLM
    from app.chains import run_simple_rag
    from app.graph import run_graph_rag

    correct_answers = 0
    correct_refusals = 0
    latencies = []
    llm_calls_list = []

    for q in questions:
        q_text = q["question"]
        is_unanswerable = q.get("expect_refusal", False)

        t0 = time.perf_counter()
        if mode == "simple":
            result = run_simple_rag(question=q_text, top_k=top_k, strategy=strategy)
            calls = 1 if result.get("sources") else 0
        else:
            result = run_graph_rag(question=q_text, top_k=top_k, strategy=strategy)
            steps = result.get("steps", [])
            llm_nodes = {"grade_documents", "rewrite_query", "generate"}
            calls = sum(1 for s in steps if s in llm_nodes)

        elapsed = time.perf_counter() - t0
        latencies.append(elapsed)
        llm_calls_list.append(calls)

        ans = result.get("answer", "")
        steps = result.get("steps", [])

        if is_unanswerable:
            if is_refusal(ans, steps):
                correct_refusals += 1
        else:
            if check_keywords(ans, q.get("expected_keywords", [])):
                correct_answers += 1

    answer_correctness = round(correct_answers / len(answerable_questions), 4) if answerable_questions else 0.0
    refusal_rate = round(correct_refusals / len(unanswerable_questions), 4) if unanswerable_questions else 0.0
    avg_latency = round(sum(latencies) / len(latencies), 3) if latencies else 0.0
    avg_llm_calls = round(sum(llm_calls_list) / len(llm_calls_list), 2) if llm_calls_list else 0.0

    return {
        "mode": mode,
        "strategy": strategy,
        "status": "completed",
        "retrieval_hit_rate": retrieval_hit_rate,
        "answer_correctness": answer_correctness,
        "refusal_rate": refusal_rate,
        "avg_latency_seconds": avg_latency,
        "avg_llm_calls": avg_llm_calls,
        "total_questions": len(questions),
    }


def log_to_mlflow(run_data: Dict[str, Any], settings, results_file: str):
    """Log parameters, metrics, and artifact to a local MLflow run."""
    try:
        import mlflow

        mlflow.set_tracking_uri(settings.mlflow_tracking_uri)
        mlflow.set_experiment(settings.mlflow_experiment_name)

        run_name = f"{run_data['mode']}_{run_data['strategy']}"
        with mlflow.start_run(run_name=run_name):
            # Parameters
            mlflow.log_params({
                "mode": run_data["mode"],
                "strategy": run_data["strategy"],
                "chunk_size": settings.chunk_size,
                "chunk_overlap": settings.chunk_overlap,
                "top_k": settings.top_k,
                "model_name": settings.openai_model,
                "max_rewrites": settings.max_rewrites,
            })

            # Metrics
            mlflow.log_metric("retrieval_hit_rate", run_data["retrieval_hit_rate"])
            if run_data.get("status") == "completed":
                mlflow.log_metric("answer_correctness", run_data["answer_correctness"])
                mlflow.log_metric("refusal_rate", run_data["refusal_rate"])
                mlflow.log_metric("avg_latency_seconds", run_data["avg_latency_seconds"])
                mlflow.log_metric("avg_llm_calls", run_data["avg_llm_calls"])

            if os.path.exists(results_file):
                mlflow.log_artifact(results_file)

        print(f"[MLflow] Logged run '{run_name}' to {settings.mlflow_tracking_uri}")
    except Exception as e:
        print(f"[Warning] Failed to log run to MLflow: {e}", file=sys.stderr)


def print_comparison_table(results: List[Dict[str, Any]]):
    """Render a clean GitHub-flavored markdown / ASCII comparison table."""
    header = (
        f"| {'Mode':<8} | {'Strategy':<14} | {'Hit Rate':<10} | {'Correctness':<13} | "
        f"{'Refusal':<10} | {'Avg Latency (s)':<17} | {'Avg LLM Calls':<15} |"
    )
    separator = (
        f"|{'-'*10}|{'-'*16}|{'-'*12}|{'-'*15}|{'-'*12}|{'-'*19}|{'-'*17}|"
    )

    print("\n" + "=" * 95)
    print("EVALUATION RESULTS COMPARISON TABLE")
    print("=" * 95)
    print(header)
    print(separator)

    for r in results:
        hit = f"{r['retrieval_hit_rate'] * 100:.1f}%" if isinstance(r['retrieval_hit_rate'], (int, float)) else str(r['retrieval_hit_rate'])
        corr = f"{r['answer_correctness'] * 100:.1f}%" if isinstance(r['answer_correctness'], (int, float)) else str(r['answer_correctness'])
        ref = f"{r['refusal_rate'] * 100:.1f}%" if isinstance(r['refusal_rate'], (int, float)) else str(r['refusal_rate'])
        lat = f"{r['avg_latency_seconds']:.2f}s" if isinstance(r['avg_latency_seconds'], (int, float)) else str(r['avg_latency_seconds'])
        calls = f"{r['avg_llm_calls']:.1f}" if isinstance(r['avg_llm_calls'], (int, float)) else str(r['avg_llm_calls'])

        row = (
            f"| {r['mode']:<8} | {r['strategy']:<14} | {hit:<10} | {corr:<13} | "
            f"{ref:<10} | {lat:<17} | {calls:<15} |"
        )
        print(row)
    print("=" * 95 + "\n")


def main():
    parser = argparse.ArgumentParser(description="Run Document Q&A Evaluation Suite.")
    parser.add_argument("--mode", choices=["simple", "graph", "both"], default="both", help="Pipeline mode to evaluate")
    parser.add_argument("--strategy", choices=["dense", "hybrid", "hybrid_rerank", "all"], default="all", help="Retrieval strategy")
    parser.add_argument("--questions", default="eval/questions.json", help="Path to questions dataset")
    parser.add_argument("--output", default="eval/results.json", help="Path to save results JSON")
    args = parser.parse_args()

    settings = get_settings()

    # Verify vector store is populated
    doc_count = get_document_count()
    if doc_count == 0:
        print("[Error] Vector store is empty! Run ingestion first: python -m app.ingest", file=sys.stderr)
        sys.exit(1)

    if not os.path.exists(args.questions):
        print(f"[Error] Questions file not found at: {args.questions}", file=sys.stderr)
        sys.exit(1)

    with open(args.questions, "r", encoding="utf-8") as f:
        questions = json.load(f)

    modes = ["simple", "graph"] if args.mode == "both" else [args.mode]
    strategies = ["dense", "hybrid", "hybrid_rerank"] if args.strategy == "all" else [args.strategy]

    has_api_key = bool(settings.openai_api_key and settings.openai_api_key.strip())
    if not has_api_key:
        print(
            "\n[Notice] No OPENAI_API_KEY detected in environment or .env file.\n"
            "Retrieval hit rates will be evaluated locally with FastEmbed/BM25.\n"
            "Generation metrics (correctness, refusal, latency) will be marked 'not yet run'.\n"
            "To evaluate full end-to-end generation, configure OPENAI_API_KEY in .env.\n"
        )

    all_results = []
    for mode in modes:
        for strategy in strategies:
            res = evaluate_configuration(
                questions=questions,
                mode=mode,
                strategy=strategy,
                has_api_key=has_api_key,
                top_k=settings.top_k,
            )
            all_results.append(res)

    # Save results to JSON
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    print(f"\n[Saved] Detailed results written to {args.output}")

    # Log to MLflow
    for res in all_results:
        log_to_mlflow(res, settings, args.output)

    # Print summary table
    print_comparison_table(all_results)


if __name__ == "__main__":
    main()
