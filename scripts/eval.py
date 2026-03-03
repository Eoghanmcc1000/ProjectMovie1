from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Any

import httpx


def _extract_bolded_titles(text: str) -> list[str]:
    return [t.strip() for t in re.findall(r"\*\*(.*?)\*\*", text or "") if t.strip()]


def _precision_recall(predicted: set[int], expected: set[int]) -> tuple[float, float]:
    if not expected and not predicted:
        return 1.0, 1.0
    if not predicted:
        return 0.0, 0.0
    tp = len(predicted & expected)
    precision = tp / len(predicted) if predicted else 0.0
    recall = tp / len(expected) if expected else 1.0
    return precision, recall


def run_eval(cases_path: Path, base_url: str, timeout_s: float) -> dict[str, Any]:
    cases = json.loads(cases_path.read_text(encoding="utf-8"))
    if not isinstance(cases, list):
        raise ValueError("Cases file must be a JSON array")

    strategy_hits = 0
    strategy_total = 0
    min_results_hits = 0
    min_results_total = 0
    precision_vals: list[float] = []
    recall_vals: list[float] = []
    hallucination_hits = 0
    reasoning_hits = 0
    latencies: list[float] = []

    with httpx.Client(timeout=timeout_s) as client:
        for i, case in enumerate(cases, start=1):
            query = case["query"]
            payload = {"message": query, "session_id": f"eval-{i}"}
            resp = client.post(f"{base_url}/chat", json=payload)
            resp.raise_for_status()
            data = resp.json()

            strategy = data["metadata"]["retrieval_strategy"]
            results_found = data["metadata"]["results_found"]
            response_text = data["response"]
            movies = data.get("movies", [])
            returned_titles = {m["title"] for m in movies}
            returned_ids = {m["id"] for m in movies}
            latencies.append(float(data["metadata"].get("retrieval_time_ms", 0.0)))
            if data["metadata"].get("reasoning"):
                reasoning_hits += 1

            expected_strategy = case.get("expected_strategy")
            if expected_strategy:
                strategy_total += 1
                if strategy == expected_strategy:
                    strategy_hits += 1

            expected_min_results = case.get("expected_min_results")
            if expected_min_results is not None:
                min_results_total += 1
                if results_found >= int(expected_min_results):
                    min_results_hits += 1

            expected_movie_ids = case.get("expected_movie_ids")
            if isinstance(expected_movie_ids, list):
                p, r = _precision_recall(returned_ids, set(expected_movie_ids))
                precision_vals.append(p)
                recall_vals.append(r)

            bolded_titles = _extract_bolded_titles(response_text)
            hallucinated = [t for t in bolded_titles if t not in returned_titles]
            if hallucinated:
                hallucination_hits += 1

            print(
                json.dumps(
                    {
                        "query": query,
                        "strategy": strategy,
                        "results_found": results_found,
                        "hallucinated_titles": hallucinated,
                    },
                    ensure_ascii=True,
                )
            )

    total = len(cases)
    summary = {
        "total_cases": total,
        "strategy_accuracy": round(strategy_hits / strategy_total, 3)
        if strategy_total
        else None,
        "min_results_accuracy": round(min_results_hits / min_results_total, 3)
        if min_results_total
        else None,
        "avg_id_precision": round(sum(precision_vals) / len(precision_vals), 3)
        if precision_vals
        else None,
        "avg_id_recall": round(sum(recall_vals) / len(recall_vals), 3)
        if recall_vals
        else None,
        "hallucination_rate": round(hallucination_hits / total, 3) if total else None,
        "reasoning_rate": round(reasoning_hits / total, 3) if total else None,
        "avg_retrieval_time_ms": round(sum(latencies) / len(latencies), 2)
        if latencies
        else None,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Offline evaluation harness for /chat")
    parser.add_argument("--cases", default="data/eval_queries.json")
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--timeout", type=float, default=30.0)
    args = parser.parse_args()

    summary = run_eval(Path(args.cases), args.base_url.rstrip("/"), args.timeout)
    print(json.dumps({"summary": summary}, indent=2, ensure_ascii=True))


if __name__ == "__main__":
    main()
