from __future__ import annotations

import argparse
import csv
from pathlib import Path

from kb_core import chunk_documents, load_markdown_documents
from questions_core import load_demo_questions
from retrieval_core import TfidfRetriever, make_chunk_items


PROBE_QUERY = "несуществующийтерминxyz"
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Create a retrieval report for every demo question."
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("local_data/knowledge-base.zip"),
    )
    parser.add_argument(
        "--questions",
        type=Path,
        default=Path("local_data/demo_questions.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("runtime_data/retrieval_report.csv"),
    )
    parser.add_argument("--top-k", type=int, default=3)
    return parser
def main() -> None:
    args = build_parser().parse_args()
    documents = load_markdown_documents(args.zip)
    chunks = chunk_documents(documents)
    questions = load_demo_questions(args.questions)
    retriever = TfidfRetriever(make_chunk_items(chunks))

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = (
        "id",
        "text",
        "top_source",
        "top_heading",
        "top_score",
        "candidate_count",
    )

    questions_with_candidates = 0
    with args.output.open(
        "w", encoding="utf-8-sig", newline=""
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for question in questions:
            results = retriever.search(question.text, top_k=args.top_k)
            top_result = results[0] if results else None
            if results:
                questions_with_candidates += 1

            writer.writerow(
                {
                    "id": question.question_id,
                    "text": question.text,
                    "top_source": (
                        top_result.item.source_path if top_result else ""
                    ),
                    "top_heading": (
                        top_result.item.heading_path if top_result else ""
                    ),
                    "top_score": (
                        f"{top_result.score:.4f}" if top_result else ""
                    ),
                    "candidate_count": len(results),
                }
            )
            print(
                f"{question.question_id}: candidates={len(results)} "
                f"top_score={top_result.score:.4f}"
                if top_result
                else f"{question.question_id}: candidates=0"
            )
    probe_results = retriever.search(PROBE_QUERY, top_k=args.top_k)
    print(
        f"\nQuestions with candidates: "
        f"{questions_with_candidates}/{len(questions)}"
    )
    print(f"Report: {args.output}")
    print(f"Out-of-vocabulary probe results: {len(probe_results)}")

    if probe_results:
        raise SystemExit(
            "VALIDATION FAILED: the out-of-vocabulary probe returned evidence"
        )
    print("VALIDATION PASSED")


if __name__ == "__main__":
    main()