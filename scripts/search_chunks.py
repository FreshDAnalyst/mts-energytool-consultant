from __future__ import annotations

import argparse
from pathlib import Path

from kb_core import chunk_documents, load_markdown_documents
from questions_core import load_demo_questions
from retrieval_core import TfidfRetriever, compact_excerpt, make_chunk_items


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Search heading-aware Markdown chunks with TF-IDF."
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
    parser.add_argument("--question-number", type=int, default=1)
    parser.add_argument("--top-k", type=int, default=3)
    return parser
def main() -> None:
    args = build_parser().parse_args()
    documents = load_markdown_documents(args.zip)
    chunks = chunk_documents(documents)
    questions = load_demo_questions(args.questions)

    if not 1 <= args.question_number <= len(questions):
        raise SystemExit(
            f"question-number must be between 1 and {len(questions)}"
        )

    question = questions[args.question_number - 1]
    retriever = TfidfRetriever(make_chunk_items(chunks))
    results = retriever.search(question.text, top_k=args.top_k)

    print(f"Question {question.question_id}: {question.text}")
    print(f"Search unit: heading-aware chunks ({len(chunks)} indexed)\n")

    if not results:
        print("No evidence found.")
        return

    for rank, result in enumerate(results, start=1):
        item = result.item
        print(f"{rank}. score={result.score:.4f}")
        print(f"   source={item.source_path}")
        print(f"   category={item.category}")
        print(f"   heading={item.heading_path}")
        print(f"   excerpt={compact_excerpt(item.content)}\n")
if __name__ == "__main__":
    main()