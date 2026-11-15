from __future__ import annotations

import argparse
from collections import Counter
from pathlib import Path

from kb_core import load_markdown_documents


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Inspect and validate the original MTS knowledge archive."
    )
    parser.add_argument(
        "--zip",
        type=Path,
        default=Path("local_data/knowledge-base.zip"),
        help="Path to the original knowledge-base.zip file.",
    )
    parser.add_argument(
        "--expected-count",
        type=int,
        default=19,
        help="Expected number of Markdown documents.",
    )
    return parser
def main() -> None:
    args = build_parser().parse_args()
    documents = load_markdown_documents(args.zip)

    print(f"Archive: {args.zip}")
    print(f"Markdown documents: {len(documents)}")
    print("\nCategories:")
    category_counts = Counter(document.category for document in documents)
    for category, count in sorted(category_counts.items()):
        print(f"  {category}: {count}")

    print("\nMarkdown members:")
    for document in documents:
        print(f"  [{document.category}] {document.source_path}")

    if len(documents) != args.expected_count:
        raise SystemExit(
            "VALIDATION FAILED: "
            f"expected {args.expected_count} Markdown files, "
            f"found {len(documents)}"
        )

    print("\nVALIDATION PASSED")


if __name__ == "__main__":
    main()