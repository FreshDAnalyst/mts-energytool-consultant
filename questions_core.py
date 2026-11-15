from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DemoQuestion:
    question_id: str
    text: str
def load_demo_questions(csv_path: Path) -> list[DemoQuestion]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Questions file not found: {csv_path}")

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = set(reader.fieldnames or [])
        required = {"id", "text"}
        missing = required - fieldnames
        if missing:
            missing_list = ", ".join(sorted(missing))
            raise ValueError(f"Questions CSV is missing columns: {missing_list}")

        questions: list[DemoQuestion] = []
        for row_number, row in enumerate(reader, start=2):
            question_id = (row.get("id") or "").strip()
            text = (row.get("text") or "").strip()
            if not question_id or not text:
                raise ValueError(
                    f"Questions CSV row {row_number} has an empty id or text"
                )
            questions.append(DemoQuestion(question_id=question_id, text=text))

    if not questions:
        raise ValueError("Questions CSV contains no data rows")
    return questions   