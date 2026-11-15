
from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from zipfile import BadZipFile, ZipFile

KNOWN_CATEGORIES = (
    "Справочная информация",
    "Знания по энергетике",
    "Продукты",
)

MAX_CHUNK_CHARS = 1800


@dataclass(frozen=True)
class KnowledgeDocument:
    source_path: str
    category: str
    title: str
    text: str


@dataclass(frozen=True)
class KnowledgeChunk:
    chunk_id: str
    source_path: str
    category: str
    title: str
    heading_path: str
    content: str

    @property
    def search_text(self) -> str:
        return "\n".join(
            (
                self.title,
                self.category,
                self.source_path,
                self.heading_path,
                self.content,
            )
        )


def _category_from_path(source_path: str) -> str:
    parts = PurePosixPath(source_path).parts

    for part in parts:
        if part in KNOWN_CATEGORIES:
            return part

    if len(parts) > 1:
        return parts[-2]

    return "Без категории"


def _heading_from_line(line: str) -> tuple[int, str] | None:
    stripped = line.strip()
    marker, separator, heading = stripped.partition(" ")

    if not separator:
        return None

    if not 1 <= len(marker) <= 6:
        return None

    if set(marker) != {"#"}:
        return None

    clean_heading = heading.strip().strip("#").strip()

    if not clean_heading:
        return None

    return len(marker), clean_heading


def _title_from_text(text: str, source_path: str) -> str:
    in_fence = False

    for line in text.splitlines():
        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):
            in_fence = not in_fence
            continue

        if in_fence:
            continue

        heading = _heading_from_line(line)

        if heading and heading[0] == 1:
            return heading[1]

    return PurePosixPath(source_path).stem.replace("_", " ")


def load_markdown_documents(
    archive_path: Path,
) -> list[KnowledgeDocument]:

    if not archive_path.is_file():
        raise FileNotFoundError(
            f"Archive not found: {archive_path}"
        )

    try:
        with ZipFile(archive_path, mode="r") as archive:

            members = sorted(
                (
                    info
                    for info in archive.infolist()
                    if not info.is_dir()
                    and PurePosixPath(info.filename).suffix.lower() == ".md"
                ),
                key=lambda info: info.filename.casefold(),
            )

            documents: list[KnowledgeDocument] = []

            for info in members:

                with archive.open(info, mode="r") as member:
                    raw_text = member.read()

                try:
                    text = raw_text.decode("utf-8-sig")

                except UnicodeDecodeError as exc:
                    raise ValueError(
                        f"Markdown member is not valid UTF-8: {info.filename}"
                    ) from exc

                documents.append(
                    KnowledgeDocument(
                        source_path=info.filename,
                        category=_category_from_path(info.filename),
                        title=_title_from_text(text, info.filename),
                        text=text,
                    )
                )

    except BadZipFile as exc:
        raise ValueError(
            f"Invalid ZIP archive: {archive_path}"
        ) from exc

    return documents


def _without_front_matter(text: str) -> list[str]:

    lines = text.splitlines()

    if not lines or lines[0].strip() != "---":
        return lines

    for index in range(1, len(lines)):

        if lines[index].strip() == "---":
            return lines[index + 1:]

    return lines


def _separator_columns(line: str) -> list[tuple[int, int]]:
    """
    Find column boundaries from a space-aligned Markdown separator.

    Example:
        ----- ----------- -----

    Returns the start and end positions of each dash group.
    """

    matches = list(re.finditer(r"-{3,}", line))

    if len(matches) < 2:
        return []

    return [
        (match.start(), match.end())
        for match in matches
    ]


def _split_table_line(
    line: str,
    columns: list[tuple[int, int]],
) -> list[str]:
    """
    Split a table line using the detected column boundaries.
    """

    cells = []

    for index, (start, _) in enumerate(columns):

        if index + 1 < len(columns):
            next_start = columns[index + 1][0]
            cell = line[start:next_start]
        else:
            cell = line[start:]

        cells.append(cell.strip())

    return cells


def _is_closing_table_separator(line: str) -> bool:
    """
    Detect a closing separator consisting only of spaces and dashes.
    """

    stripped = line.strip()

    return (
        len(stripped) >= 3
        and bool(re.fullmatch(r"-+", stripped))
    )


def _normalise_aligned_tables(lines: list[str]) -> list[str]:
    """
    Convert space-aligned tables into labelled text records.

    The original table format uses:
    - A header row.
    - A dashed separator with multiple column groups.
    - Data rows that may wrap across multiple lines.
    - Blank lines between logical records.

    If a table cannot be parsed safely, its original lines are retained.
    """

    output: list[str] = []
    index = 0

    while index < len(lines):

        columns = _separator_columns(lines[index])

        if not columns or index == 0:
            output.append(lines[index])
            index += 1
            continue

        header_line = lines[index - 1]
        headers = _split_table_line(header_line, columns)

        # Every column must have a usable header.
        if len(headers) != len(columns) or any(
            not header for header in headers
        ):
            output.append(lines[index])
            index += 1
            continue

        # Collect the table body until its closing separator.
        body_start = index + 1
        closing_index = body_start

        while closing_index < len(lines):

            if _is_closing_table_separator(lines[closing_index]):
                break

            closing_index += 1

        if closing_index >= len(lines):
            output.append(lines[index])
            index += 1
            continue

        body_lines = lines[body_start:closing_index]

        records: list[str] = []
        current_cells = [""] * len(headers)

        def flush_record() -> None:
            nonlocal current_cells

            if not any(current_cells):
                return

            fields = []

            for header, value in zip(headers, current_cells):
                if value:
                    fields.append(f"{header}: {value}")

            if fields:
                records.append("\n".join(fields))

            current_cells = [""] * len(headers)

        for body_line in body_lines:

            if not body_line.strip():
                flush_record()
                continue

            cells = _split_table_line(body_line, columns)

            if len(cells) != len(headers):
                continue

            # A non-empty first cell starts a new logical record.
            if cells[0] and current_cells[0]:
                flush_record()

            for cell_index, value in enumerate(cells):

                if value:
                    if current_cells[cell_index]:
                        current_cells[cell_index] += " " + value
                    else:
                        current_cells[cell_index] = value

        flush_record()

        if not records:
            # Preserve the original table if no records were reconstructed.
            output.extend(lines[index - 1:closing_index + 1])
        else:
            output.append("")
            output.extend(records)
            output.append("")

        index = closing_index + 1

    return output


def _split_section_content(
    content: str,
    max_chars: int = MAX_CHUNK_CHARS,
) -> list[str]:
    """
    Split a section into smaller chunks.

    Prefer blank-line boundaries and keep individual blocks together.
    A single block may exceed max_chars if it cannot safely be divided.
    """

    blocks = [
        block.strip()
        for block in content.split("\n\n")
        if block.strip()
    ]

    if not blocks:
        return []

    chunks: list[str] = []
    current_blocks: list[str] = []
    current_length = 0

    for block in blocks:

        block_length = len(block)

        if not current_blocks:
            current_blocks.append(block)
            current_length = block_length
            continue

        proposed_length = current_length + 2 + block_length

        if proposed_length <= max_chars:

            current_blocks.append(block)
            current_length = proposed_length

        else:

            chunks.append("\n\n".join(current_blocks))

            current_blocks = [block]
            current_length = block_length

    if current_blocks:
        chunks.append("\n\n".join(current_blocks))

    return chunks


def chunk_markdown(
    document: KnowledgeDocument,
    document_number: int,
) -> list[KnowledgeChunk]:

    chunks: list[KnowledgeChunk] = []

    heading_stack: list[str] = []
    section_lines: list[str] = []

    section_number = 0
    in_fence = False

    def flush_section() -> None:

        nonlocal section_number

        content = "\n".join(section_lines).strip()
        section_lines.clear()

        if not content:
            return

        heading_path = (
            " > ".join(heading_stack)
            if heading_stack
            else document.title
        )

        content_parts = _split_section_content(content)

        for part in content_parts:

            section_number += 1

            chunks.append(
                KnowledgeChunk(
                    chunk_id=f"{document_number:02d}-{section_number:03d}",
                    source_path=document.source_path,
                    category=document.category,
                    title=document.title,
                    heading_path=heading_path,
                    content=part,
                )
            )

    # First reconstruct supported space-aligned tables.
    processed_lines = _normalise_aligned_tables(
        _without_front_matter(document.text)
    )

    for line in processed_lines:

        stripped = line.strip()

        if stripped.startswith("```") or stripped.startswith("~~~"):

            in_fence = not in_fence
            section_lines.append(line)
            continue

        heading = (
            None
            if in_fence
            else _heading_from_line(line)
        )

        if heading is None:

            section_lines.append(line)
            continue

        flush_section()

        level, heading_text = heading

        del heading_stack[level - 1:]

        heading_stack.append(heading_text)

    flush_section()

    return chunks


def chunk_documents(
    documents: list[KnowledgeDocument],
) -> list[KnowledgeChunk]:

    chunks: list[KnowledgeChunk] = []

    for document_number, document in enumerate(
        documents,
        start=1,
    ):

        chunks.extend(
            chunk_markdown(
                document,
                document_number,
            )
        )

    return chunks
