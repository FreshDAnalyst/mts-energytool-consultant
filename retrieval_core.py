from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from kb_core import KnowledgeDocument


@dataclass(frozen=True)
class SearchItem:
    item_id: str
    source_path: str
    category: str
    title: str
    heading_path: str
    content: str
    search_text: str


@dataclass(frozen=True)
class SearchResult:
    item: SearchItem
    score: float
class TfidfRetriever:
    def __init__(self, items: Sequence[SearchItem]) -> None:
        if not items:
            raise ValueError("Cannot build a retriever without search items")

        self._items = tuple(items)
        self._vectorizer = TfidfVectorizer(
            lowercase=True,
            ngram_range=(1, 2),
            sublinear_tf=True,
        )
        self._matrix = self._vectorizer.fit_transform(
            item.search_text for item in self._items
        )
    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.02,
    ) -> list[SearchResult]:
        clean_query = query.strip()
        if not clean_query:
            raise ValueError("Search query cannot be empty")
        if top_k <= 0:
            raise ValueError("top_k must be greater than zero")

        query_vector = self._vectorizer.transform([clean_query])
        if query_vector.nnz == 0:
            return []

        scores = cosine_similarity(query_vector, self._matrix)[0]
        ranked_indices = scores.argsort()[::-1]

        results: list[SearchResult] = []
        for index in ranked_indices:
            score = float(scores[index])
            if score < min_score:
                continue
            results.append(SearchResult(item=self._items[index], score=score))
            if len(results) == top_k:
                break
        return results
def make_document_items(
    documents: Sequence[KnowledgeDocument],
) -> list[SearchItem]:
    items: list[SearchItem] = []
    for number, document in enumerate(documents, start=1):
        search_text = "\n".join(
            (
                document.title,
                document.category,
                document.source_path,
                document.text,
            )
        )
        items.append(
            SearchItem(
                item_id=f"document-{number:02d}",
                source_path=document.source_path,
                category=document.category,
                title=document.title,
                heading_path="(whole document)",
                content=document.text,
                search_text=search_text,
            )
        )
    return items
def make_chunk_items(chunks: Sequence[object]) -> list[SearchItem]:
    items: list[SearchItem] = []
    for chunk in chunks:
        items.append(
            SearchItem(
                item_id=chunk.chunk_id,
                source_path=chunk.source_path,
                category=chunk.category,
                title=chunk.title,
                heading_path=chunk.heading_path,
                content=chunk.content,
                search_text=chunk.search_text,
            )
        )
    return items


def compact_excerpt(text: str, limit: int = 240) -> str:
    compact = " ".join(text.split())
    if len(compact) <= limit:
        return compact
    return compact[: limit - 1].rstrip() + "…"