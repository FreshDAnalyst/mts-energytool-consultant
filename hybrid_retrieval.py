
from __future__ import annotations

from collections.abc import Sequence

from organizer_api import OrganizerAPI
from retrieval_core import (
    SearchItem,
    SearchResult,
    TfidfRetriever,
)
from semantic_core import SemanticRetriever


class HybridRetriever:
    """
    Combines lexical (TF-IDF) and semantic retrieval
    using Reciprocal Rank Fusion (RRF).
    """

    def __init__(
        self,
        items: Sequence[SearchItem],
        api: OrganizerAPI,
        rrf_k: int = 60,
        candidate_k: int = 10,
    ) -> None:

        if not items:
            raise ValueError("HybridRetriever requires at least one item.")

        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than zero.")

        if candidate_k <= 0:
            raise ValueError("candidate_k must be greater than zero.")

        self._items = list(items)
        self._rrf_k = rrf_k
        self._candidate_k = candidate_k

        self._lexical = TfidfRetriever(self._items)

        self._semantic = SemanticRetriever(
            items=self._items,
            api=api,
        )

    def search(
        self,
        query: str,
        top_k: int = 5,
    ) -> list[SearchResult]:

        if not query or not query.strip():
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        lexical_results = self._lexical.search(
            query,
            top_k=self._candidate_k,
        )

        semantic_results = self._semantic.search(
            query,
            top_k=self._candidate_k,
        )

        # Reciprocal Rank Fusion
        scores: dict[str, float] = {}
        items_by_id: dict[str, SearchItem] = {}

        for rank, result in enumerate(lexical_results, start=1):
            item_id = result.item.item_id
            items_by_id[item_id] = result.item
            scores[item_id] = (
                scores.get(item_id, 0.0)
                + 1.0 / (self._rrf_k + rank)
            )

        for rank, result in enumerate(semantic_results, start=1):
            item_id = result.item.item_id
            items_by_id[item_id] = result.item
            scores[item_id] = (
                scores.get(item_id, 0.0)
                + 1.0 / (self._rrf_k + rank)
            )

        ranked_items = sorted(
            scores.items(),
            key=lambda pair: pair[1],
            reverse=True,
        )

        return [
            SearchResult(
                item=items_by_id[item_id],
                score=score,
            )
            for item_id, score in ranked_items[:top_k]
        ]
