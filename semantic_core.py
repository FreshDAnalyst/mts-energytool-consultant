
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from organizer_api import OrganizerAPI
from retrieval_core import SearchItem, SearchResult


class SemanticRetriever:
    """
    Retrieves knowledge chunks using semantic similarity.

    Each knowledge chunk is converted into an embedding.
    A user's question is also converted into an embedding.
    Cosine similarity determines which chunks are most relevant.
    """

    def __init__(
        self,
        items: Sequence[SearchItem],
        api: OrganizerAPI,
    ) -> None:

        if not items:
            raise ValueError(
                "Cannot build a semantic retriever without search items."
            )

        self._items = tuple(items)
        self._api = api

        # Convert all knowledge chunks into embeddings.
        texts = [item.search_text for item in self._items]

        embeddings = self._api.embed(texts)

        # Store embeddings as a NumPy matrix.
        self._matrix = np.asarray(embeddings, dtype=np.float32)

        # Verify the embedding matrix.
        if self._matrix.ndim != 2:
            raise ValueError("Invalid embedding matrix dimensions.")

        if self._matrix.shape[0] != len(self._items):
            raise ValueError(
                "Embedding count does not match knowledge chunk count."
            )

        if not np.all(np.isfinite(self._matrix)):
            raise ValueError("Embedding matrix contains invalid values.")

        # Normalize vectors for cosine similarity.
        norms = np.linalg.norm(self._matrix, axis=1, keepdims=True)

        if np.any(norms == 0):
            raise ValueError("Embedding matrix contains zero-length vectors.")

        self._matrix = self._matrix / norms

    def search(
        self,
        query: str,
        top_k: int = 3,
        min_score: float = 0.0,
    ) -> list[SearchResult]:

        clean_query = query.strip()

        if not clean_query:
            raise ValueError("Search query cannot be empty.")

        if top_k <= 0:
            raise ValueError("top_k must be greater than zero.")

        # Convert the question into an embedding.
        query_embedding = self._api.embed(clean_query)

        query_vector = np.asarray(
            query_embedding[0],
            dtype=np.float32,
        )

        if query_vector.ndim != 1:
            raise ValueError("Invalid query embedding dimensions.")

        if query_vector.shape[0] != self._matrix.shape[1]:
            raise ValueError(
                "Query and document embedding dimensions do not match."
            )

        if not np.all(np.isfinite(query_vector)):
            raise ValueError("Query embedding contains invalid values.")

        query_norm = np.linalg.norm(query_vector)

        if query_norm == 0:
            raise ValueError("Query embedding has zero length.")

        query_vector = query_vector / query_norm

        # Calculate cosine similarity.
        scores = self._matrix @ query_vector

        # Rank results from highest similarity to lowest.
        ranked_indices = np.argsort(scores)[::-1]

        results: list[SearchResult] = []

        for index in ranked_indices:
            score = float(scores[index])

            if score < min_score:
                continue

            results.append(
                SearchResult(
                    item=self._items[index],
                    score=score,
                )
            )

            if len(results) >= top_k:
                break

        return results
