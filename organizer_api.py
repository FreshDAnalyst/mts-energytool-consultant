
from __future__ import annotations

import math
import os
import time
from collections.abc import Sequence
from typing import Any

import httpx


class OrganizerAPI:
    """
    Local AI client using Ollama.

    Chat model: gemma3:1b
    Embedding model: nomic-embed-text
    """

    def __init__(self) -> None:

        # Ollama connection
        self.base_url = (
            os.getenv(
                "OLLAMA_BASE_URL",
                "http://localhost:11434",
            )
        ).strip().rstrip("/")

        # Local models
        self.llm_model = os.getenv(
            "LLM_MODEL",
            "gemma3:1b",
        ).strip()

        self.embedding_model = os.getenv(
            "EMB_MODEL",
            "nomic-embed-text",
        ).strip()

        # Generation settings
        self.temperature = float(
            os.getenv("TEMPERATURE", "0.0")
        )

        self.max_tokens = int(
            os.getenv("MAX_TOKENS", "2048")
        )

        # Increased timeout for local model processing
        self.timeout = 300

        # Validate configuration
        if not self.base_url:
            raise ValueError(
                "Ollama base URL is missing."
            )

        if not self.llm_model:
            raise ValueError(
                "LLM_MODEL is not configured."
            )

        if not self.embedding_model:
            raise ValueError(
                "EMB_MODEL is not configured."
            )

        if self.temperature < 0:
            raise ValueError(
                "TEMPERATURE cannot be negative."
            )

        if self.max_tokens <= 0:
            raise ValueError(
                "MAX_TOKENS must be greater than zero."
            )

    def _post(
        self,
        endpoint: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Send a request to the local Ollama API with one retry."""

        for attempt in range(2):

            try:
                with httpx.Client(
                    timeout=self.timeout
                ) as client:

                    response = client.post(
                        f"{self.base_url}{endpoint}",
                        json=payload,
                    )

                    response.raise_for_status()

                    result = response.json()

                    if not isinstance(result, dict):
                        raise ValueError(
                            f"Invalid response from {endpoint}."
                        )

                    return result

            except (
                httpx.TimeoutException,
                httpx.TransportError,
                httpx.HTTPStatusError,
            ):

                if attempt == 1:
                    raise

                time.sleep(1)

        raise RuntimeError(
            f"Request to Ollama endpoint {endpoint} failed."
        )

    def embed(
        self,
        texts: str | Sequence[str],
    ) -> list[list[float]]:
        """
        Convert text into normalized embedding vectors.

        Processes inputs in batches to avoid overloading
        the local Ollama embedding model.
        """

        if isinstance(texts, str):
            texts = [texts]
        else:
            texts = list(texts)

        if not texts or any(
            not isinstance(text, str) or not text.strip()
            for text in texts
        ):
            raise ValueError(
                "Embedding inputs must be non-empty text."
            )

        batch_size = 16

        all_vectors: list[list[float]] = []

        total_texts = len(texts)

        for start in range(
            0,
            total_texts,
            batch_size,
        ):

            batch = texts[
                start:start + batch_size
            ]

            result = self._post(
                "/api/embed",
                {
                    "model": self.embedding_model,
                    "input": batch,
                },
            )

            raw_embeddings = result.get(
                "embeddings"
            )

            if (
                not isinstance(raw_embeddings, list)
                or len(raw_embeddings) != len(batch)
            ):
                raise ValueError(
                    "Invalid embeddings response from Ollama."
                )

            for raw_vector in raw_embeddings:

                if (
                    not isinstance(raw_vector, list)
                    or not raw_vector
                ):
                    raise ValueError(
                        "Embedding vector is missing or empty."
                    )

                vector = [
                    float(value)
                    for value in raw_vector
                ]

                if not all(
                    math.isfinite(value)
                    for value in vector
                ):
                    raise ValueError(
                        "Embedding contains invalid numbers."
                    )

                norm = math.sqrt(
                    sum(
                        value * value
                        for value in vector
                    )
                )

                if norm == 0:
                    raise ValueError(
                        "Received a zero-length embedding."
                    )

                normalized_vector = [
                    value / norm
                    for value in vector
                ]

                all_vectors.append(
                    normalized_vector
                )

            print(
                f"Embedded {len(all_vectors)} "
                f"of {total_texts} chunks..."
            )

        # Validate embedding dimensions
        dimensions = {
            len(vector)
            for vector in all_vectors
        }

        if len(dimensions) != 1:
            raise ValueError(
                "Embedding dimensions are inconsistent."
            )

        return all_vectors

    def rerank(
        self,
        query: str,
        documents: Sequence[str],
        top_n: int = 3,
    ) -> list[dict[str, Any]]:
        """
        Reranking is not configured in this local Ollama setup.
        The current hybrid retrieval system does not require it.
        """

        raise NotImplementedError(
            "Reranking is not configured. "
            "The MTS application currently uses hybrid retrieval."
        )

    def generate(
        self,
        messages: Sequence[dict[str, str]],
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> str:
        """Generate an answer using the local Ollama chat model."""

        if not messages:
            raise ValueError(
                "Messages cannot be empty."
            )

        selected_temperature = (
            self.temperature
            if temperature is None
            else temperature
        )

        selected_max_tokens = (
            self.max_tokens
            if max_tokens is None
            else max_tokens
        )

        result = self._post(
            "/api/chat",
            {
                "model": self.llm_model,
                "messages": list(messages),
                "stream": False,
                "options": {
                    "temperature": selected_temperature,
                    "num_predict": selected_max_tokens,
                },
            },
        )

        message = result.get(
            "message",
            {},
        )

        if not isinstance(message, dict):
            raise ValueError(
                "Invalid chat response from Ollama."
            )

        answer = message.get(
            "content",
            "",
        )

        if (
            not isinstance(answer, str)
            or not answer.strip()
        ):
            raise ValueError(
                "The local language model returned an empty answer."
            )

        return answer.strip()
