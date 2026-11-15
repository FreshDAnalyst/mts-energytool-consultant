
from __future__ import annotations

from semantic_core import SemanticRetriever
from retrieval_core import SearchItem


class FakeOrganizerAPI:
    """
    Simulates the embedding API for local testing.

    These are artificial vectors, not real AI embeddings.
    """

    def embed(self, texts):
        if isinstance(texts, str):
            texts = [texts]

        vectors = []

        for text in texts:
            clean_text = text.lower()

            # Electricity-related content
            if any(word in clean_text for word in [
                "electricity",
                "electric",
                "costs",
                "consumption",
                "optimization",
            ]):
                vector = [1.0, 0.0, 0.0]

            # Solar-related content
            elif any(word in clean_text for word in [
                "solar",
                "sunlight",
                "sun",
            ]):
                vector = [0.4, 0.8, 0.0]

            # Battery-related content
            elif any(word in clean_text for word in [
                "battery",
                "storage",
            ]):
                vector = [0.2, 0.0, 0.8]

            # General content
            else:
                vector = [0.5, 0.5, 0.0]

            vectors.append(vector)

        return vectors


def main():
    print("=" * 50)
    print("LOCAL SEMANTIC RETRIEVAL TEST")
    print("=" * 50)

    # 1. Create sample knowledge chunks
    items = [
        SearchItem(
            item_id="chunk_1",
            source_path="energy.md",
            category="Energy",
            title="Electricity optimization",
            heading_path="Energy > Optimization",
            content="Methods for reducing electricity consumption.",
            search_text=(
                "Electricity energy consumption optimization."
            ),
        ),

        SearchItem(
            item_id="chunk_2",
            source_path="solar.md",
            category="Products",
            title="Solar energy",
            heading_path="Products > Solar",
            content="Information about solar energy systems.",
            search_text="Solar energy systems and sunlight.",
        ),

        SearchItem(
            item_id="chunk_3",
            source_path="battery.md",
            category="Products",
            title="Battery storage",
            heading_path="Products > Storage",
            content="Information about battery energy storage.",
            search_text="Battery storage systems.",
        ),
    ]

    print("\nKnowledge chunks created:", len(items))

    # 2. Initialize simulated API
    fake_api = FakeOrganizerAPI()

    # 3. Build semantic retriever
    retriever = SemanticRetriever(
        items=items,
        api=fake_api,
    )

    print("Knowledge chunks indexed:", len(items))

    # 4. Search for relevant information
    query = "How can a company reduce electricity costs?"

    print("\nQuestion:", query)
    print("\nSearch results:")

    results = retriever.search(
        query=query,
        top_k=3,
    )

    # 5. Display search results
    for rank, result in enumerate(results, start=1):
        print(f"\nRank: {rank}")
        print("ID:", result.item.item_id)
        print("Title:", result.item.title)
        print("Category:", result.item.category)
        print("Source:", result.item.source_path)
        print("Similarity:", round(result.score, 4))
        print("Content:", result.item.content)

    # 6. Validate retrieval results
    assert len(results) == 3, (
        "Expected exactly three search results."
    )

    assert results[0].item.item_id == "chunk_1", (
        "Electricity optimization should rank first."
    )

    assert results[1].item.item_id == "chunk_2", (
        "Solar energy should rank second."
    )

    assert results[2].item.item_id == "chunk_3", (
        "Battery storage should rank third."
    )

    print("\n" + "=" * 50)
    print("LOCAL SEMANTIC TEST PASSED!")
    print("=" * 50)


if __name__ == "__main__":
    main()
