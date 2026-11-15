
from app.main import initialize_ai_engine, retriever

def inspect_results(label, results):
    print(f"\n{'=' * 70}")
    print(label)
    print("=" * 70)

    if not results:
        print("No results found.")
        return

    for rank, result in enumerate(results, start=1):
        item = result.item

        print(f"\nRank: {rank}")
        print(f"Score: {result.score:.6f}")
        print(f"Source: {item.source_path}")
        print(f"Title: {item.title}")
        print(f"Heading: {item.heading_path}")
        print(f"Content: {item.content[:350]}")
        print("-" * 70)

def main():
    global retriever

    print("Initializing MTS retrieval engine...")
    initialize_ai_engine()

    question = "What is MTS EnergyTool?"

    print(f"\nDiagnostic question: {question}")

    print("\nRunning hybrid retrieval...")
    results = retriever.search(
        question,
        top_k=10,
    )

    inspect_results("HYBRID RETRIEVAL RESULTS", results)

if __name__ == "__main__":
    main()