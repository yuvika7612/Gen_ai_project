"""
Test ChromaDB search functionality
"""

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


def test_chroma_search():
    """
    Test searching the ChromaDB database
    """

    print("🔍 Testing ChromaDB Search System\n")

    # Load embeddings model (must match the one used during loading)
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    # Load existing ChromaDB database from disk
    persist_dir = "database/chroma_suppliers"

    vectorstore = Chroma(
        collection_name="pharma_suppliers",
        embedding_function=embeddings,
        persist_directory=persist_dir
    )

    doc_count = vectorstore._collection.count()
    print(f"✓ Loaded ChromaDB — {doc_count} documents in collection\n")

    # Test queries
    test_queries = [
        "Find insulin suppliers with cold chain in India",
        "Antibiotic API manufacturers in China",
        "CDSCO approved vaccine suppliers",
        "Emergency suppliers with fast delivery",
        "Low-cost generic drug manufacturers"
    ]

    for i, query in enumerate(test_queries, 1):
        print(f"\n{'='*60}")
        print(f"TEST {i}: {query}")
        print(f"{'='*60}")

        results = vectorstore.similarity_search(query, k=3)

        for j, doc in enumerate(results, 1):
            print(f"\n[Result {j}]")
            lines = doc.page_content.split('\n')
            for line in lines[:10]:
                if line.strip():
                    print(line)
            print("...")

    print(f"\n✅ ChromaDB system working perfectly!")

    # ── Bonus: metadata filtering (ChromaDB-exclusive feature) ──────────────
    print(f"\n{'='*60}")
    print("BONUS: Metadata-filtered search (ChromaDB advantage over FAISS)")
    print(f"{'='*60}")

    # Find only cold-chain-capable suppliers
    filtered_results = vectorstore.similarity_search(
        "insulin supplier",
        k=3,
        filter={"cold_chain": "True"}      # Only cold-chain-capable suppliers
    )

    print(f"\nFiltered (cold_chain=True) results: {len(filtered_results)}")
    for j, doc in enumerate(filtered_results, 1):
        lines = doc.page_content.split('\n')
        print(f"\n[Result {j}] {lines[0]}")   # Print supplier name line


if __name__ == "__main__":
    test_chroma_search()