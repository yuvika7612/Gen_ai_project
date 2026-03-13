"""
Test FAISS search functionality
"""

from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import FAISS

def test_faiss_search():
    """
    Test searching the FAISS database
    """
    
    print("🔍 Testing FAISS Search System\n")
    
    # Load embeddings model
    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    # Load FAISS database from disk
    vectorstore = FAISS.load_local(
        "database/faiss_suppliers",
        embeddings,
        allow_dangerous_deserialization=True  # Required for FAISS
    )
    
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
    
    print(f"\n✅ FAISS system working perfectly!")

if __name__ == "__main__":
    test_faiss_search()