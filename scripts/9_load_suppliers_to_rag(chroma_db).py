"""
Load pharmaceutical suppliers into ChromaDB vector database
ChromaDB = Persistent, no server needed, easy setup!
"""

import pandas as pd
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.docstore.document import Document
import os

def load_suppliers_to_chroma():
    """
    Load supplier CSV into ChromaDB vector database
    """

    print("📥 Loading suppliers into ChromaDB database...\n")

    # Load supplier CSV
    csv_path = 'data/suppliers/pharma_suppliers.csv'

    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found")
        return

    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} suppliers from CSV")

    # Convert each supplier to a document
    documents = []

    for _, row in df.iterrows():
        text = f"""
Supplier: {row['company_name']}
ID: {row['supplier_id']}
Location: {row['city']}, {row['country']}

Product Category: {row['product_category']}
Current Stock: {row['current_stock']} {row['unit']}
Unit Price: ₹{row['unit_price_inr']}
Lead Time: {row['lead_time_days']} days
Minimum Order: {row['minimum_order_quantity']} {row['unit']}

Cold Chain Capable: {'Yes' if row['cold_chain_capable'] else 'No'}
Temperature Range: {row['temperature_range']}

Quality & Compliance:
- Certifications: {row['quality_certifications']}
- CDSCO Approved: {'Yes' if row['cdsco_approved'] else 'No'}
- GMP Certified: {'Yes' if row['gmp_certified'] else 'No'}
- Reliability Score: {row['reliability_score']}%
- On-Time Delivery: {row['on_time_delivery_rate']}%

Payment Terms: {row['payment_terms']}
Credit Rating: {row['credit_rating']}

Contact: {row['contact_email']}
        """.strip()

        doc = Document(
            page_content=text,
            metadata={
                "supplier_id": str(row['supplier_id']),        # ChromaDB requires strings
                "company_name": str(row['company_name']),
                "country": str(row['country']),
                "product_category": str(row['product_category']),
                "cold_chain": str(bool(row['cold_chain_capable'])),   # ChromaDB metadata must be str/int/float/bool
                "cdsco_approved": str(bool(row['cdsco_approved'])),
                "price": float(row['unit_price_inr']),
                "lead_time": int(row['lead_time_days'])
            }
        )
        documents.append(doc)

    print(f"✓ Created {len(documents)} document objects")

    # Create embeddings model
    print("\n📊 Creating embeddings (2-3 minutes)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'}
    )

    print("✓ Embeddings model loaded")

    # Create ChromaDB vector store with persistent storage
    print(f"\n🗄️  Creating ChromaDB database...")

    persist_dir = "database/chroma_suppliers"
    os.makedirs(persist_dir, exist_ok=True)

    # If a database already exists, delete the collection first to avoid duplicates on re-runs
    import chromadb
    client = chromadb.PersistentClient(path=persist_dir)
    existing = [c.name for c in client.list_collections()]
    if "pharma_suppliers" in existing:
        client.delete_collection("pharma_suppliers")
        print("⚠️  Existing collection deleted — recreating fresh...")

    vectorstore = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        collection_name="pharma_suppliers",   # Named collection inside ChromaDB
        persist_directory=persist_dir          # Persists to disk automatically
    )

    print(f"\n✅ SUCCESS!")
    print(f"   Loaded {len(documents)} suppliers into ChromaDB")
    print(f"   Saved to: {persist_dir}/")

    # Test search
    print(f"\n🔍 Testing search...")
    results = vectorstore.similarity_search(
        "Find insulin suppliers in India with cold chain",
        k=3
    )

    print(f"✓ Test search returned {len(results)} results")
    print(f"\nTop result:")
    print(results[0].page_content[:200] + "...")

    return vectorstore


if __name__ == "__main__":
    load_suppliers_to_chroma()