"""
Load pharmaceutical suppliers into PostgreSQL + pgvector RAG database
"""

import pandas as pd
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores import PGVector
from langchain.docstore.document import Document
import os

def load_suppliers_to_rag():
    """
    Load supplier CSV into vector database for RAG
    """

    print("📥 Loading suppliers into RAG database...\n")

    # Load supplier CSV
    csv_path = 'data/suppliers/pharma_suppliers.csv'

    if not os.path.exists(csv_path):
        print(f"❌ Error: {csv_path} not found")
        print("Run scripts/2_generate_pharma_suppliers.py first")
        return

    df = pd.read_csv(csv_path)
    print(f"✓ Loaded {len(df)} suppliers from CSV")

    # Convert each supplier to a document
    documents = []

    for _, row in df.iterrows():
        # Create rich text description
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

Contact: {row['contact_email']} | {row['phone']}
        """.strip()

        # Create document with metadata
        doc = Document(
            page_content=text,
            metadata={
                "supplier_id": row['supplier_id'],
                "company_name": row['company_name'],
                "country": row['country'],
                "product_category": row['product_category'],
                "cold_chain": bool(row['cold_chain_capable']),
                "cdsco_approved": bool(row['cdsco_approved']),
                "price": float(row['unit_price_inr']),
                "lead_time": int(row['lead_time_days']),
                "reliability": float(row['reliability_score'])
            }
        )
        documents.append(doc)

    print(f"✓ Created {len(documents)} document objects")

    # Create embeddings model
    print("\n📊 Creating embeddings (this may take 2-3 minutes)...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    print("✓ Embeddings model loaded")

    # PostgreSQL connection string
    # Change this if you have a password or different setup
    CONNECTION_STRING = "postgresql://localhost:5433/pharma_supply_chain"

    print(f"\n🗄️  Storing in PostgreSQL database...")
    print(f"   Connection: {CONNECTION_STRING}")

    try:
        # Create vector store
        vectorstore = PGVector.from_documents(
            documents=documents,
            embedding=embeddings,
            collection_name="pharma_suppliers",
            connection_string=CONNECTION_STRING,
            pre_delete_collection=True  # Clear existing data
        )

        print(f"\n✅ SUCCESS!")
        print(f"   Loaded {len(documents)} suppliers into RAG database")
        print(f"   Collection name: pharma_suppliers")
        print(f"   Database: pharma_supply_chain")

        # Test search
        print(f"\n🔍 Testing search functionality...")
        results = vectorstore.similarity_search(
            "Find insulin suppliers in India with cold chain",
            k=3
        )

        print(f"✓ Test search returned {len(results)} results")
        print(f"\nTop result:")
        print(results[0].page_content[:200] + "...")

        return vectorstore

    except Exception as e:
        print(f"\n❌ Error connecting to database:")
        print(f"   {str(e)}")
        print(f"\nTroubleshooting:")
        print(f"   1. Is PostgreSQL running? (brew services list)")
        print(f"   2. Does database exist? (psql -l)")
        print(f"   3. Is pgvector installed? (psql pharma_supply_chain -c '\\dx')")
        return None

if __name__ == "__main__":
    load_suppliers_to_rag()
