"""
Simple pharmaceutical supply chain agent
Uses Llama3 + ChromaDB RAG + local GGUF model
IMPROVED VERSION - ChromaDB where clause filtering (no manual Python filtering loop)
"""

import json
import chromadb
from llama_cpp import Llama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma


class PharmaSupplyChainAgent:

    def __init__(self):
        print("🤖 Initializing Pharma Supply Chain Agent...\n")

        with open("data/company/company_profile.json") as f:
            self.company = json.load(f)

        with open("data/company/current_inventory.json") as f:
            self.inventory = json.load(f)

        print("📊 Loading ChromaDB supplier database...")

        self.embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        # ✅ Use PersistentClient directly for where-clause support
        self.chroma_client = chromadb.PersistentClient(
            path="database/chroma_suppliers"
        )

        self.collection = self.chroma_client.get_collection("pharma_suppliers")

        # Keep LangChain vectorstore for embedding queries
        self.vectorstore = Chroma(
            client=self.chroma_client,
            collection_name="pharma_suppliers",
            embedding_function=self.embeddings,
        )

        print("🧠 Loading GGUF model...")

        self.llm = Llama(
            model_path="models/llama-3-8b_300.Q4_K_M.gguf",
            n_ctx=4096,
            n_gpu_layers=0,
            chat_format="chatml"
        )

        # Strict prompt — no hallucination room
        self.prompt_template = """You are a pharmaceutical supply chain expert for MediCare Pharmaceuticals India.

STRICT RULES — NEVER VIOLATE THESE:
1. ONLY recommend suppliers listed in the QUALIFIED SUPPLIERS section below
2. NEVER invent suppliers, prices, emergency options, or procurement channels not in the data
3. Compare suppliers using ONLY the data fields provided (price, lead time, reliability)
4. If a supplier has a lower lead time number, they are FASTER — do not contradict this
5. Do NOT add government stockpile options, air freight estimates, or external contacts

Qualified supplier data:
{context}

User question: {question}

Respond in this exact format:

REQUIREMENTS ANALYSIS:
- [List each mandatory requirement detected from the question only — no invented numbers]

QUALIFIED SUPPLIERS:
[For each supplier: Name, Price, Lead Time, Reliability, CDSCO status, Cold Chain status]

RECOMMENDATION:
[Pick ONE supplier. Justify using exact numbers from the data above. If comparing lead times, state both numbers explicitly before concluding which is faster.]

Answer:"""

        print("✅ Agent ready!\n")

    def _build_where_clause(self, requires_cdsco, requires_cold_chain,
                         requires_india, requires_fast):
        conditions = []

        if requires_cdsco:
            conditions.append({"cdsco_approved": {"$eq": "True"}})  # string

        if requires_cold_chain:
            conditions.append({"cold_chain": {"$eq": "True"}})      # string

        if requires_india:
            conditions.append({"country": {"$eq": "India"}})        # string, capital I

        if requires_fast:
            conditions.append({"lead_time": {"$lte": 10}})          # int, no quotes

        if len(conditions) == 0:
            return None
        elif len(conditions) == 1:
            return conditions[0]
        else:
            return {"$and": conditions}

    def ask(self, question):
        question_lower = question.lower()

        requires_cdsco = any(w in question_lower for w in
                            ['cdsco', 'approved', 'regulatory', 'compliance'])
        requires_cold_chain = any(w in question_lower for w in
                                ['insulin', 'vaccine', 'cold chain',
                                    'temperature', 'refrigerat'])
        requires_india = 'india' in question_lower
        requires_fast = any(w in question_lower for w in
                            ['fast', 'urgent', 'quick', 'immediate', 'emergency'])

        print(f"\n🔍 Detected requirements:")
        print(f"   - CDSCO approval: {requires_cdsco}")
        print(f"   - Cold chain: {requires_cold_chain}")
        print(f"   - India location: {requires_india}")
        print(f"   - Fast delivery: {requires_fast}")

        # ✅ Embed the question directly
        query_embedding = self.embeddings.embed_query(question)

        # ✅ Build where clause with correct types
        where_clause = self._build_where_clause(
            requires_cdsco, requires_cold_chain, requires_india, requires_fast
        )

        # ✅ Query ChromaDB directly — no LangChain wrapper
        query_kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": 5,
            "include": ["documents", "metadatas", "distances"]
        }
        if where_clause:
            query_kwargs["where"] = where_clause

        qualified_results = self.collection.query(**query_kwargs)

        # Also fetch all 5 unfiltered to show disqualified
        all_results = self.collection.query(
            query_embeddings=[query_embedding],
            n_results=5,
            include=["documents", "metadatas"]
        )

        # Build qualified docs list
        qualified_names = set()
        qualified_docs = []
        if qualified_results["metadatas"] and qualified_results["metadatas"][0]:
            for meta, doc in zip(qualified_results["metadatas"][0],
                                qualified_results["documents"][0]):
                qualified_names.add(meta.get("company_name"))
                qualified_docs.append({"metadata": meta, "content": doc})

        # Work out disqualified
        disqualified = []
        if all_results["metadatas"] and all_results["metadatas"][0]:
            for meta in all_results["metadatas"][0]:
                name = meta.get("company_name", "Unknown")
                if name not in qualified_names:
                    reasons = []
                    if requires_cdsco and meta.get("cdsco_approved") != "True":
                        reasons.append("Not CDSCO approved")
                    if requires_cold_chain and meta.get("cold_chain") != "True":
                        reasons.append("No cold chain capability")
                    if requires_india and meta.get("country", "").lower() != "india":
                        reasons.append("Not in India")
                    if requires_fast and int(meta.get("lead_time", 999)) > 10:
                        reasons.append(f"Lead time too long ({meta.get('lead_time')} days)")
                    disqualified.append({"name": name, "reasons": reasons})

        print(f"\n📋 Filtering results:")
        print(f"   - Initial suppliers from ChromaDB: {len(all_results['metadatas'][0])}")
        print(f"   - Qualified after filtering: {len(qualified_docs)}")
        print(f"   - Disqualified: {len(disqualified)}")

        # Handle no results
        if not qualified_docs:
            disqualified_info = "\n".join([
                f"   - {d['name']}: {', '.join(d['reasons'])}"
                for d in disqualified
            ])
            return f"""❌ NO QUALIFIED SUPPLIERS FOUND

REQUIREMENTS CHECKED:
{'✅ CDSCO approved' if requires_cdsco else ''}
{'✅ Cold chain capability (2-8°C)' if requires_cold_chain else ''}
{'✅ Located in India' if requires_india else ''}
{'✅ Fast delivery (<10 days)' if requires_fast else ''}

DISQUALIFIED SUPPLIERS:
{disqualified_info}

RECOMMENDATION:
Please adjust requirements or verify supplier certifications directly."""

        # Build context
        context_parts = []
        for i, doc in enumerate(qualified_docs, 1):
            meta = doc["metadata"]
            try:
                price_str = f"₹{float(meta.get('price', 0)):.2f}"
            except (ValueError, TypeError):
                price_str = f"₹{meta.get('price', 'N/A')}"

            context_parts.append(f"""
QUALIFIED SUPPLIER {i}:
Company: {meta.get('company_name', 'Unknown')}
CDSCO Approved: {meta.get('cdsco_approved')}
Cold Chain: {meta.get('cold_chain')}
Country: {meta.get('country')}
Price: {price_str}
Lead Time: {meta.get('lead_time')} days
Reliability: {meta.get('reliability')}%
Emergency Supply: {meta.get('emergency_supply')}
""")

        context = "\n".join(context_parts)

        if disqualified:
            context += "\n\nDISQUALIFIED SUPPLIERS (DO NOT RECOMMEND):\n"
            for d in disqualified:
                context += f"- {d['name']}: {', '.join(d['reasons'])}\n"

        prompt = self.prompt_template.format(context=context, question=question)

        # ✅ Filtering rule in system message, not user prompt (stops model printing it)
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a pharmaceutical supply chain expert. "
                        "ONLY use facts from the qualified supplier data provided. "
                        "NEVER invent suppliers, prices, lead times, or procurement options. "
                        "When comparing lead times, state both numbers before concluding."
                    )
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,
            temperature=0.1,
            top_p=0.9,
            repeat_penalty=1.1
        )

        return response["choices"][0]["message"]["content"]


if __name__ == "__main__":

    print("🚀 Starting Pharma Supply Chain Agent\n")

    agent = PharmaSupplyChainAgent()

    test_questions = [
        "Find CDSCO approved insulin manufacturers in India with fast delivery.",
        "Find vaccine suppliers with cold chain capability.",
        "Find low-cost generic drug manufacturers.",
    ]

    print("\n" + "=" * 60)
    print("🧪 TESTING WITH SAMPLE QUESTIONS")
    print("=" * 60)

    try:
        use_test = input("\nRun test questions? (y/n): ").strip().lower() == 'y'
    except EOFError:
        use_test = False

    if use_test:
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'=' * 60}")
            print(f"TEST {i}: {question}")
            print(f"{'=' * 60}\n")
            response = agent.ask(question)
            print("💡 Answer:\n")
            print(response)
            print("\n" + "=" * 60)
            input("\nPress Enter for next test...")
    else:
        while True:
            query = input("\nAsk a supplier question (type 'exit' to quit): ")
            if query.lower() == "exit":
                print("👋 Goodbye!")
                break
            print("\n🔎 Searching supplier database...\n")
            response = agent.ask(query)
            print("💡 Answer:\n")
            print(response)
