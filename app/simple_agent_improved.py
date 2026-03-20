"""
Simple pharmaceutical supply chain agent
Uses Llama3 + FAISS RAG + local GGUF model
IMPROVED VERSION with better filtering and prompt engineering
"""

import json
from llama_cpp import Llama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import PromptTemplate


class PharmaSupplyChainAgent:

    def __init__(self):
        print("🤖 Initializing Pharma Supply Chain Agent...\n")

        # Load company profile
        with open("data/company/company_profile.json") as f:
            self.company = json.load(f)

        # Load inventory
        with open("data/company/current_inventory.json") as f:
            self.inventory = json.load(f)

        print("📊 Loading FAISS supplier database...")

        embeddings = HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )

        vectorstore = FAISS.load_local(
            "database/faiss_suppliers",
            embeddings,
            allow_dangerous_deserialization=True
        )

        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 5})  # ✅ Increased to 5 for better filtering

        print("🧠 Loading GGUF model...")

        self.llm = Llama(
            model_path="models/llama-3-8b.Q4_K_M.gguf",
            n_ctx=4096,
            n_gpu_layers=0,
            chat_format="chatml"
        )

        # ✅ IMPROVED PROMPT with strict filtering instructions
        self.prompt_template = """You are a pharmaceutical supply chain expert for MediCare Pharmaceuticals India.

CRITICAL RULES FOR SUPPLIER RECOMMENDATIONS:
1. CDSCO approval is MANDATORY for pharmaceutical sales in India
2. Cold chain (2-8°C) is MANDATORY for insulin, vaccines, and biologics
3. If a supplier does NOT meet mandatory requirements → EXCLUDE them entirely
4. Only recommend suppliers that meet ALL criteria mentioned in the question
5. Be specific: state CDSCO status, cold chain capability, lead time, and price
6. If NO suppliers qualify, clearly state this instead of recommending unqualified ones

Supplier Database Context:
{context}

Question:
{question}

INSTRUCTIONS:
- Identify mandatory requirements from the question (CDSCO? Cold chain? Location?)
- Filter suppliers: Only include those meeting ALL mandatory criteria
- List qualified suppliers with full details
- Clearly mark disqualified suppliers and explain why
- Provide a final recommendation based on best match

Answer format:
1. REQUIREMENTS ANALYSIS: [List mandatory criteria]
2. QUALIFIED SUPPLIERS: [Only those meeting ALL criteria]
3. DISQUALIFIED SUPPLIERS: [Why they failed]
4. RECOMMENDATION: [Best choice with justification]

Answer:"""

        print("✅ Agent ready!\n")

    def ask(self, question):
        """
        Ask the agent a question with intelligent filtering
        """

        # Retrieve relevant docs from FAISS
        docs = self.retriever.invoke(question)

        # ✅ PRE-FILTER suppliers based on question keywords
        question_lower = question.lower()

        # Detect mandatory requirements
        requires_cdsco = any(word in question_lower for word in ['cdsco', 'approved', 'regulatory', 'compliance'])
        requires_cold_chain = any(word in question_lower for word in ['insulin', 'vaccine', 'cold chain', 'temperature', 'refrigerat'])
        requires_india = 'india' in question_lower
        requires_fast = any(word in question_lower for word in ['fast', 'urgent', 'quick', 'immediate', 'emergency'])

        print(f"\n🔍 Detected requirements:")
        print(f"   - CDSCO approval: {requires_cdsco}")
        print(f"   - Cold chain: {requires_cold_chain}")
        print(f"   - India location: {requires_india}")
        print(f"   - Fast delivery: {requires_fast}")

        # Filter suppliers
        filtered_docs = []
        disqualified = []

        for doc in docs:
            metadata = doc.metadata
            reasons = []

            # Check CDSCO requirement
            if requires_cdsco:
                if not metadata.get('cdsco_approved', False):
                    reasons.append("Not CDSCO approved")

            # Check cold chain requirement
            if requires_cold_chain:
                if not metadata.get('cold_chain', False):
                    reasons.append("No cold chain capability")

            # Check India location
            if requires_india:
                if metadata.get('country', '').lower() != 'india':
                    reasons.append("Not in India")

            # Check fast delivery (< 10 days)
            if requires_fast:
                if metadata.get('lead_time', 999) > 10:
                    reasons.append(f"Lead time too long ({metadata.get('lead_time')} days)")

            # If disqualified, track why
            if reasons:
                disqualified.append({
                    'name': metadata.get('company_name', 'Unknown'),
                    'reasons': reasons
                })
            else:
                # Supplier passed all filters
                filtered_docs.append(doc)

        print(f"\n📋 Filtering results:")
        print(f"   - Initial suppliers from FAISS: {len(docs)}")
        print(f"   - Qualified after filtering: {len(filtered_docs)}")
        print(f"   - Disqualified: {len(disqualified)}")

        # If no suppliers qualify, return clear message
        if not filtered_docs:
            disqualified_info = "\n".join([
                f"   - {d['name']}: {', '.join(d['reasons'])}"
                for d in disqualified
            ])

            return f"""❌ NO QUALIFIED SUPPLIERS FOUND

REQUIREMENTS:
{'✅ CDSCO approved' if requires_cdsco else ''}
{'✅ Cold chain capability (2-8°C)' if requires_cold_chain else ''}
{'✅ Located in India' if requires_india else ''}
{'✅ Fast delivery (<10 days)' if requires_fast else ''}

DISQUALIFIED SUPPLIERS:
{disqualified_info}

RECOMMENDATION:
Please adjust your requirements or contact suppliers to verify if they can meet certification needs."""

        # Build context from FILTERED suppliers only
        context_parts = []
        for i, doc in enumerate(filtered_docs, 1):
            meta = doc.metadata
            context_parts.append(f"""
QUALIFIED SUPPLIER {i}:
Company: {meta.get('company_name', 'Unknown')}
✅ CDSCO Approved: {meta.get('cdsco_approved', False)}
✅ Cold Chain: {meta.get('cold_chain', False)}
✅ Country: {meta.get('country', 'Unknown')}
Price: ₹{meta.get('price', 0):.2f}
Lead Time: {meta.get('lead_time', 0)} days
Reliability: {meta.get('reliability', 0)}%
Emergency Supply: {meta.get('emergency_supply', False)}

Full Details:
{doc.page_content}
""")

        context = "\n".join(context_parts)

        # Add disqualified suppliers to context for transparency
        if disqualified:
            disqualified_context = "\n\nDISQUALIFIED SUPPLIERS (DO NOT RECOMMEND):\n"
            for d in disqualified:
                disqualified_context += f"- {d['name']}: {', '.join(d['reasons'])}\n"
            context += disqualified_context

        # Build the prompt
        prompt = self.prompt_template.format(context=context, question=question)

        # Run inference
        response = self.llm.create_chat_completion(
            messages=[
                {
                    "role": "system",
                    "content": "You are a pharmaceutical supply chain expert. Follow the filtering rules strictly. Only recommend qualified suppliers."
                },
                {"role": "user", "content": prompt}
            ],
            max_tokens=800,  # ✅ Increased for detailed responses
            temperature=0.1,  # Low temperature for factual responses
            top_p=0.9,
            repeat_penalty=1.1  # ✅ Reduce repetition
        )

        return response["choices"][0]["message"]["content"]


if __name__ == "__main__":

    print("🚀 Starting Pharma Supply Chain Agent\n")

    agent = PharmaSupplyChainAgent()

    # ✅ Test with sample questions
    test_questions = [
        "Find CDSCO approved insulin manufacturers in India with fast delivery.",
        "Find vaccine suppliers with cold chain capability.",
        "Find low-cost generic drug manufacturers.",
    ]

    print("\n" + "="*60)
    print("🧪 TESTING WITH SAMPLE QUESTIONS")
    print("="*60)

    use_test = input("\nRun test questions? (y/n): ").lower() == 'y'

    if use_test:
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {question}")
            print(f"{'='*60}\n")

            response = agent.ask(question)
            print("💡 Answer:\n")
            print(response)
            print("\n" + "="*60)

            input("\nPress Enter for next test...")
    else:
        # Interactive mode
        while True:
            query = input("\nAsk a supplier question (type 'exit' to quit): ")

            if query.lower() == "exit":
                print("👋 Goodbye!")
                break

            print("\n🔎 Searching supplier database...\n")
            response = agent.ask(query)
            print("💡 Answer:\n")
            print(response)
