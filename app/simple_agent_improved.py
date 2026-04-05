"""
app/simple_agent_improved.py
MediCare Pharmaceuticals India — Advanced RAG Agent
----------------------------------------------------
Architecture: "LLM as ranker, not fact source"
  - ALL supplier facts built from doc.metadata in Python
  - LLM writes one sentence justification only
  - LLM never touches facts

FIX v4:
  - ask() now accepts blocked_country param — passed from orchestrator
    via tools.py → excludes suppliers from banned countries
    e.g. "China blocks exports" → Chinese suppliers disqualified
  - Removed unreachable dead code in _llm_recommend()
  - DEBUG_METADATA set to False by default (set True once to inspect keys)

Install deps:
    pip install sentence-transformers rank_bm25
"""

import json
from llama_cpp import Llama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from sentence_transformers import CrossEncoder
from rank_bm25 import BM25Okapi

# ── Set True on first run to print raw metadata keys, then set False ──────
DEBUG_METADATA = False


class PharmaSupplyChainAgent:

    def __init__(self):
        print("🤖 Initializing Pharma Supply Chain Agent...\n")

        with open("data/company/company_profile.json") as f:
            self.company = json.load(f)

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
        self.retriever = vectorstore.as_retriever(search_kwargs={"k": 10})

        print("📑 Building BM25 index...")
        self.all_docs = self.retriever.invoke("pharmaceutical supplier India")
        corpus = [doc.page_content.split() for doc in self.all_docs]
        self.bm25 = BM25Okapi(corpus)

        print("⚖️  Loading cross-encoder re-ranker...")
        self.reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")

        # Initialise so tools.py never hits AttributeError
        self.last_retrieved_docs = []
        self._metadata_printed   = False

        print("🧠 Loading GGUF model...")
        self.llm = Llama(
            model_path="models/llama-3-8b_300.Q4_K_M.gguf",
            n_ctx=4096,
            n_gpu_layers=0,
            chat_format="chatml"
        )

        print("✅ Agent ready!\n")

    # ──────────────────────────────────────────────────────────────────────
    # DEBUG: print raw metadata keys from first doc — run once to inspect
    # ──────────────────────────────────────────────────────────────────────
    def _debug_metadata(self, doc):
        if DEBUG_METADATA and not self._metadata_printed:
            print("\n" + "="*50)
            print("🔑 RAW METADATA KEYS IN FAISS (first doc):")
            for k, v in doc.metadata.items():
                print(f"   {k!r}: {v!r}")
            print("="*50 + "\n")
            self._metadata_printed = True

    # ──────────────────────────────────────────────────────────────────────
    # HELPER: try multiple metadata key names
    # ──────────────────────────────────────────────────────────────────────
    @staticmethod
    def _meta(m: dict, *keys, default="?"):
        for k in keys:
            v = m.get(k)
            if v is not None and str(v).strip() not in ("", "nan", "None", "?"):
                return v
        return default

    # ──────────────────────────────────────────────────────────────────────
    # QUERY REWRITING
    # ──────────────────────────────────────────────────────────────────────
    def rewrite_query(self, question: str) -> str:
        try:
            response = self.llm.create_chat_completion(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "Output 5-8 search keywords only. "
                            "No sentences, no bullet points, no numbers, no supplier names. "
                            "Example: insulin CDSCO India cold-chain fast lead-time"
                        )
                    },
                    {"role": "user", "content": question}
                ],
                max_tokens=30,
                temperature=0.1,
            )
            rewritten = response["choices"][0]["message"]["content"].strip()

            if any(c in rewritten for c in ['1.', '2.', '\n', ':', 'Lupin', 'Cipla']):
                return question
            if not rewritten or len(rewritten) > 120:
                return question

            return rewritten

        except Exception as e:
            print(f"  [WARN] Query rewrite failed: {e}")
            return question

    # ──────────────────────────────────────────────────────────────────────
    # RETRIEVAL: FAISS + BM25 → merge → rerank → top 5
    # ──────────────────────────────────────────────────────────────────────
    def _retrieve(self, question: str) -> tuple:
        rewritten = self.rewrite_query(question)
        if rewritten != question:
            print(f"  📝 Rewritten: {rewritten}")

        faiss_docs   = self.retriever.invoke(rewritten)
        bm25_scores  = self.bm25.get_scores(rewritten.split())
        top_bm25_idx = bm25_scores.argsort()[-5:][::-1]
        bm25_docs    = [self.all_docs[i] for i in top_bm25_idx]

        seen, merged = set(), []
        for doc in faiss_docs + bm25_docs:
            sid = doc.metadata.get("supplier_id") or doc.page_content[:40]
            if sid not in seen:
                seen.add(sid)
                merged.append(doc)

        pairs    = [[question, doc.page_content] for doc in merged]
        scores   = self.reranker.predict(pairs)
        reranked = [doc for _, doc in sorted(zip(scores, merged), reverse=True)]
        top5     = reranked[:5]

        if top5:
            self._debug_metadata(top5[0])

        return top5, rewritten

    # ──────────────────────────────────────────────────────────────────────
    # METADATA FILTER
    # FIX v4: accepts blocked_country to exclude suppliers from banned nations
    # ──────────────────────────────────────────────────────────────────────
    def _filter(self, docs: list, question_lower: str,
                blocked_country: str = None) -> tuple:

        requires_cdsco      = any(w in question_lower for w in
                                  ['cdsco', 'approved', 'regulatory', 'compliance'])
        requires_cold_chain = any(w in question_lower for w in
                                  ['insulin', 'vaccine', 'cold chain', 'temperature', 'refrigerat'])
        requires_india      = 'india' in question_lower
        requires_fast       = any(w in question_lower for w in
                                  ['fast', 'urgent', 'quick', 'immediate', 'emergency',
                                   'alternative', 'block'])

        # ── FIX: detect blocked country from query if not passed explicitly ──
        if blocked_country is None:
            blocked_countries = ['china', 'russia', 'pakistan', 'north korea']
            for country in blocked_countries:
                if country in question_lower and any(
                    w in question_lower for w in
                    ['block', 'ban', 'sanction', 'alternative', 'export', 'restrict']
                ):
                    blocked_country = country
                    break

        print(f"\n🔍 Requirements detected:")
        print(f"   CDSCO={requires_cdsco} | ColdChain={requires_cold_chain} "
              f"| India={requires_india} | Fast={requires_fast}")
        if blocked_country:
            print(f"   BlockedCountry={blocked_country} ← excluding these suppliers")

        qualified, disqualified = [], []

        for doc in docs:
            m       = doc.metadata
            reasons = []

            if requires_cdsco and not m.get('cdsco_approved', False):
                reasons.append("Not CDSCO approved")
            if requires_cold_chain and not m.get('cold_chain', False):
                reasons.append("No cold chain")
            if requires_india and m.get('country', '').lower() != 'india':
                reasons.append("Not in India")
            if requires_fast and m.get('lead_time', 999) > 21:
                reasons.append(f"Lead time {m.get('lead_time')} days > 21")

            # ── FIX: exclude suppliers from blocked country ──────────────
            if blocked_country:
                supplier_country = m.get('country', '').lower()
                if blocked_country in supplier_country:
                    reasons.append(
                        f"Supplier in blocked country ({m.get('country', '?')}) "
                        f"— seeking alternatives"
                    )

            if reasons:
                disqualified.append({'name': m.get('company_name', '?'), 'reasons': reasons})
            else:
                qualified.append(doc)

        print(f"   Qualified={len(qualified)} | Disqualified={len(disqualified)}")
        return qualified, disqualified

    # ──────────────────────────────────────────────────────────────────────
    # BUILD SUPPLIER TABLE — all facts from Python metadata
    # ──────────────────────────────────────────────────────────────────────
    def _build_supplier_table(self, docs: list) -> tuple:
        lines    = []
        llm_list = []

        for i, doc in enumerate(docs, 1):
            m = doc.metadata

            name    = self._meta(m, 'company_name', default='Unknown')
            sid     = self._meta(m, 'supplier_id',  default='?')
            country = self._meta(m, 'country',       default='Unknown')

            city = self._meta(
                m, 'city', 'location', 'headquarters', 'region',
                'state', 'district', 'address',
                default='Unknown'
            )

            price = self._meta(m, 'price', 'unit_price_inr', 'unit_price',
                               'price_inr', default=0)
            lead  = self._meta(m, 'lead_time', 'lead_time_days',
                               'delivery_days', 'lead_days', default='?')

            reliability = self._meta(
                m, 'reliability_score', 'reliability',
                'on_time_delivery', 'delivery_reliability',
                'supplier_reliability', 'score',
                default='?'
            )

            cdsco     = '✅' if m.get('cdsco_approved', False) else '❌'
            cold      = '✅' if m.get('cold_chain', False) else '❌'
            category  = self._meta(m, 'product_category', 'category',
                                   'product_type', default='Unknown')
            emergency = '✅' if m.get('emergency_supply_available', False) else '❌'

            try:
                price_str = f"₹{float(price):.2f}"
            except (ValueError, TypeError):
                price_str = f"₹{price}"

            block = (
                f"SUPPLIER {i}: {name} ({sid})\n"
                f"  Category    : {category}\n"
                f"  Location    : {city}, {country}\n"
                f"  Price       : {price_str}/unit\n"
                f"  Lead Time   : {lead} days\n"
                f"  Reliability : {reliability}%\n"
                f"  CDSCO       : {cdsco}\n"
                f"  Cold Chain  : {cold}\n"
                f"  Emergency   : {emergency}"
            )
            lines.append(block)

            llm_list.append(
                f"Supplier {i} ({sid}): lead={lead}d, "
                f"reliability={reliability}%, price=₹{price}"
            )

        return "\n\n".join(lines), llm_list

    # ──────────────────────────────────────────────────────────────────────
    # RECOMMENDATION — pure Python, no LLM, no hallucination risk
    # Dead code removed from v3
    # ──────────────────────────────────────────────────────────────────────
    def _llm_recommend(self, question: str, llm_list: list) -> str:
        if not llm_list:
            return "No qualified suppliers found."

        top = llm_list[0]

        if len(llm_list) == 1:
            return (
                f"RECOMMEND {top.split(':')[0]} — only qualified supplier. "
                f"Verify availability before ordering."
            )

        second = llm_list[1]
        return (
            f"RECOMMEND {top.split(':')[0]} — fastest lead time. "
            f"Alternative: {second.split(':')[0]} if Supplier 1 is unavailable."
        )

    # ──────────────────────────────────────────────────────────────────────
    # MAIN ask() METHOD
    # FIX v4: accepts blocked_country — passed from tools.py → _filter()
    # ──────────────────────────────────────────────────────────────────────
    def ask(self, question: str, blocked_country: str = None) -> str:

        docs, rewritten = self._retrieve(question)
        qualified, disqualified = self._filter(
            docs, question.lower(), blocked_country=blocked_country
        )

        # Set last_retrieved_docs to qualified only — tools.py reads this
        self.last_retrieved_docs = qualified

        if not qualified:
            disq_lines = "\n".join(
                f"  - {d['name']}: {', '.join(d['reasons'])}"
                for d in disqualified
            )
            return (
                "❌ NO QUALIFIED SUPPLIERS FOUND\n\n"
                "DISQUALIFIED:\n" + disq_lines +
                "\n\nTry relaxing requirements (e.g. remove 'fast' or 'India')."
            )

        # Sort fastest first
        qualified.sort(
            key=lambda d: d.metadata.get(
                'lead_time_days', d.metadata.get('lead_time', 999)
            )
        )

        supplier_table, llm_list = self._build_supplier_table(qualified)
        recommendation = self._llm_recommend(question, llm_list)

        disq_section = ""
        if disqualified:
            disq_section = "\n\nDISQUALIFIED SUPPLIERS:\n" + "\n".join(
                f"  ❌ {d['name']}: {', '.join(d['reasons'])}"
                for d in disqualified
            )

        return (
            f"QUALIFIED SUPPLIERS ({len(qualified)} found):\n\n"
            f"{supplier_table}"
            f"{disq_section}\n\n"
            f"{'─'*50}\n"
            f"RECOMMENDATION:\n{recommendation}"
        )


if __name__ == "__main__":

    print("🚀 Starting Pharma Supply Chain Agent\n")
    agent = PharmaSupplyChainAgent()

    test_questions = [
        "Find CDSCO approved insulin manufacturers in India with fast delivery.",
        "Find vaccine suppliers with cold chain capability.",
        "Find low-cost generic drug manufacturers.",
        "China blocks API exports for antibiotics. Find alternative suppliers urgently.",
        "I need insulin stuff urgently",
    ]

    use_test = input("\nRun test questions? (y/n): ").lower() == 'y'

    if use_test:
        for i, question in enumerate(test_questions, 1):
            print(f"\n{'='*60}")
            print(f"TEST {i}: {question}")
            print(f"{'='*60}\n")
            print(agent.ask(question))
            input("\nPress Enter for next test...")
    else:
        while True:
            query = input("\nAsk a supplier question (type 'exit' to quit): ")
            if query.lower() == "exit":
                print("👋 Goodbye!")
                break
            print("\n🔎 Searching...\n")
            print(agent.ask(query))
