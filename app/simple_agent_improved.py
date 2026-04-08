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
from pathlib import Path
import pandas as pd
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

        # Load CSV for fields not stored in FAISS metadata (city, reliability_score)
        csv_path = Path("data/suppliers/pharma_suppliers.csv")
        if csv_path.exists():
            df = pd.read_csv(csv_path)
            self._supplier_lookup = {
                row['supplier_id']: row.to_dict()
                for _, row in df.iterrows()
            }
        else:
            self._supplier_lookup = {}

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

        print("📑 Building BM25 index over ALL suppliers...")
        # Use the vectorstore directly to get all docs (not limited to k=10)
        self.all_docs = vectorstore.docstore._dict.values() if hasattr(vectorstore.docstore, '_dict') else []
        self.all_docs = list(self.all_docs)
        if not self.all_docs:
            # Fallback: retrieve a large set across diverse queries
            seen_ids, all_docs = set(), []
            for seed in ["antibiotic", "insulin", "vaccine", "generic", "cardiac",
                         "respiratory", "oncology", "cold chain", "India", "API"]:
                for doc in vectorstore.similarity_search(seed, k=15):
                    sid = doc.metadata.get("supplier_id", doc.page_content[:40])
                    if sid not in seen_ids:
                        seen_ids.add(sid)
                        all_docs.append(doc)
            self.all_docs = all_docs
        print(f"   BM25 indexed {len(self.all_docs)} suppliers")
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
            chat_format="chatml",
            verbose=False       # suppress llama_kv_cache / layer logs
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
    def _retrieve(self, question: str, blocked_country: str = None) -> tuple:
        rewritten = self.rewrite_query(question)
        if rewritten != question:
            print(f"  📝 Rewritten: {rewritten}")

        # When a country is blocked, bias retrieval toward India/other alternatives
        retrieval_query = rewritten
        if blocked_country:
            retrieval_query = rewritten + " India alternative supplier"
            print(f"  🔄 Retrieval biased away from {blocked_country}: {retrieval_query}")

        faiss_docs   = self.retriever.invoke(retrieval_query)
        bm25_scores  = self.bm25.get_scores(retrieval_query.split())
        top_bm25_idx = bm25_scores.argsort()[-5:][::-1]
        bm25_docs    = [self.all_docs[i] for i in top_bm25_idx]

        seen, merged = set(), []
        for doc in faiss_docs + bm25_docs:
            sid = doc.metadata.get("supplier_id") or doc.page_content[:40]
            if sid not in seen:
                seen.add(sid)
                merged.append(doc)

        # Pre-filter blocked country BEFORE re-ranking so the cross-encoder
        # cannot score them back into the top-5 (it sees "China" in the query
        # and may rank Chinese suppliers highest even when we want alternatives)
        if blocked_country:
            before = len(merged)
            merged = [
                doc for doc in merged
                if blocked_country not in doc.metadata.get('country', '').lower()
            ]
            print(f"  🚫 Pre-filter removed {before - len(merged)} {blocked_country} "
                  f"suppliers; {len(merged)} remain for re-ranking")

        if not merged:
            return [], rewritten

        # Use retrieval_query (not original question) for cross-encoder so
        # country-name in the query doesn't bias scores toward blocked country
        pairs    = [[retrieval_query, doc.page_content] for doc in merged]
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
    # Maps query keywords → accepted product_category substrings (lowercase)
    _CATEGORY_MAP = {
        'insulin'    : ['diabetes', 'insulin'],
        'diabetes'   : ['diabetes', 'insulin'],
        'antibiotic' : ['antibiotic', 'api', 'active pharmaceutical'],
        'antibiotics': ['antibiotic', 'api', 'active pharmaceutical'],
        'vaccine'    : ['vaccine'],
        'vaccines'   : ['vaccine'],
        'covid'      : ['vaccine'],
        'cardiac'    : ['cardiac'],
        'oncology'   : ['oncology'],
        'cancer'     : ['oncology'],
        'respiratory': ['respiratory'],
        'pain'       : ['pain relief'],
        'generic'    : ['generic'],
    }

    def _filter(self, docs: list, question_lower: str,
                blocked_country: str = None) -> tuple:

        requires_cdsco      = any(w in question_lower for w in
                                  ['cdsco', 'approved', 'regulatory', 'compliance'])
        requires_cold_chain = any(w in question_lower for w in
                                  ['insulin', 'vaccine', 'cold chain', 'temperature', 'refrigerat'])
        # Only require India when query explicitly asks for Indian suppliers.
        # Phrases like "our warehouse is in India" or "global sourcing" should NOT
        # lock the search to India-only suppliers.
        _global_override = any(p in question_lower for p in
                               ['global sourcing', 'international', 'no domestic',
                                'worldwide', 'outside india'])
        requires_india      = 'india' in question_lower and not _global_override
        # Only enforce fast lead-time when query explicitly asks for urgency.
        # "fastest available delivery" without an emergency keyword should not
        # disqualify all suppliers whose lead time is >21 days.
        requires_fast       = any(w in question_lower for w in
                                  ['urgent', 'quick', 'immediate', 'emergency'])

        # Detect required product categories from query keywords
        required_categories = []
        for keyword, cats in self._CATEGORY_MAP.items():
            if keyword in question_lower:
                required_categories.extend(cats)
        required_categories = list(set(required_categories))

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
        if required_categories:
            print(f"   Categories={required_categories}")
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

            # Category match — only reject if query clearly targets a specific product
            if required_categories:
                supplier_cat = m.get('product_category', '').lower()
                if not any(cat in supplier_cat for cat in required_categories):
                    reasons.append(
                        f"Product mismatch ({m.get('product_category', '?')} "
                        f"vs requested: {', '.join(required_categories)})"
                    )

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

            # Enrich from CSV for fields not stored in FAISS metadata
            csv_row = self._supplier_lookup.get(sid, {})

            city = self._meta(
                m, 'city', 'location', 'headquarters', 'region',
                'state', 'district', 'address',
                default=None
            ) or str(csv_row.get('city', 'Unknown'))

            price = self._meta(m, 'price', 'unit_price_inr', 'unit_price',
                               'price_inr', default=0)
            lead  = self._meta(m, 'lead_time', 'lead_time_days',
                               'delivery_days', 'lead_days', default='?')

            reliability = self._meta(
                m, 'reliability_score', 'reliability',
                'on_time_delivery', 'delivery_reliability',
                'supplier_reliability', 'score',
                default=None
            )
            if reliability is None:
                reliability = csv_row.get('reliability_score', '?')

            # Extra fields from CSV
            stock          = csv_row.get('current_stock', '?')
            unit           = csv_row.get('unit', 'units')
            min_order      = csv_row.get('minimum_order_quantity', '?')
            email          = csv_row.get('contact_email', '')
            phone          = csv_row.get('phone', '')
            business_hours = csv_row.get('business_hours', '')

            cdsco     = '✅' if m.get('cdsco_approved', False) else '❌'
            cold      = '✅' if m.get('cold_chain', False) else '❌'
            category  = self._meta(m, 'product_category', 'category',
                                   'product_type', default='Unknown')

            try:
                price_str = f"₹{float(price):.2f}"
            except (ValueError, TypeError):
                price_str = f"₹{price}"

            # Format stock level with urgency hint
            try:
                stock_int = int(stock)
                if stock_int == 0:
                    stock_str = "⚠️ Out of stock"
                elif stock_int < 500:
                    stock_str = f"⚠️ Low — {stock_int:,} {unit}"
                else:
                    stock_str = f"✅ {stock_int:,} {unit}"
            except (ValueError, TypeError):
                stock_str = "Unknown"

            rank_label = ["🥇 BEST MATCH", "🥈 GOOD ALTERNATIVE", "🥉 BACKUP OPTION"]
            rank_str   = rank_label[i - 1] if i <= 3 else f"Option {i}"

            compliance_tags = []
            if cdsco == '✅':
                compliance_tags.append("CDSCO Approved")
            if cold == '✅':
                compliance_tags.append("Cold Chain Ready")
            compliance_str = " | ".join(compliance_tags) if compliance_tags else "Check compliance"

            try:
                rel_float = float(reliability)
                # Use round() so 98.8% shows as 10/10 not 9/10
                filled    = min(round(rel_float / 10), 10)
                rel_bar   = "█" * filled + "░" * (10 - filled)
                rel_str   = f"{rel_float:.1f}%  [{rel_bar}]"
            except (ValueError, TypeError):
                rel_str = "N/A"

            # Inline warnings for important compliance gaps
            warnings = []
            if cdsco == '❌':
                warnings.append("⚠️ Not CDSCO approved — verify regulatory status before ordering in India")
            high_risk = ['china', 'russia', 'pakistan', 'north korea']
            if any(c in country.lower() for c in high_risk):
                warnings.append(f"⚠️ {country}-based — consider geopolitical supply risk")
            warning_lines = ("\n\n" + "\n\n".join(warnings)) if warnings else ""

            # Use markdown formatting so Streamlit renders each field on its own line.
            # \n\n = paragraph break in markdown (single \n is collapsed).
            contact_parts = []
            if phone:  contact_parts.append(f"📞 {phone}")
            if email:  contact_parts.append(f"✉️ {email}")
            if business_hours: contact_parts.append(f"🕐 {business_hours}")
            contact_str = " &nbsp;|&nbsp; ".join(contact_parts)

            block = (
                f"\n\n---\n\n"
                f"**{rank_str} — {name}**\n\n"
                f"**ID:** {sid} &nbsp;|&nbsp; **Product:** {category}\n\n"
                f"**Location:** {city}, {country}\n\n"
                + (f"**Contact:** {contact_str}\n\n" if contact_str else "")
                + f"**Price:** {price_str}/unit &nbsp;|&nbsp; "
                f"**Min. Order:** {min_order} {unit} &nbsp;|&nbsp; "
                f"**Delivery:** {lead} days\n\n"
                f"**Stock:** {stock_str} &nbsp;|&nbsp; "
                f"**Reliability:** {rel_str}\n\n"
                f"**Badges:** {compliance_str}"
                f"{warning_lines}"
            )
            lines.append(block)

            llm_list.append({
                "rank": i, "sid": sid, "name": name,
                "lead": lead, "reliability": reliability,
                "price": price, "country": country,
                "cdsco": cdsco == '✅', "cold": cold == '✅',
                "category": category, "email": email,
                "stock": stock_str,
            })

        return "\n".join(lines), llm_list

    # ──────────────────────────────────────────────────────────────────────
    # RECOMMENDATION — pure Python, no LLM, no hallucination risk
    # Dead code removed from v3
    # ──────────────────────────────────────────────────────────────────────
    def _llm_recommend(self, question: str, llm_list: list) -> str:
        if not llm_list:
            return "No qualified suppliers found."

        top = llm_list[0]
        q   = question.lower()

        # Detect scenario for contextual advice
        is_emergency    = any(w in q for w in ['urgent', 'emergency', 'power failure',
                                               'threat', 'block', 'ban'])
        is_cold_chain   = any(w in q for w in ['cold chain', 'vaccine', 'insulin',
                                               'temperature', 'refrigerat'])
        is_geopolitical = any(w in q for w in ['block', 'ban', 'sanction', 'export'])
        is_price_query  = any(w in q for w in ['low-cost', 'low cost', 'cheap',
                                               'affordable', 'cost', 'price',
                                               'inexpensive', 'budget'])

        try:
            lead_days = int(top['lead'])
            price_str = f"₹{float(top['price']):.2f}"
        except (ValueError, TypeError):
            lead_days = top['lead']
            price_str = f"₹{top['price']}"

        badges = []
        if top['cdsco']:  badges.append("CDSCO approved")
        if top['cold']:   badges.append("cold chain capable")
        badge_str = " and ".join(badges) if badges else "check compliance before ordering"

        email     = top.get('email', '')
        email_str = f" Reach them at **{email}**." if email else ""

        lines = []

        # Primary recommendation — mention price advantage for cost queries
        if is_price_query:
            lines.append(
                f"Contact {top['name']} ({top['sid']}) first — most affordable option "
                f"at {price_str}/unit with {lead_days}-day delivery.{email_str}"
            )
        else:
            lines.append(
                f"Contact {top['name']} ({top['sid']}) first. "
                f"They can deliver in {lead_days} days at {price_str}/unit "
                f"and are {badge_str}.{email_str}"
            )

        if is_emergency:
            lines.append(
                "Given the urgency, confirm stock availability and request "
                "an expedited shipment confirmation before raising a purchase order."
            )

        if is_cold_chain:
            lines.append(
                "Verify cold chain continuity documents (temperature logs, "
                "GDP certification) with the supplier before dispatch."
            )

        if is_geopolitical:
            lines.append(
                "This supplier is outside the affected region, reducing "
                "geopolitical supply risk."
            )

        # Fallback option
        if len(llm_list) >= 2:
            alt = llm_list[1]
            try:
                alt_lead = int(alt['lead'])
                alt_price = f"₹{float(alt['price']):.2f}"
            except (ValueError, TypeError):
                alt_lead  = alt['lead']
                alt_price = f"₹{alt['price']}"

            # Warn if fallback supplier is from a geopolitically sensitive country
            high_risk_countries = ['china', 'russia', 'pakistan', 'north korea']
            alt_country_lower   = alt['country'].lower()
            risk_note = ""
            if any(c in alt_country_lower for c in high_risk_countries):
                risk_note = (
                    f" Note: {alt['country']}-based suppliers carry higher "
                    f"geopolitical supply risk — use only if no other option is available."
                )

            lines.append(
                f"If {top['name']} cannot fulfil the order, escalate to "
                f"{alt['name']} ({alt['sid']}) — {alt_lead}-day lead time "
                f"at {alt_price}/unit.{risk_note}"
            )

        lines.append(
            "All draft purchase orders require sign-off from a procurement "
            "officer before submission."
        )

        return "  ".join(lines)

    # ──────────────────────────────────────────────────────────────────────
    # MAIN ask() METHOD
    # FIX v5: detect blocked_country BEFORE retrieval so _retrieve() can
    # bias the query away from that country (e.g. Streamlit calls ask()
    # directly without passing blocked_country)
    # ──────────────────────────────────────────────────────────────────────
    def ask(self, question: str, blocked_country: str = None) -> str:

        # Detect blocked country early so retrieval can be biased
        if blocked_country is None:
            q_lower = question.lower()
            _blocked_map = {
                'china': ['block', 'ban', 'sanction', 'alternative', 'export', 'restrict'],
                'russia': ['block', 'ban', 'sanction', 'alternative', 'export', 'restrict'],
                'pakistan': ['block', 'ban', 'sanction', 'alternative', 'export', 'restrict'],
                'north korea': ['block', 'ban', 'sanction', 'alternative', 'export', 'restrict'],
            }
            for country, triggers in _blocked_map.items():
                if country in q_lower and any(w in q_lower for w in triggers):
                    blocked_country = country
                    break

        docs, rewritten = self._retrieve(question, blocked_country=blocked_country)
        qualified, disqualified = self._filter(
            docs, question.lower(), blocked_country=blocked_country
        )

        # Set last_retrieved_docs to qualified only — tools.py reads this
        self.last_retrieved_docs = qualified

        # Build scenario header
        q = question.lower()
        is_geo       = any(w in q for w in ['block', 'ban', 'sanction', 'export'])
        is_emergency = any(w in q for w in ['power failure', 'emergency', 'threat',
                                             'urgent', 'urgently'])
        is_cold      = any(w in q for w in ['cold chain', 'vaccine', 'insulin',
                                             'temperature', 'refrigerat'])
        is_compliance = any(w in q for w in ['cdsco', 'compliance', 'regulatory',
                                              'approved', 'fast delivery', 'fast'])

        if is_geo:
            scenario = "🚨 GEOPOLITICAL DISRUPTION — Alternative Supplier Search"
        elif is_emergency and is_cold:
            scenario = "⚡❄️ EMERGENCY COLD CHAIN — Urgent Temperature-Sensitive Supply"
        elif is_emergency:
            scenario = "⚡ EMERGENCY RESPONSE — Urgent Supplier Needed"
        elif is_cold and is_compliance:
            scenario = "❄️ COLD CHAIN + COMPLIANCE — Regulated Temperature-Sensitive Supply"
        elif is_cold:
            scenario = "❄️ COLD CHAIN CRITICAL — Temperature-Sensitive Supply"
        elif is_compliance:
            scenario = "📋 COMPLIANCE SEARCH — Regulatory Requirements"
        else:
            scenario = "🔍 SUPPLIER SEARCH RESULTS"

        if not qualified:
            disq_lines = "\n".join(
                f"- **{d['name']}**: {', '.join(d['reasons'])}"
                for d in disqualified
            )
            return (
                f"## {scenario}\n\n"
                "No suppliers matched all your requirements. "
                "Here is what was found and why each was excluded:\n\n"
                + disq_lines +
                "\n\n> **Suggestion:** Try broadening the search — for example, "
                "remove the urgency constraint or expand to other regions."
            )

        # Sort by price for cost queries, otherwise by lead time
        price_query = any(w in q for w in ['low-cost', 'low cost', 'cheap', 'affordable',
                                            'cost', 'price', 'inexpensive', 'budget'])
        if price_query:
            qualified.sort(
                key=lambda d: float(d.metadata.get('price', 9999))
            )
        else:
            qualified.sort(
                key=lambda d: d.metadata.get('lead_time', 999)
            )

        supplier_table, llm_list = self._build_supplier_table(qualified)
        recommendation = self._llm_recommend(question, llm_list)

        disq_section = ""
        if disqualified:
            disq_names = "\n".join(
                f"- {d['name']}: {', '.join(d['reasons'])}"
                for d in disqualified
            )
            disq_section = f"\n\n**Also considered but excluded:**\n\n{disq_names}"

        return (
            f"## {scenario}\n\n"
            f"Found **{len(qualified)} supplier{'s' if len(qualified) != 1 else ''}** "
            f"that meet your requirements:"
            f"{supplier_table}"
            f"{disq_section}\n\n"
            f"---\n\n"
            f"### What to do next\n\n{recommendation}"
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
