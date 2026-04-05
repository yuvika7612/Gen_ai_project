"""
app/orchestrator.py
MediCare Pharmaceuticals India — LLM Orchestrator
--------------------------------------------------
Decides which tools to call based on query intent,
runs them, assembles the final answer.

Replanning loop: if the first plan produces no results,
the orchestrator tries a fallback plan (max 3 attempts).

FIX v3: Replaced LLM-based JSON planning with Python keyword routing.
  - Small quantized models (GGUF 4-bit) reliably ignore JSON format
    instructions and output plain English instead, causing JSON parse
    failures on every call.
  - Python keyword routing is instant, 100% reliable, and zero tokens.
  - The orchestrator still uses the LLM for supplier_search and
    recommendation — just NOT for tool selection planning.

FIX v4:
  - self.memory now resets at the start of each run() call — previously
    accumulated across unrelated queries and corrupted replanning context.
  - attempt variable initialised before the loop — avoids UnboundLocalError
    if MAX_REPLANS is ever set to 0.
  - _execute() now detects blocked country and passes it to supplier_search
    → simple_agent_improved._filter() excludes those suppliers correctly.
    e.g. "China blocks antibiotic exports" → Chinese suppliers disqualified.

Usage:
    from app.orchestrator import Orchestrator
    orc = Orchestrator(agent)
    result = orc.run("Find insulin suppliers urgently")
    print(result["final_answer"])

Or run directly:
    python app/orchestrator.py
"""

import sys
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from app.tools import (
    supplier_search,
    inventory_check,
    news_monitor,
    compliance_check,
    price_compare,
    draft_order,
)

# Countries that trigger export-ban / alternative-supplier logic
BLOCKED_COUNTRY_TRIGGERS = {
    'china'      : ['block', 'ban', 'sanction', 'alternative', 'export', 'restrict'],
    'russia'     : ['block', 'ban', 'sanction', 'alternative', 'export', 'restrict'],
    'pakistan'   : ['block', 'ban', 'sanction', 'alternative', 'export', 'restrict'],
    'north korea': ['block', 'ban', 'sanction', 'alternative', 'export', 'restrict'],
}


# ══════════════════════════════════════════════════════════════════════════
# ORCHESTRATOR CLASS
# ══════════════════════════════════════════════════════════════════════════
class Orchestrator:

    MAX_REPLANS = 3

    def __init__(self, agent=None):
        self.agent  = agent
        self.memory = []   # reset per run() call — see FIX v4

    # ──────────────────────────────────────────────────────────────────────
    # FIX: PYTHON KEYWORD ROUTING — replaces LLM JSON planning entirely
    # ──────────────────────────────────────────────────────────────────────
    def _plan(self, query: str) -> dict:
        """
        Decide which tools to call using Python keyword matching.
        No LLM call — instant and 100% reliable.
        """
        q = query.lower()

        # supplier_search is always called — it's the core tool
        tools = ["supplier_search"]
        reasoning_parts = ["supplier_search always called for supplier queries"]

        # inventory_check: stock / shortage / reorder keywords
        if any(w in q for w in ['stock', 'shortage', 'reorder', 'inventory',
                                  'running low', 'days of supply']):
            tools.append("inventory_check")
            reasoning_parts.append("inventory keywords detected")

        # news_monitor: disruption / geopolitical / ban keywords
        if any(w in q for w in ['disruption', 'ban', 'block', 'blocks',
                                  'shortage', 'china', 'news', 'recall',
                                  'contamination', 'export', 'sanction']):
            tools.append("news_monitor")
            reasoning_parts.append("disruption/news keywords detected")

        # compliance_check: only when a specific supplier ID is mentioned
        if re.search(r'pharm-\d{4}', q):
            tools.append("compliance_check")
            reasoning_parts.append("supplier ID found in query")

        # price_compare: explicit comparison / ranking request
        if any(w in q for w in ['compare', 'rank', 'cheapest', 'best price',
                                  'most affordable', 'cost comparison']):
            tools.append("price_compare")
            reasoning_parts.append("price comparison requested")

        # draft_order: explicit order / purchase request
        if any(w in q for w in ['draft order', 'place order', 'purchase order',
                                  'create order', 'draft a po', 'draft po']):
            tools.append("draft_order")
            reasoning_parts.append("order drafting requested")

        reasoning = "; ".join(reasoning_parts)
        print(f"  🗂️  Plan (keyword routing): {tools}")
        print(f"  💭 Reasoning: {reasoning}")

        return {"tools_to_call": tools, "reasoning": reasoning}

    # ──────────────────────────────────────────────────────────────────────
    # FIX v4: detect blocked country from query string
    # ──────────────────────────────────────────────────────────────────────
    def _detect_blocked_country(self, query: str) -> str | None:
        """
        Returns lowercase country name if the query implies a supply ban
        from that country, otherwise None.
        Example: "China blocks antibiotic exports" → "china"
        """
        q = query.lower()
        for country, trigger_words in BLOCKED_COUNTRY_TRIGGERS.items():
            if country in q and any(w in q for w in trigger_words):
                return country
        return None

    # ──────────────────────────────────────────────────────────────────────
    # EXECUTE: run the planned tools
    # ──────────────────────────────────────────────────────────────────────
    def _execute(self, query: str, plan: dict) -> dict:
        results      = {}
        tools_called = plan.get("tools_to_call", ["supplier_search"])

        # FIX v4: detect blocked country once, pass to supplier_search
        blocked_country = self._detect_blocked_country(query)
        if blocked_country:
            print(f"  🚫 Blocked country detected: {blocked_country} "
                  f"— will exclude from results")

        for tool_name in tools_called:
            print(f"  ▶ Running {tool_name}...")

            try:
                if tool_name == "supplier_search":
                    # FIX v4: pass blocked_country so filter excludes them
                    results["supplier_search"] = supplier_search(
                        query,
                        agent=self.agent,
                        blocked_country=blocked_country
                    )

                elif tool_name == "inventory_check":
                    product = self._extract_product(query)
                    results["inventory_check"] = inventory_check(product)

                elif tool_name == "news_monitor":
                    keyword = self._extract_keyword(query)
                    results["news_monitor"] = news_monitor(keyword)

                elif tool_name == "compliance_check":
                    supplier_id      = self._extract_supplier_id(query)
                    needs_cold_chain = any(w in query.lower() for w in
                                          ['insulin', 'vaccine', 'cold chain'])
                    if supplier_id:
                        results["compliance_check"] = compliance_check(
                            supplier_id, requires_cold_chain=needs_cold_chain
                        )
                    else:
                        results["compliance_check"] = {
                            "status"  : "skipped",
                            "message" : "No supplier ID found in query."
                        }

                elif tool_name == "price_compare":
                    ids = []
                    if "supplier_search" in results:
                        ids = [
                            s["id"]
                            for s in results["supplier_search"].get("suppliers", [])
                        ]
                    if ids:
                        results["price_compare"] = price_compare(ids)
                    else:
                        results["price_compare"] = {
                            "status"  : "skipped",
                            "message" : "Run supplier_search first to get supplier IDs."
                        }

                elif tool_name == "draft_order":
                    if "supplier_search" in results:
                        suppliers = results["supplier_search"].get("suppliers", [])
                        if suppliers:
                            top     = suppliers[0]
                            product = self._extract_product(query)
                            results["draft_order"] = draft_order(
                                supplier_id=top["id"],
                                product=product,
                                quantity=10000,
                                urgency="URGENT" if any(
                                    w in query.lower()
                                    for w in ['urgent', 'emergency', 'fast']
                                ) else "STANDARD"
                            )
                        else:
                            results["draft_order"] = {
                                "status"  : "skipped",
                                "message" : "No qualified supplier found."
                            }

            except Exception as e:
                results[tool_name] = {"status": "error", "message": str(e)}
                print(f"  [ERROR] {tool_name} failed: {e}")

        return results

    # ──────────────────────────────────────────────────────────────────────
    # ASSEMBLE: build final answer from tool results
    # All facts come from Python tool results — never from LLM generation.
    # ──────────────────────────────────────────────────────────────────────
    def _assemble(self, query: str, tool_results: dict) -> str:
        sections = [f"🔍 Query: {query}\n"]

        # Supplier search results
        if "supplier_search" in tool_results:
            sr = tool_results["supplier_search"]
            if sr.get("status") == "ok":
                sections.append(sr.get("answer", ""))
            else:
                sections.append("❌ No qualified suppliers found.")

        # Inventory status
        if "inventory_check" in tool_results:
            inv = tool_results["inventory_check"]
            if inv.get("status") == "found":
                reorder_flag = "⚠️ YES — reorder needed" if inv["reorder_needed"] else "✅ NO"
                sections.append(
                    f"\n📦 INVENTORY STATUS — {inv['product_name']}\n"
                    f"  Current Stock  : {inv['current_stock']:,} {inv['unit']}\n"
                    f"  Days of Supply : {inv['days_of_supply']} days\n"
                    f"  Urgency        : {inv['urgency']}\n"
                    f"  Reorder Needed : {reorder_flag}"
                )
            else:
                sections.append(f"\n📦 Inventory: {inv.get('message', 'Not found')}")

        # News alerts
        if "news_monitor" in tool_results:
            nm = tool_results["news_monitor"]
            if nm.get("status") == "ok" and nm.get("alerts"):
                sections.append(
                    f"\n📰 NEWS ALERTS ({nm['total_found']} found "
                    f"for '{nm['keyword']}'):"
                )
                for alert in nm["alerts"][:3]:
                    title = alert.get("title") or alert.get("headline", "No title")
                    date  = alert.get("date", "")
                    sections.append(f"  • [{date}] {title}")
            else:
                sections.append(
                    f"\n📰 News: No disruption alerts found "
                    f"for '{nm.get('keyword', '')}'"
                )

        # Compliance check
        if "compliance_check" in tool_results:
            cc = tool_results["compliance_check"]
            if cc.get("status") in ("pass", "fail"):
                icon = "✅" if cc["status"] == "pass" else "❌"
                sections.append(
                    f"\n🔒 COMPLIANCE — "
                    f"{cc.get('company_name', cc.get('supplier_id'))}\n"
                    f"  Result         : {icon} {cc['status'].upper()}\n"
                    f"  CDSCO Approved : {'✅' if cc['cdsco_approved'] else '❌'}\n"
                    f"  GMP Certified  : {'✅' if cc['gmp_certified'] else '❌'}\n"
                    f"  Cold Chain     : {'✅' if cc['cold_chain'] else '❌'}\n"
                    f"  Score          : {cc['compliance_score']}/100"
                )
                if cc.get("issues"):
                    sections.append("  Issues: " + "; ".join(cc["issues"]))

        # Price comparison
        if "price_compare" in tool_results:
            pc = tool_results["price_compare"]
            if pc.get("status") == "ok":
                sections.append(
                    "\n💰 PRICE COMPARISON (lower score = better value):"
                )
                for s in pc["ranked"]:
                    sections.append(
                        f"  #{s['rank']} {s['name']}: "
                        f"₹{s['price']}/unit | {s['lead_time']}d lead | "
                        f"{s['reliability']}% reliability | score={s['score']}"
                    )

        # Draft order
        if "draft_order" in tool_results:
            do = tool_results["draft_order"]
            if do.get("status") == "ok":
                sections.append(f"\n📋 DRAFT PURCHASE ORDER: {do['po_number']}")
                sections.append("  ⚠️ Requires human approval before submission")

        return "\n".join(sections)

    # ──────────────────────────────────────────────────────────────────────
    # REPLANNING LOOP
    # FIX v4: self.memory reset per run() — was accumulating across queries
    #         attempt initialised before loop — avoids UnboundLocalError
    # ──────────────────────────────────────────────────────────────────────
    def run(self, query: str) -> dict:
        print(f"\n{'='*60}")
        print(f"🧭 Orchestrator running: {query[:60]}...")
        print(f"{'='*60}")

        # FIX v4: reset memory per query so unrelated queries don't pollute
        self.memory  = []
        last_results = {}
        last_answer  = ""
        plan         = {}
        attempt      = 0   # FIX v4: initialise before loop

        for attempt in range(1, self.MAX_REPLANS + 1):
            print(f"\n  Attempt {attempt}/{self.MAX_REPLANS}")

            query_with_memory = query
            if attempt > 1 and self.memory:
                query_with_memory = (
                    query + f" [Previous attempt found: {self.memory[-1]}]"
                )

            plan         = self._plan(query_with_memory)
            tool_results = self._execute(query, plan)
            answer       = self._assemble(query, tool_results)

            sr = tool_results.get("supplier_search", {})
            if sr.get("status") == "ok" and sr.get("suppliers"):
                print(f"  ✅ Good results on attempt {attempt}")
                last_results = tool_results
                last_answer  = answer
                break
            elif attempt < self.MAX_REPLANS:
                print(f"  ⚠️ No suppliers found, replanning...")
                self.memory.append(
                    f"Attempt {attempt}: no qualified suppliers for '{query}'"
                )
                last_results = tool_results
                last_answer  = answer
            else:
                print(f"  ❌ Max replans reached")
                last_results = tool_results
                last_answer  = answer

        draft = (
            last_results.get("draft_order")
            if last_results.get("draft_order", {}).get("status") == "ok"
            else None
        )

        return {
            "query"        : query,
            "plan"         : plan,
            "tool_results" : last_results,
            "final_answer" : last_answer,
            "attempts"     : attempt,
            "draft_order"  : draft,
        }

    # ──────────────────────────────────────────────────────────────────────
    # HELPERS
    # ──────────────────────────────────────────────────────────────────────
    def _extract_product(self, query: str) -> str:
        pharma_products = [
            'insulin', 'amoxicillin', 'paracetamol', 'vaccine', 'antibiotic',
            'oncology', 'cardiac', 'respiratory', 'pain relief', 'generic',
            'api', 'active pharmaceutical'
        ]
        q = query.lower()
        for p in pharma_products:
            if p in q:
                return p
        words = [w for w in query.split() if len(w) > 4 and w.isalpha()]
        return words[0].lower() if words else "pharmaceutical"

    def _extract_keyword(self, query: str) -> str:
        disruption_words = [
            'china', 'india', 'antibiotic', 'insulin', 'vaccine', 'shortage',
            'ban', 'block', 'recall', 'contamination', 'supply chain', 'export'
        ]
        q = query.lower()
        for w in disruption_words:
            if w in q:
                return w
        return query.split()[0].lower()

    def _extract_supplier_id(self, query: str) -> str:
        match = re.search(r'PHARM-\d{4}', query, re.IGNORECASE)
        return match.group(0).upper() if match else ""


# ══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# python app/orchestrator.py
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("Loading agent...")
    from app.simple_agent_improved import PharmaSupplyChainAgent
    agent = PharmaSupplyChainAgent()

    orc = Orchestrator(agent=agent)

    test_queries = [
        "Find CDSCO approved insulin suppliers in India urgently.",
        "China blocks antibiotic exports. Check news and find alternatives.",
        "Check compliance for PHARM-0072 and compare prices with PHARM-0053.",
    ]

    for q in test_queries:
        result = orc.run(q)
        print("\n" + "="*60)
        print("FINAL ANSWER:")
        print(result["final_answer"])
        print(f"Tools used: {result['plan'].get('tools_to_call', [])}")
        print(f"Attempts  : {result['attempts']}")
        input("\nPress Enter for next query...")
