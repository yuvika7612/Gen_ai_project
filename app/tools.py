"""
app/tools.py
MediCare Pharmaceuticals India — Agentic Tool Belt
---------------------------------------------------
Six standalone tool functions the orchestrator can call.

IMPORTANT DESIGN RULES:
  - Tools are pure Python — NO LLM calls
  - Tools read from files/CSV directly — zero hallucination risk
  - Each tool returns a plain dict — easy for orchestrator to parse
  - Tools never import from each other — fully independent

Fixes v3:
  - supplier_search: reads 'reliability_score' OR 'reliability' key
  - supplier_search: reads 'city'/'location' for location field
  - supplier_search: reads 'lead_time' OR 'lead_time_days' key
  - supplier_search: reads 'price' OR 'unit_price_inr' key

FIX v4:
  - supplier_search: accepts blocked_country param — excludes suppliers
    from a country that is banned/blocking exports in the query
  - This fixes the China-ban query returning Chinese suppliers as alternatives

Tools:
  supplier_search()   — full RAG pipeline via existing agent
  inventory_check()   — reads current_inventory.json
  news_monitor()      — searches GDELT CSV for disruption alerts
  compliance_check()  — verifies CDSCO + cold chain from CSV
  price_compare()     — ranks suppliers by weighted price+lead time score
  draft_order()       — generates draft purchase order text
"""

import json
import os
import pandas as pd
from datetime import datetime
from pathlib import Path

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT            = Path(__file__).resolve().parent.parent
INVENTORY_PATH  = ROOT / "data" / "company" / "current_inventory.json"
SUPPLIERS_CSV   = ROOT / "data" / "suppliers" / "pharma_suppliers.csv"
GDELT_CSV       = ROOT / "data" / "gdelt" / "gdelt_pharma_clean.csv"

# ── Lazy-load suppliers CSV once ───────────────────────────────────────────
_suppliers_df = None

def _get_suppliers_df() -> pd.DataFrame:
    global _suppliers_df
    if _suppliers_df is None:
        _suppliers_df = pd.read_csv(SUPPLIERS_CSV)
    return _suppliers_df


# ── Helper: try multiple dict keys, return first non-empty value ───────────
def _meta(m: dict, *keys, default="?"):
    for k in keys:
        v = m.get(k)
        if v is not None and str(v).strip() not in ("", "nan", "None", "?"):
            return v
    return default


# ══════════════════════════════════════════════════════════════════════════
# TOOL 1: supplier_search
# Wraps the full Advanced RAG pipeline (Steps 2-4).
# ══════════════════════════════════════════════════════════════════════════
def supplier_search(query: str, agent=None, blocked_country: str = None) -> dict:
    """
    Run the full RAG pipeline and return structured supplier results.

    Args:
        query           : natural language supplier query
        agent           : PharmaSupplyChainAgent instance (passed in by orchestrator)
        blocked_country : lowercase country name to exclude from results
                          e.g. 'china' when query says "China blocks exports"

    Returns:
        {
            "status"    : "ok" | "no_results" | "error",
            "query"     : original query,
            "answer"    : full agent response string,
            "suppliers" : [{"id", "name", "lead_time", "price", "reliability",
                            "location", "cdsco", "cold_chain", "country", "category"}, ...]
        }
    """
    try:
        if agent is None:
            import sys
            sys.path.insert(0, str(ROOT))
            from app.simple_agent_improved import PharmaSupplyChainAgent
            agent = PharmaSupplyChainAgent()

        # Pass blocked_country into agent so _filter() can exclude it
        answer = agent.ask(query, blocked_country=blocked_country)

        # Build structured list from last_retrieved_docs
        # These are the QUALIFIED suppliers after filtering (set in ask())
        suppliers = []
        if hasattr(agent, 'last_retrieved_docs'):
            for doc in agent.last_retrieved_docs:
                m = doc.metadata

                suppliers.append({
                    "id"          : _meta(m, "supplier_id",    default="?"),
                    "name"        : _meta(m, "company_name",   default="Unknown"),
                    "lead_time"   : _meta(m, "lead_time",      "lead_time_days",  default="?"),
                    "price"       : _meta(m, "price",          "unit_price_inr",  default=0),
                    "reliability" : _meta(m, "reliability_score", "reliability",  default="?"),
                    "location"    : f"{_meta(m, 'city', 'location', 'headquarters', default='Unknown')}, "
                                    f"{_meta(m, 'country', default='Unknown')}",
                    "cdsco"       : m.get("cdsco_approved", False),
                    "cold_chain"  : m.get("cold_chain", False),
                    "country"     : _meta(m, "country",        default="Unknown"),
                    "category"    : _meta(m, "product_category", default="Unknown"),
                })

        status = "no_results" if not suppliers else "ok"
        return {"status": status, "query": query, "answer": answer, "suppliers": suppliers}

    except Exception as e:
        return {"status": "error", "query": query, "answer": "", "suppliers": [], "error": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# TOOL 2: inventory_check
# ══════════════════════════════════════════════════════════════════════════
def inventory_check(product: str) -> dict:
    """
    Check current inventory levels for a product.

    Args:
        product : product name or keyword (e.g. "insulin", "amoxicillin")

    Returns:
        {
            "status"         : "found" | "not_found" | "error",
            "product_name"   : str,
            "current_stock"  : int,
            "days_of_supply" : int,
            "urgency"        : "HIGH" | "MEDIUM" | "LOW",
            "reorder_needed" : bool
        }
    """
    try:
        with open(INVENTORY_PATH) as f:
            inventory = json.load(f)

        product_lower = product.lower()
        products      = inventory.get("products", {})

        matched_key  = None
        matched_item = None

        for key, item in products.items():
            if (product_lower in key.lower() or
                product_lower in item.get("product_name", "").lower() or
                product_lower in item.get("category", "").lower()):
                matched_key  = key
                matched_item = item
                break

        if not matched_item:
            return {
                "status"  : "not_found",
                "product" : product,
                "message" : f"'{product}' not found in inventory. "
                            f"Available: {list(products.keys())}"
            }

        days    = matched_item.get("days_of_supply", 0)
        urgency = matched_item.get("urgency", "LOW")
        reorder = days < 30 or urgency in ["HIGH", "MEDIUM"]

        return {
            "status"         : "found",
            "product"        : matched_key,
            "product_name"   : matched_item.get("product_name", matched_key),
            "current_stock"  : matched_item.get("current_stock", 0),
            "unit"           : matched_item.get("unit", "units"),
            "monthly_demand" : matched_item.get("monthly_demand", 0),
            "days_of_supply" : days,
            "stock_status"   : matched_item.get("status", "UNKNOWN"),
            "urgency"        : urgency,
            "reorder_needed" : reorder,
            "suppliers"      : matched_item.get("suppliers", []),
            "last_updated"   : inventory.get("last_updated", "unknown"),
        }

    except FileNotFoundError:
        return {"status": "error", "product": product,
                "message": f"Inventory file not found: {INVENTORY_PATH}"}
    except Exception as e:
        return {"status": "error", "product": product, "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# TOOL 3: news_monitor
# ══════════════════════════════════════════════════════════════════════════
def news_monitor(keyword: str, max_results: int = 5) -> dict:
    """
    Search GDELT pharma news for disruption alerts related to keyword.
    """
    try:
        if not GDELT_CSV.exists():
            return {
                "status"  : "error",
                "keyword" : keyword,
                "message" : f"GDELT file not found: {GDELT_CSV}",
                "alerts"  : []
            }

        df            = pd.read_csv(GDELT_CSV)
        keyword_lower = keyword.lower()
        text_cols     = [c for c in df.columns if df[c].dtype == object]
        mask          = pd.Series([False] * len(df))

        for col in text_cols:
            mask = mask | df[col].str.lower().str.contains(keyword_lower, na=False)

        matches = df[mask].head(max_results)

        if matches.empty:
            return {
                "status"      : "no_results",
                "keyword"     : keyword,
                "total_found" : 0,
                "alerts"      : [],
                "message"     : f"No news found for '{keyword}' in GDELT dataset."
            }

        alerts = []
        for _, row in matches.iterrows():
            alert = {}
            for field in ['title', 'headline', 'url', 'date', 'source',
                          'sourceurl', 'tone', 'domain']:
                if field in row and pd.notna(row[field]):
                    alert[field] = str(row[field])
            alerts.append(alert)

        return {
            "status"      : "ok",
            "keyword"     : keyword,
            "total_found" : int(mask.sum()),
            "alerts"      : alerts
        }

    except Exception as e:
        return {"status": "error", "keyword": keyword, "message": str(e), "alerts": []}


# ══════════════════════════════════════════════════════════════════════════
# TOOL 4: compliance_check
# ══════════════════════════════════════════════════════════════════════════
def compliance_check(supplier_id: str, requires_cold_chain: bool = False) -> dict:
    """
    Verify regulatory compliance for a given supplier.
    Reads directly from pharma_suppliers.csv — no LLM.
    """
    try:
        df  = _get_suppliers_df()
        row = df[df['supplier_id'] == supplier_id]

        if row.empty:
            return {
                "status"      : "not_found",
                "supplier_id" : supplier_id,
                "message"     : f"Supplier {supplier_id} not found in database."
            }

        row        = row.iloc[0]
        cdsco      = bool(row.get('cdsco_approved', False))
        gmp        = bool(row.get('gmp_certified', False))
        cold_chain = bool(row.get('cold_chain_capable', False))
        certs      = str(row.get('quality_certifications', 'None'))

        issues = []
        score  = 100

        if not cdsco:
            issues.append("Not CDSCO approved — cannot supply regulated drugs in India")
            score -= 40
        if not gmp:
            issues.append("Not GMP certified — quality manufacturing not verified")
            score -= 20
        if requires_cold_chain and not cold_chain:
            issues.append("No cold chain capability — cannot supply temperature-sensitive products")
            score -= 40

        return {
            "status"              : "pass" if not issues else "fail",
            "supplier_id"         : supplier_id,
            "company_name"        : str(row.get('company_name', 'Unknown')),
            "country"             : str(row.get('country', 'Unknown')),
            "cdsco_approved"      : cdsco,
            "gmp_certified"       : gmp,
            "cold_chain"          : cold_chain,
            "cold_chain_required" : requires_cold_chain,
            "certifications"      : certs,
            "compliance_score"    : max(0, score),
            "issues"              : issues,
        }

    except Exception as e:
        return {"status": "error", "supplier_id": supplier_id, "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# TOOL 5: price_compare
# ══════════════════════════════════════════════════════════════════════════
def price_compare(supplier_ids: list,
                  price_weight: float = 0.5,
                  lead_time_weight: float = 0.3,
                  reliability_weight: float = 0.2) -> dict:
    """
    Rank suppliers by weighted score of price, lead time, reliability.
    Lower score = better value.
    """
    try:
        df   = _get_suppliers_df()
        rows = df[df['supplier_id'].isin(supplier_ids)].copy()

        if rows.empty:
            return {"status": "error", "message": "No matching suppliers found.", "ranked": []}

        price_min, price_max = rows['unit_price_inr'].min(), rows['unit_price_inr'].max()
        lead_min,  lead_max  = rows['lead_time_days'].min(),  rows['lead_time_days'].max()
        rel_min,   rel_max   = rows['reliability_score'].min(), rows['reliability_score'].max()

        def norm(val, lo, hi):
            return 0.0 if hi == lo else (val - lo) / (hi - lo)

        ranked = []
        for _, row in rows.iterrows():
            price_norm = norm(row['unit_price_inr'],    price_min, price_max)
            lead_norm  = norm(row['lead_time_days'],    lead_min,  lead_max)
            rel_norm   = 1 - norm(row['reliability_score'], rel_min, rel_max)

            score = (price_weight       * price_norm +
                     lead_time_weight   * lead_norm  +
                     reliability_weight * rel_norm)

            ranked.append({
                "supplier_id" : row['supplier_id'],
                "name"        : row['company_name'],
                "price"       : float(row['unit_price_inr']),
                "lead_time"   : int(row['lead_time_days']),
                "reliability" : float(row['reliability_score']),
                "score"       : round(score, 4),
            })

        ranked.sort(key=lambda x: x['score'])
        for i, s in enumerate(ranked, 1):
            s['rank'] = i

        return {
            "status" : "ok",
            "ranked" : ranked,
            "note"   : "Lower score = better (price=50%, lead_time=30%, reliability=20%)"
        }

    except Exception as e:
        return {"status": "error", "message": str(e), "ranked": []}


# ══════════════════════════════════════════════════════════════════════════
# TOOL 6: draft_order
# ══════════════════════════════════════════════════════════════════════════
def draft_order(supplier_id: str,
                product: str,
                quantity: int,
                unit: str = "units",
                urgency: str = "STANDARD") -> dict:
    """
    Generate a draft purchase order for human review.
    Does NOT place any real order — output requires human approval.
    """
    try:
        df  = _get_suppliers_df()
        row = df[df['supplier_id'] == supplier_id]

        if row.empty:
            return {
                "status"      : "not_found",
                "supplier_id" : supplier_id,
                "message"     : f"Supplier {supplier_id} not found."
            }

        row         = row.iloc[0]
        unit_price  = float(row.get('unit_price_inr', 0))
        total_value = unit_price * quantity
        lead_time   = int(row.get('lead_time_days', 0))
        today       = datetime.now()
        po_number   = f"DRAFT-PO-{today.strftime('%Y%m%d%H%M%S')}"

        po_text = f"""
╔══════════════════════════════════════════════════════════════╗
║          MEDICARE PHARMACEUTICALS INDIA                      ║
║          DRAFT PURCHASE ORDER — PENDING HUMAN APPROVAL       ║
╚══════════════════════════════════════════════════════════════╝

PO Number    : {po_number}
Date         : {today.strftime('%d %B %Y')}
Status       : ⚠️  DRAFT — Requires approval before submission
Urgency      : {urgency}

── SUPPLIER ─────────────────────────────────────────────────────
Supplier ID  : {supplier_id}
Company      : {row.get('company_name', 'Unknown')}
Country      : {row.get('country', 'Unknown')}
City         : {row.get('city', row.get('location', 'Unknown'))}
Contact      : {row.get('contact_email', 'N/A')}
Phone        : {row.get('phone', 'N/A')}
CDSCO        : {'✅ Approved' if row.get('cdsco_approved') else '❌ Not approved'}
GMP          : {'✅ Certified' if row.get('gmp_certified') else '❌ Not certified'}

── ORDER DETAILS ────────────────────────────────────────────────
Product      : {product}
Quantity     : {quantity:,} {unit}
Unit Price   : ₹{unit_price:,.2f}
Total Value  : ₹{total_value:,.2f}
Lead Time    : {lead_time} days
Payment      : {row.get('payment_terms', 'To be negotiated')}

── COMPLIANCE NOTE ──────────────────────────────────────────────
This is a system-generated draft. Before approval verify:
  □ Supplier CDSCO certificate is current
  □ Product batch release time is acceptable
  □ Cold chain requirements are confirmed with supplier
  □ Finance team has approved budget

── APPROVAL REQUIRED ────────────────────────────────────────────
This draft must be reviewed and approved by an authorised
MediCare procurement officer before submission to supplier.
Automated orders are NOT permitted by company policy.
══════════════════════════════════════════════════════════════════
"""

        return {
            "status"           : "ok",
            "po_number"        : po_number,
            "supplier_id"      : supplier_id,
            "supplier_name"    : str(row.get('company_name', 'Unknown')),
            "product"          : product,
            "quantity"         : quantity,
            "unit_price"       : unit_price,
            "total_value"      : total_value,
            "lead_time_days"   : lead_time,
            "po_text"          : po_text.strip(),
            "requires_approval": True,
        }

    except Exception as e:
        return {"status": "error", "supplier_id": supplier_id, "message": str(e)}


# ══════════════════════════════════════════════════════════════════════════
# QUICK TEST
# python app/tools.py
# ══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    import pprint
    pp = pprint.PrettyPrinter(indent=2)

    print("\n" + "="*60)
    print("TOOL 2: inventory_check('insulin')")
    print("="*60)
    pp.pprint(inventory_check("insulin"))

    print("\n" + "="*60)
    print("TOOL 3: news_monitor('antibiotic')")
    print("="*60)
    pp.pprint(news_monitor("antibiotic"))

    print("\n" + "="*60)
    print("TOOL 4: compliance_check('PHARM-0072', requires_cold_chain=True)")
    print("="*60)
    pp.pprint(compliance_check("PHARM-0072", requires_cold_chain=True))

    print("\n" + "="*60)
    print("TOOL 5: price_compare(['PHARM-0072', 'PHARM-0053', 'PHARM-0054'])")
    print("="*60)
    pp.pprint(price_compare(["PHARM-0072", "PHARM-0053", "PHARM-0054"]))

    print("\n" + "="*60)
    print("TOOL 6: draft_order('PHARM-0072', 'Insulin Glargine', 10000)")
    print("="*60)
    result = draft_order("PHARM-0072", "Insulin Glargine 100 IU/mL", 10000)
    print(result["po_text"])

    print("\nNote: Tool 1 (supplier_search) not tested here — loads full agent.")
    print("Test via: python app/orchestrator.py")
