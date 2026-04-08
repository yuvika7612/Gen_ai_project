"""
scripts/1b_create_inventory.py
Create MediCare Pharmaceuticals — current warehouse inventory snapshot

Design rules:
  - days_of_supply = current_stock / (monthly_demand / 30)
  - urgency thresholds based on CDSCO safety stock policy:
      CRITICAL  (HIGH)   : days_of_supply < 60  for critical drugs
      BELOW_TARGET (MEDIUM): days_of_supply < 75
      ADEQUATE  (LOW)    : days_of_supply >= 75
  - Products cover all 8 categories present in pharma_suppliers.csv
    so that inventory alerts map directly to supplier search queries

Run:
    python scripts/1b_create_inventory.py
"""

import json
import os
from datetime import datetime


def days_of_supply(stock: int, monthly_demand: int) -> int:
    """Calculate how many days of stock remain at current consumption rate."""
    if monthly_demand == 0:
        return 999
    return round(stock / (monthly_demand / 30))


def urgency(days: int, is_critical_drug: bool = True) -> tuple:
    """
    Return (status, urgency_level) based on CDSCO safety stock policy.
    Critical drugs: 90-day target → HIGH if <60d, MEDIUM if <75d
    Essential drugs: 60-day target → HIGH if <45d, MEDIUM if <60d
    """
    target = 90 if is_critical_drug else 60
    if days < target * 0.65:          # below 65% of target → CRITICAL
        return "CRITICAL", "HIGH"
    elif days < target * 0.85:        # below 85% of target → BELOW_TARGET
        return "BELOW_TARGET", "MEDIUM"
    else:
        return "ADEQUATE", "LOW"


def create_inventory():

    # sourcing_note explains WHY domestic or international sourcing is needed.
    # This drives the query template in streamlit_ui.py.
    # ✅ India available = domestic sourcing preferred
    # 🌍 No India CDSCO = must source internationally
    products_raw = [
        # SCENARIO 1: India has CDSCO-approved suppliers → domestic sourcing
        # Insulin — India has 2 CDSCO+cold chain suppliers
        (
            "insulin_glargine",
            "Insulin Glargine 100 IU/mL",
            "Diabetes",
            500_000, "vials", 300_000, True,
            ["Biocon", "Novo Nordisk", "Sanofi"],
            "Contact Biocon or Novo Nordisk for urgent order",
            "india_available"
        ),
        # SCENARIO 2: India has NO CDSCO-approved cardiac supplier → must go international
        # Cardiac — only non-India CDSCO suppliers exist in DB
        (
            "atorvastatin",
            "Atorvastatin 10mg Tablets",
            "Cardiac",
            800_000, "tablets", 600_000, True,
            ["Lupin Ltd", "Dr Reddys", "Torrent Pharma"],
            "No India CDSCO cardiac supplier — source internationally",
            "international_required"
        ),
        # SCENARIO 3: India available, cold chain required
        # Respiratory — India has 2 CDSCO respiratory suppliers
        (
            "salbutamol",
            "Salbutamol 100mcg Inhaler",
            "Respiratory",
            120_000, "inhalers", 90_000, True,
            ["GSK India", "Cipla Respiratory"],
            "Source from respiratory medicine suppliers urgently",
            "india_available"
        ),
        # SCENARIO 4: India available, cold chain + urgent
        # Vaccines — India has 2 CDSCO vaccine suppliers
        (
            "hepatitis_b_vaccine",
            "Hepatitis B Vaccine 1mL",
            "Vaccines",
            80_000, "vials", 50_000, True,
            ["Serum Institute", "Bharat Biotech"],
            "Contact Serum Institute or Bharat Biotech urgently",
            "india_available"
        ),
        # SCENARIO 5: Geopolitical risk — China API ban disrupts supply
        # Antibiotics — India has 1 CDSCO supplier but also China risk scenario
        (
            "amoxicillin",
            "Amoxicillin 500mg Tablets",
            "Antibiotics",
            2_000_000, "tablets", 1_000_000, True,
            ["Cipla Ltd", "Sun Pharma", "Hetero Drugs"],
            "Reorder from domestic antibiotic manufacturers; China API ban risk",
            "geopolitical_risk"
        ),
        # SCENARIO 6: International sourcing for specialty oncology drug
        # Oncology — only 1 India CDSCO supplier, may need international backup
        (
            "imatinib",
            "Imatinib 400mg Capsules",
            "Oncology",
            150_000, "capsules", 80_000, False,
            ["Natco Pharma", "Cipla Oncology"],
            "Check India first; escalate internationally if unavailable",
            "india_preferred"
        ),
        # SCENARIO 7: Adequate stock, domestic supply stable
        (
            "paracetamol",
            "Paracetamol 500mg Tablets",
            "Pain Relief",
            10_000_000, "tablets", 4_000_000, False,
            ["Generic manufacturers"],
            None,
            "india_available"
        ),
        # SCENARIO 8: Adequate stock, international specialty drug
        (
            "methotrexate",
            "Methotrexate 2.5mg Tablets",
            "Oncology",
            500_000, "tablets", 200_000, False,
            ["Pfizer India", "BDR Pharma"],
            None,
            "india_preferred"
        ),
    ]

    products = {}
    critical_alerts = []

    for pid, name, category, stock, unit, demand, is_critical, suppliers, action, sourcing in products_raw:
        dos      = days_of_supply(stock, demand)
        status, urg = urgency(dos, is_critical)

        products[pid] = {
            "product_name"  : name,
            "category"      : category,
            "current_stock" : stock,
            "unit"          : unit,
            "monthly_demand": demand,
            "days_of_supply": dos,
            "status"        : status,
            "urgency"       : urg,
            "suppliers"     : suppliers,
            "sourcing"      : sourcing,   # drives query template in UI
        }

        if urg in ("HIGH", "MEDIUM") and action:
            critical_alerts.append({
                "product"        : pid,
                "alert_type"     : "LOW_STOCK",
                "message"        : f"{name} stock {status.lower().replace('_',' ')} — {dos} days remaining",
                "urgency"        : urg,
                "action_required": action,
            })

    inventory = {
        "last_updated"         : datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warehouse_location"   : "Bangalore Main Warehouse",
        "total_inventory_value": "₹350 crore",
        "products"             : products,
        "critical_alerts"      : critical_alerts,
    }

    os.makedirs("data/company", exist_ok=True)
    out_path = "data/company/current_inventory.json"
    with open(out_path, "w") as f:
        json.dump(inventory, f, indent=2)

    print(f"✅ Inventory written to {out_path}")
    print(f"   Last updated : {inventory['last_updated']}")
    print(f"   Products     : {len(products)}")
    print(f"   Alerts       : {len(critical_alerts)}")
    print()
    print(f"{'Product':<40} {'Days':>5}  {'Urgency':<8}  Status")
    print("-" * 70)
    for pid, d in products.items():
        flag = "🔴" if d['urgency'] == 'HIGH' else ("🟡" if d['urgency'] == 'MEDIUM' else "🟢")
        print(f"{flag} {d['product_name']:<38} {d['days_of_supply']:>5}d  "
              f"{d['urgency']:<8}  {d['status']}")


if __name__ == "__main__":
    create_inventory()
