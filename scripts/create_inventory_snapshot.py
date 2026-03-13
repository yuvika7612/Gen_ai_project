"""
Create initial inventory snapshot for MediCare Pharmaceuticals
"""

import json
import os
from datetime import datetime, timedelta


def create_inventory_snapshot():

    today = datetime.now()

    inventory = {
        "last_updated": today.strftime("%Y-%m-%d %H:%M:%S"),
        "warehouse_location": "Bangalore Main Warehouse",
        "total_inventory_value": "₹350 crore",

        "products": {
            "amoxicillin": {
                "product_name": "Amoxicillin 500mg Tablets",
                "category": "Antibiotics",
                "current_stock": 2000000,
                "unit": "tablets",
                "monthly_demand": 1000000,
                "days_of_supply": 60,
                "status": "BELOW_TARGET",
                "urgency": "MEDIUM",
                "suppliers": ["Cipla Ltd", "Sun Pharma", "Hetero Drugs"]
            },

            "insulin_glargine": {
                "product_name": "Insulin Glargine 100 IU/mL",
                "category": "Diabetes",
                "current_stock": 500000,
                "unit": "vials",
                "monthly_demand": 300000,
                "days_of_supply": 50,
                "status": "CRITICAL",
                "urgency": "HIGH",
                "suppliers": ["Biocon", "Novo Nordisk", "Sanofi"]
            },

            "paracetamol": {
                "product_name": "Paracetamol 500mg Tablets",
                "category": "Pain Relief",
                "current_stock": 10000000,
                "unit": "tablets",
                "monthly_demand": 4000000,
                "days_of_supply": 75,
                "status": "ADEQUATE",
                "urgency": "LOW",
                "suppliers": ["Generic manufacturers"]
            }
        },

        "critical_alerts": [
            {
                "product": "insulin_glargine",
                "alert_type": "LOW_STOCK",
                "message": "Insulin stock below target level",
                "urgency": "HIGH",
                "action_required": "Contact Biocon for urgent order"
            }
        ]
    }

    os.makedirs("data/company", exist_ok=True)

    with open("data/company/current_inventory.json", "w") as f:
        json.dump(inventory, f, indent=2)

    print("✅ Created inventory snapshot")
    print(f"📦 Products tracked: {len(inventory['products'])}")
    print("📁 Saved to: data/company/current_inventory.json")


if __name__ == "__main__":
    create_inventory_snapshot()