"""
Generate pharmaceutical supplier database
These are API manufacturers, finished drug suppliers, and logistics partners
"""

from faker import Faker
import pandas as pd
import random

fake = Faker()

def generate_pharma_suppliers(n=100):
    """
    Generate realistic pharmaceutical suppliers
    """

    suppliers = []

    # Product categories for pharma
    products = [
        "Active Pharmaceutical Ingredients (API)",
        "Finished Antibiotics",
        "Finished Diabetes Medications",
        "Finished Cardiac Drugs",
        "Vaccines",
        "Pain Relief Medications",
        "Respiratory Medicines",
        "Oncology Drugs",
        "Generic Essential Drugs",
        "Cold Chain Logistics Services"
    ]

    # Countries (realistic pharma manufacturing hubs)
    countries = [
        "India", "India", "India", "India",  # High domestic manufacturing
        "China", "China", "China",  # Major API source
        "USA", "Germany", "Switzerland", "UK", "France",  # High-quality finished goods
        "Denmark", "Belgium", "Ireland",  # Biotech hubs
        "Israel", "South Korea", "Japan",  # Advanced pharma
        "Bangladesh", "Vietnam"  # Emerging manufacturers
    ]

    # Regulatory certifications
    certifications = [
        "WHO-GMP", "US-FDA Approved", "EU-GMP", "ISO 13485",
        "CDSCO Approved", "TGA Australia", "MHRA UK"
    ]

    for i in range(n):
        country = random.choice(countries)
        product_category = random.choice(products)

        # Adjust pricing based on country and product type
        base_price = {
            "Active Pharmaceutical Ingredients (API)": random.uniform(500, 5000),
            "Finished Antibiotics": random.uniform(2, 50),
            "Finished Diabetes Medications": random.uniform(100, 800),
            "Finished Cardiac Drugs": random.uniform(1, 100),
            "Vaccines": random.uniform(150, 800),
            "Pain Relief Medications": random.uniform(0.5, 20),
            "Respiratory Medicines": random.uniform(50, 300),
            "Oncology Drugs": random.uniform(5000, 50000),
            "Generic Essential Drugs": random.uniform(0.5, 10),
            "Cold Chain Logistics Services": random.uniform(5, 50)  # per kg
        }.get(product_category, random.uniform(10, 500))

        # Quality premium for developed countries
        country_multiplier = {
            "Switzerland": 1.8, "Germany": 1.6, "USA": 1.5,
            "Denmark": 1.5, "UK": 1.4, "France": 1.3,
            "India": 1.0, "China": 0.85, "Bangladesh": 0.75
        }.get(country, 1.0)

        unit_price = round(base_price * country_multiplier, 2)

        # Stock levels (higher for API, lower for specialized drugs)
        if "API" in product_category:
            stock_range = (10000, 100000)  # kg
            unit = "kg"
        elif "Vaccine" in product_category or "Oncology" in product_category:
            stock_range = (1000, 10000)  # doses
            unit = "doses"
        elif "Logistics" in product_category:
            stock_range = (0, 0)  # Service, not product
            unit = "service"
        else:
            stock_range = (50000, 500000)  # tablets/doses
            unit = "units"

        current_stock = random.randint(*stock_range) if stock_range != (0, 0) else 0

        # Lead times (longer for international, shorter for domestic)
        lead_time = {
            "India": random.randint(3, 10),
            "China": random.randint(14, 30),
            "Bangladesh": random.randint(10, 21),
            "Vietnam": random.randint(14, 28)
        }.get(country, random.randint(21, 45))  # International default

        # Cold chain capability
        cold_chain_capable = random.choice([True, False]) if "Vaccine" not in product_category else True

        # Quality certifications (developed countries have more)
        num_certs = 3 if country in ["USA", "Germany", "Switzerland", "Denmark"] else random.randint(1, 2)
        supplier_certs = random.sample(certifications, min(num_certs, len(certifications)))

        supplier = {
            "supplier_id": f"PHARM-{i+1:04d}",
            "company_name": fake.company() + random.choice([" Pharma", " Laboratories", " Life Sciences", " Biotech", " Healthcare"]),
            "country": country,
            "city": fake.city(),
            "product_category": product_category,
            "unit": unit,

            # Inventory
            "current_stock": current_stock,
            "max_capacity_monthly": random.randint(int(current_stock * 1.5), int(current_stock * 3)) if current_stock > 0 else 0,
            "unit_price_inr": unit_price,

            # Logistics
            "lead_time_days": lead_time,
            "minimum_order_quantity": random.choice([500, 1000, 5000, 10000]) if current_stock > 0 else 0,
            "cold_chain_capable": cold_chain_capable,
            "temperature_range": "2-8°C" if cold_chain_capable else "15-25°C",

            # Quality & Compliance
            "quality_certifications": ", ".join(supplier_certs),
            "cdsco_approved": random.choice([True, True, True, False]),  # 75% approved
            "gmp_certified": random.choice([True, True, False]),  # 67% certified
            "reliability_score": round(random.uniform(70, 99), 1),
            "on_time_delivery_rate": round(random.uniform(75, 98), 1),

            # API-specific details
            "api_purity": f"{random.uniform(98.5, 99.9):.1f}%" if "API" in product_category else "N/A",
            "batch_release_time": f"{random.randint(3, 10)} days" if "API" in product_category else "N/A",

            # Payment & Risk
            "payment_terms": random.choice(["Net 30", "Net 45", "Net 60", "LC at sight", "Advance payment"]),
            "credit_rating": random.choice(["AAA", "AA", "A", "BBB", "BB"]),
            "political_risk": "Low" if country in ["USA", "Germany", "Switzerland", "UK"] else random.choice(["Low", "Medium", "High"]),

            # Contact
            "contact_email": fake.company_email(),
            "phone": fake.phone_number(),
            "business_hours": "9 AM - 6 PM IST" if country == "India" else "24/7 (international)",

            # Special capabilities
            "emergency_supply_available": random.choice([True, False]),
            "contract_manufacturing": random.choice([True, False]) if "Finished" in product_category else False,
            "regulatory_filing_support": random.choice([True, False])
        }

        suppliers.append(supplier)

    return pd.DataFrame(suppliers)

if __name__ == "__main__":
    import os
    os.makedirs('data/suppliers', exist_ok=True)

    df = generate_pharma_suppliers(100)
    df.to_csv("data/suppliers/pharma_suppliers.csv", index=False)

    print(f"✅ Generated {len(df)} pharmaceutical suppliers")
    print(f"\n📊 Breakdown:")
    print(f"   Product categories: {df['product_category'].nunique()}")
    print(f"   Countries: {df['country'].nunique()}")
    print(f"   Cold chain capable: {df['cold_chain_capable'].sum()}")
    print(f"   CDSCO approved: {df['cdsco_approved'].sum()}")
    print(f"   GMP certified: {df['gmp_certified'].sum()}")

    print(f"\n🌍 Top supplier countries:")
    print(df['country'].value_counts().head(5))

    print(f"\n💊 Product distribution:")
    print(df['product_category'].value_counts())

    print(f"\n📁 Saved to: data/suppliers/pharma_suppliers.csv")
