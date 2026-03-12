"""
Create MediCare Pharmaceuticals company profile
This defines who "we" are and what inventory we manage
"""

import json
import os

def create_company_profile():
    """
    MediCare Pharmaceuticals India - Mid-sized pharma distributor
    """

    company = {
        "company_name": "MediCare Pharmaceuticals India Pvt. Ltd.",
        "business_type": "Pharmaceutical Distributor & Supply Chain Manager",
        "headquarters": "Bangalore, Karnataka, India",
        "founded": "2015",
        "annual_revenue": "₹500 crore (~$60M USD)",
        "employees": 450,

        "business_description": "MediCare is a mid-sized pharmaceutical distributor serving hospitals, clinics, and pharmacies across India. We specialize in ensuring continuous supply of critical medications, managing cold chain logistics, and maintaining regulatory compliance with CDSCO standards.",

        "customer_base": {
            "hospitals": 2500,
            "pharmacies": 8000,
            "clinics": 5000,
            "total_customers": 15500
        },

        "geographic_coverage": [
            "Karnataka", "Tamil Nadu", "Andhra Pradesh", "Telangana",
            "Maharashtra", "Gujarat", "Delhi NCR", "Uttar Pradesh"
        ],

        "product_categories": [
            "Antibiotics",
            "Diabetes Medications",
            "Cardiac Medications",
            "Vaccines",
            "Pain Relief & Analgesics",
            "Respiratory Medicines",
            "Cancer Treatment Drugs",
            "Generic Essential Drugs"
        ],

        "regulatory_compliance": {
            "cdsco_license": "Yes - Valid until 2027",
            "gmp_certified": "Yes",
            "gdp_certified": "Yes - WHO Good Distribution Practices",
            "iso_certification": "ISO 9001:2015",
            "drug_licenses": ["Schedule H", "Schedule H1", "Schedule X"]
        },

        "infrastructure": {
            "warehouses": 5,
            "cold_storage_facilities": 3,
            "distribution_centers": 12,
            "cold_chain_vehicles": 45,
            "total_storage_capacity_sqft": 150000
        },

        "supply_chain_policy": {
            "safety_stock_critical_drugs": "90 days (CDSCO requirement)",
            "safety_stock_essential_drugs": "60 days",
            "safety_stock_general_drugs": "30 days",
            "reorder_trigger": "When stock falls below safety stock + 15 days",
            "shortage_reporting": "Within 24 hours to CDSCO if <30 days stock"
        },

        "supplier_strategy": {
            "primary_api_source": "China (40%) - Active Pharmaceutical Ingredients",
            "domestic_manufacturers": "India (45%) - Formulation & packaging",
            "international_finished_goods": "Europe (10%), USA (5%)",
            "max_single_supplier_dependency": "25% for critical drugs"
        },

        "financial_metrics": {
            "inventory_value": "₹350 crore ($42M)",
            "monthly_revenue": "₹40 crore ($4.8M)",
            "inventory_turnover_days": 45,
            "gross_margin": "18%",
            "operating_margin": "8%"
        }
    }

    # Create directory
    os.makedirs('data/company', exist_ok=True)

    # Save as JSON
    with open('data/company/company_profile.json', 'w') as f:
        json.dump(company, f, indent=2)

    print("✅ Created company profile: MediCare Pharmaceuticals India")
    print(f"📊 Business: {company['business_type']}")
    print(f"📍 Location: {company['headquarters']}")
    print(f"💰 Revenue: {company['annual_revenue']}")
    print(f"🏥 Customers: {company['customer_base']['total_customers']:,}")
    print(f"📦 Product Categories: {len(company['product_categories'])}")
    print(f"\n📁 Saved to: data/company/company_profile.json")

if __name__ == "__main__":
    create_company_profile()
