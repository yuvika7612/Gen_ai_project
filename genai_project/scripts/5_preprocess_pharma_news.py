"""
Clean and categorize pharmaceutical news data
Filter for relevance, remove junk, categorize by pharma event type
"""

import pandas as pd
import re
import os

def preprocess_pharma_news():
    """
    Clean downloaded pharma news data
    """

    print("🧹 Preprocessing pharmaceutical news data...\n")

    # Load raw data
    input_file = 'data/gdelt/gdelt_pharma_raw.csv'

    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Run download script first.")
        return None

    df = pd.read_csv(input_file)
    print(f"📥 Loaded {len(df)} raw articles")

    original_count = len(df)

    # ============================================================
    # STEP 1: Remove duplicates
    # ============================================================
    df = df.drop_duplicates(subset=['headline'], keep='first')
    print(f"✓ Step 1: Removed {original_count - len(df)} duplicate headlines")

    # ============================================================
    # STEP 2: Clean text
    # ============================================================
    def clean_text(text):
        if pd.isna(text):
            return ""
        # Remove HTML entities
        text = text.replace('&amp;', '&').replace('&quot;', '"')
        text = text.replace('&lt;', '<').replace('&gt;', '>')
        text = text.replace('&#39;', "'")
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    df['headline'] = df['headline'].apply(clean_text)
    print(f"✓ Step 2: Text cleaning complete")

    # ============================================================
    # STEP 3: Remove short/junk headlines
    # ============================================================
    before_length = len(df)
    df = df[df['headline'].str.len() >= 30]  # At least 30 characters
    print(f"✓ Step 3: Removed {before_length - len(df)} short headlines")

    # ============================================================
    # STEP 4: Filter by pharmaceutical relevance
    # ============================================================
    pharma_keywords = [
        'drug', 'medicine', 'pharmaceutical', 'pharma', 'medication',
        'vaccine', 'insulin', 'antibiotic', 'treatment', 'therapy',
        'api', 'active ingredient', 'fda', 'cdsco', 'recall',
        'shortage', 'supply', 'manufacturing', 'clinical trial',
        'generic', 'patent', 'pfizer', 'moderna', 'astrazeneca',
        'johnson & johnson', 'merck', 'novartis', 'roche',
        'cipla', 'sun pharma', 'dr reddy', 'biocon', 'lupin'
    ]

    def is_pharma_relevant(headline):
        headline_lower = headline.lower()
        return any(keyword in headline_lower for keyword in pharma_keywords)

    before_relevance = len(df)
    df = df[df['headline'].apply(is_pharma_relevant)]
    print(f"✓ Step 4: Filtered to {len(df)} pharma-relevant articles (removed {before_relevance - len(df)})")

    # ============================================================
    # STEP 5: Categorize by pharmaceutical event type
    # ============================================================
    def categorize_pharma_event(headline):
        h_lower = headline.lower()

        # Check each category
        if any(word in h_lower for word in ['shortage', 'scarce', 'running out', 'stock out']):
            return 'Drug Shortage'
        elif any(word in h_lower for word in ['recall', 'withdraw', 'contamination', 'defect']):
            return 'Product Recall/Quality'
        elif any(word in h_lower for word in ['vaccine', 'vaccination', 'immunization']):
            return 'Vaccines'
        elif any(word in h_lower for word in ['api', 'active ingredient', 'raw material']):
            return 'API/Manufacturing'
        elif any(word in h_lower for word in ['cold chain', 'temperature', 'refrigerat', 'storage']):
            return 'Cold Chain/Logistics'
        elif any(word in h_lower for word in ['fda approval', 'cdsco', 'regulatory', 'regulation']):
            return 'Regulatory'
        elif any(word in h_lower for word in ['price', 'cost', 'pricing', 'expensive']):
            return 'Pricing/Economics'
        elif any(word in h_lower for word in ['patent', 'generic', 'biosimilar']):
            return 'Patents/Generics'
        elif any(word in h_lower for word in ['clinical trial', 'study', 'research']):
            return 'Clinical/Research'
        elif any(word in h_lower for word in ['counterfeit', 'fake', 'spurious']):
            return 'Counterfeit/Quality Crisis'
        elif any(word in h_lower for word in ['export', 'import', 'trade', 'tariff']):
            return 'Trade/Geopolitics'
        else:
            return 'General Pharma News'

    df['category'] = df['headline'].apply(categorize_pharma_event)
    print(f"✓ Step 5: Categorized articles")

    # ============================================================
    # STEP 6: Add urgency scoring
    # ============================================================
    def calculate_urgency(row):
        """
        Score urgency from 1-5 based on headline keywords
        """
        headline_lower = row['headline'].lower()
        category = row['category']

        urgency = 1  # Base urgency

        # High urgency keywords
        critical_words = ['critical', 'emergency', 'urgent', 'crisis', 'severe',
                         'immediate', 'acute', 'dire']
        if any(word in headline_lower for word in critical_words):
            urgency += 2

        # Category-based urgency
        if category in ['Drug Shortage', 'Product Recall/Quality', 'Counterfeit/Quality Crisis']:
            urgency += 2
        elif category in ['Cold Chain/Logistics', 'API/Manufacturing']:
            urgency += 1

        # Geographic urgency (India-specific)
        if any(country in headline_lower for country in ['india', 'indian', 'delhi', 'mumbai', 'bangalore']):
            urgency += 1

        # Patient impact keywords
        if any(word in headline_lower for word in ['patient', 'death', 'hospital', 'life-saving']):
            urgency += 1

        return min(urgency, 5)  # Cap at 5

    df['urgency_score'] = df.apply(calculate_urgency, axis=1)
    print(f"✓ Step 6: Urgency scoring complete")

    # ============================================================
    # STEP 7: Extract key entities
    # ============================================================
    def extract_companies(headline):
        """Extract pharmaceutical company names"""
        companies = []
        pharma_companies = [
            'pfizer', 'moderna', 'astrazeneca', 'johnson', 'merck', 'novartis',
            'roche', 'sanofi', 'glaxo', 'gsk', 'abbvie', 'bristol myers',
            'cipla', 'sun pharma', 'dr reddy', 'biocon', 'lupin', 'aurobindo',
            'hetero', 'cadila', 'zydus', 'torrent'
        ]

        headline_lower = headline.lower()
        for company in pharma_companies:
            if company in headline_lower:
                companies.append(company.title())

        return ', '.join(companies) if companies else 'None'

    df['companies_mentioned'] = df['headline'].apply(extract_companies)

    def extract_countries(headline):
        """Extract country names"""
        countries = []
        country_list = [
            'india', 'china', 'usa', 'united states', 'uk', 'britain',
            'germany', 'france', 'switzerland', 'denmark', 'israel',
            'japan', 'south korea', 'bangladesh', 'vietnam'
        ]

        headline_lower = headline.lower()
        for country in country_list:
            if country in headline_lower:
                countries.append(country.title())

        return ', '.join(set(countries)) if countries else 'Global'

    df['countries_mentioned'] = df['headline'].apply(extract_countries)

    print(f"✓ Step 7: Entity extraction complete")

    # ============================================================
    # STEP 8: Remove low-value categories
    # ============================================================
    # Keep only high-value categories
    before_filter = len(df)
    valuable_categories = [
        'Drug Shortage', 'Product Recall/Quality', 'Vaccines',
        'API/Manufacturing', 'Cold Chain/Logistics', 'Regulatory',
        'Pricing/Economics', 'Patents/Generics', 'Counterfeit/Quality Crisis',
        'Trade/Geopolitics'
    ]
    df = df[df['category'].isin(valuable_categories)]
    print(f"✓ Step 8: Filtered to valuable categories (removed {before_filter - len(df)} generic news)")

    # ============================================================
    # SAVE CLEANED DATA
    # ============================================================
    output_file = 'data/gdelt/gdelt_pharma_clean.csv'
    df.to_csv(output_file, index=False)

    print(f"\n{'='*60}")
    print(f"✅ PREPROCESSING COMPLETE!")
    print(f"{'='*60}")
    print(f"\n📊 Summary:")
    print(f"   Original articles: {original_count}")
    print(f"   Final cleaned: {len(df)}")
    print(f"   Removed: {original_count - len(df)} ({((original_count - len(df))/original_count*100):.1f}%)")

    print(f"\n📁 Saved to: {output_file}")

    print(f"\n📈 Category Distribution:")
    print(df['category'].value_counts())

    print(f"\n⚡ Urgency Distribution:")
    print(df['urgency_score'].value_counts().sort_index())

    print(f"\n🏢 Top Companies Mentioned:")
    all_companies = ', '.join(df['companies_mentioned'].tolist()).split(', ')
    from collections import Counter
    company_counts = Counter([c for c in all_companies if c != 'None'])
    for company, count in company_counts.most_common(10):
        print(f"   - {company}: {count}")

    print(f"\n🌍 Top Countries Mentioned:")
    all_countries = ', '.join(df['countries_mentioned'].tolist()).split(', ')
    country_counts = Counter([c for c in all_countries if c != 'Global'])
    for country, count in country_counts.most_common(10):
        print(f"   - {country}: {count}")

    return df

if __name__ == "__main__":
    preprocess_pharma_news()
