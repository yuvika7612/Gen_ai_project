"""
Download pharmaceutical supply chain news from GDELT API
Focus on pharma-specific keywords: drug shortages, recalls, API, vaccines, etc.
"""

import requests
import pandas as pd
import time
from datetime import datetime

def download_pharma_news():
    """
    Download pharma-specific news from GDELT using their API
    NO WEB SCRAPING - official API access
    """

    print("🌐 Downloading pharmaceutical news from GDELT API...\n")

    # GDELT DOC 2.0 API endpoint
    api_url = "https://api.gdeltproject.org/api/v2/doc/doc"

    # Pharmaceutical-specific search queries
    pharma_queries = [
        # Drug shortages
        'pharmaceutical shortage',
        'drug shortage',
        'medicine shortage',
        'API shortage',

        # Recalls and quality
        'drug recall',
        'pharmaceutical recall',
        'medication contamination',
        'counterfeit drugs',

        # Supply chain
        'pharmaceutical supply chain',
        'drug manufacturing',
        'API export restrictions',
        'pharmaceutical ingredients',

        # Specific drugs
        'insulin shortage',
        'vaccine shortage',
        'antibiotic shortage',
        'cancer drug shortage',

        # Cold chain
        'vaccine cold chain',
        'temperature-controlled pharmaceutical',

        # Regulatory
        'FDA approval',
        'drug price control',
        'pharmaceutical regulation',

        # Geographic
        'China pharmaceutical export',
        'India generic drugs',

        # Companies
        'Pfizer supply',
        'Moderna vaccine',
        'AstraZeneca shortage',
        'Cipla',
        'Sun Pharma'
    ]

    all_articles = []

    for idx, query in enumerate(pharma_queries, 1):
        print(f"📡 Query {idx}/{len(pharma_queries)}: '{query}'")

        params = {
            'query': query,
            'mode': 'artlist',
            'maxrecords': 100,  # Get up to 100 articles per query
            'format': 'json',
            'startdatetime': '20230101000000',  # Jan 1, 2023
            'enddatetime': '20240430235959',    # Apr 30, 2024
            'sourcelang': 'eng'  # English only
        }

        try:
            response = requests.get(api_url, params=params, timeout=60)

            if response.status_code == 200:
                data = response.json()
                articles = data.get('articles', [])

                if articles:
                    print(f"   ✅ Found {len(articles)} articles")
                    all_articles.extend(articles)
                else:
                    print(f"   ⚠️  No articles found")

                # Be nice to the API - wait between requests
                time.sleep(2)
            else:
                print(f"   ❌ Failed: HTTP {response.status_code}")

        except Exception as e:
            print(f"   ❌ Error: {str(e)[:50]}")

    print(f"\n📊 Total articles collected: {len(all_articles)}")

    if not all_articles:
        print("❌ No articles downloaded. Check your internet connection.")
        return None

    # Convert to DataFrame
    df = pd.DataFrame(all_articles)

    # Remove duplicates (same article from different queries)
    initial_count = len(df)
    df = df.drop_duplicates(subset=['url'], keep='first')
    print(f"📊 Removed {initial_count - len(df)} duplicates")

    # Extract and clean relevant columns
    df_clean = pd.DataFrame({
        'date': df['seendate'].str[:8],  # YYYYMMDD format
        'headline': df['title'],
        'source_domain': df['domain'],
        'url': df['url'],
        'language': df.get('language', 'eng'),
        'social_image': df.get('socialimage', ''),
        'raw_seendate': df['seendate']
    })

    # Convert date to readable format
    df_clean['date'] = pd.to_datetime(df_clean['date'], format='%Y%m%d', errors='coerce')
    df_clean = df_clean.dropna(subset=['date'])  # Remove rows with invalid dates
    df_clean['date'] = df_clean['date'].dt.strftime('%Y-%m-%d')

    # Sort by date (most recent first)
    df_clean = df_clean.sort_values('date', ascending=False)

    # Save raw data
    import os
    os.makedirs('data/gdelt', exist_ok=True)

    output_file = 'data/gdelt/gdelt_pharma_raw.csv'
    df_clean.to_csv(output_file, index=False)

    print(f"\n✅ Download complete!")
    print(f"   Total unique articles: {len(df_clean)}")
    print(f"   Date range: {df_clean['date'].min()} to {df_clean['date'].max()}")
    print(f"   Unique sources: {df_clean['source_domain'].nunique()}")
    print(f"\n📁 Saved to: {output_file}")

    # Show sample headlines
    print(f"\n📰 Sample pharmaceutical headlines:")
    for idx, row in df_clean.head(10).iterrows():
        headline_preview = row['headline'][:80] + '...' if len(row['headline']) > 80 else row['headline']
        print(f"   {idx+1}. [{row['date']}] {headline_preview}")

    print(f"\n🔝 Top news sources:")
    print(df_clean['source_domain'].value_counts().head(10))

    return df_clean

if __name__ == "__main__":
    download_pharma_news()
