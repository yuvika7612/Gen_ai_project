"""
Split cleaned pharma news into test and validation sets
"""

import pandas as pd
import os

def create_pharma_test_sets():
    """
    Create test and validation sets from cleaned pharma news
    """

    print("✂️  Creating test/validation sets for pharmaceutical news...\n")

    # Load cleaned data
    input_file = 'data/gdelt/gdelt_pharma_clean.csv'

    if not os.path.exists(input_file):
        print(f"❌ Error: {input_file} not found. Run preprocessing script first.")
        return None

    df = pd.read_csv(input_file)
    print(f"📥 Loaded {len(df)} cleaned articles")

    # Shuffle for randomness
    df = df.sample(frac=1, random_state=42).reset_index(drop=True)

    # Split: 100 for test, rest for validation
    test_size = min(100, len(df) // 2)  # Use up to 100 for testing

    test_df = df.head(test_size)
    validation_df = df.iloc[test_size:]

    # Save test set
    test_file = 'data/gdelt/gdelt_test_set.csv'
    test_df.to_csv(test_file, index=False)
    print(f"✅ Test set: {len(test_df)} articles")
    print(f"   Saved to: {test_file}")

    # Save validation set
    if len(validation_df) > 0:
        validation_file = 'data/gdelt/gdelt_validation_set.csv'
        validation_df.to_csv(validation_file, index=False)
        print(f"✅ Validation set: {len(validation_df)} articles")
        print(f"   Saved to: {validation_file}")

    # Analysis
    print(f"\n📊 Test Set Analysis:")
    print(f"\n   Categories:")
    print(test_df['category'].value_counts())

    print(f"\n   Urgency levels:")
    print(test_df['urgency_score'].value_counts().sort_index())

    print(f"\n   Date range:")
    print(f"   {test_df['date'].min()} to {test_df['date'].max()}")

    print(f"\n📰 Sample test headlines:")
    for idx, row in test_df.head(10).iterrows():
        print(f"\n   {idx+1}. [{row['category']}] Urgency: {row['urgency_score']}/5")
        print(f"      {row['headline'][:80]}...")
        print(f"      Date: {row['date']}, Source: {row['source_domain']}")

if __name__ == "__main__":
    create_pharma_test_sets()
