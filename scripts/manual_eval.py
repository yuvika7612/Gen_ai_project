"""
scripts/manual_eval.py
MediCare Pharmaceuticals India — Manual Retrieval Evaluation
-------------------------------------------------------------
Prints three metrics for each test question:

  Precision        — of the 5 retrieved suppliers, how many were actually relevant?
  Recall           — of all relevant suppliers in DB, how many did we retrieve?
  Context Relevance — do the retrieved docs contain keywords needed to answer the question?
                      (keyword overlap score — proxy for RAGAS context relevance)

Run:
    python scripts/manual_eval.py
"""

import sys
import pandas as pd

sys.path.insert(0, '.')
from app.simple_agent_improved import PharmaSupplyChainAgent

df = pd.read_csv('data/suppliers/pharma_suppliers.csv')
agent = PharmaSupplyChainAgent()

test_cases = [
    {
        'question': 'Find CDSCO approved insulin manufacturers in India with fast delivery.',
        'expected_ids': list(df[
            (df['cdsco_approved'] == True) &
            (df['cold_chain_capable'] == True) &
            (df['country'] == 'India') &
            (df['lead_time_days'] <= 21)
        ]['supplier_id']),
        # Keywords that MUST appear in retrieved docs to be considered relevant context
        'context_keywords': ['cdsco', 'india', 'diabetes', 'insulin', 'cold chain', 'lead time']
    },
    {
        'question': 'Find vaccine suppliers with cold chain capability.',
        'expected_ids': list(df[
            (df['cold_chain_capable'] == True) &
            (df['product_category'].str.contains('Vaccine', na=False))
        ]['supplier_id']),
        'context_keywords': ['vaccine', 'cold chain', 'temperature']
    },
    {
        'question': 'Find low-cost generic drug manufacturers.',
        'expected_ids': list(df[
            df['product_category'].str.contains('Generic', na=False)
        ].nsmallest(5, 'unit_price_inr')['supplier_id']),
        'context_keywords': ['generic', 'price', 'cost']
    },
    {
        'question': 'Find CDSCO approved suppliers for antibiotics in India.',
        'expected_ids': list(df[
            (df['cdsco_approved'] == True) &
            (df['country'] == 'India') &
            (df['product_category'].str.contains('Antibiotic', na=False))
        ]['supplier_id']),
        'context_keywords': ['cdsco', 'india', 'antibiotic']
    },
    {
        'question': 'Find cold chain suppliers for pain relief medications.',
        'expected_ids': list(df[
            (df['cold_chain_capable'] == True) &
            (df['product_category'].str.contains('Pain', na=False))
        ]['supplier_id']),
        'context_keywords': ['pain', 'cold chain', 'relief']
    },
]


def context_relevance_score(docs: list, keywords: list) -> float:
    """
    For each retrieved doc, check what fraction of required keywords appear in it.
    Average across all docs.
    This is a keyword-overlap proxy for RAGAS context relevance.
    """
    if not docs:
        return 0.0

    doc_scores = []
    for doc in docs:
        text = doc.page_content.lower()
        hits = sum(1 for kw in keywords if kw.lower() in text)
        doc_scores.append(hits / len(keywords))

    return sum(doc_scores) / len(doc_scores)


print()
print("=" * 70)
print("  RETRIEVAL EVALUATION — MediCare Pharmaceuticals India")
print("=" * 70)

all_precision   = []
all_recall      = []
all_ctx_rel     = []

for i, tc in enumerate(test_cases, 1):
    agent.ask(tc['question'])

    retrieved_ids = [d.metadata.get('supplier_id') for d in agent.last_retrieved_docs]
    expected_ids  = tc['expected_ids']
    hits          = [rid for rid in retrieved_ids if rid in expected_ids]

    precision = len(hits) / len(retrieved_ids) if retrieved_ids else 0
    recall    = len(hits) / len(expected_ids)  if expected_ids  else 0
    ctx_rel   = context_relevance_score(agent.last_retrieved_docs, tc['context_keywords'])

    all_precision.append(precision)
    all_recall.append(recall)
    all_ctx_rel.append(ctx_rel)

    print(f"\nQ{i}: {tc['question'][:62]}...")
    print(f"   Expected IDs      : {expected_ids}")
    print(f"   Retrieved IDs     : {retrieved_ids}")
    print(f"   Hits              : {hits}")
    print(f"   Precision         : {precision:.0%}  ({len(hits)}/{len(retrieved_ids)} retrieved were relevant)")
    print(f"   Recall            : {recall:.0%}  ({len(hits)}/{len(expected_ids)} relevant were retrieved)")
    print(f"   Context Relevance : {ctx_rel:.0%}  (keyword overlap: {tc['context_keywords']})")

avg_precision = sum(all_precision) / len(all_precision)
avg_recall    = sum(all_recall)    / len(all_recall)
avg_ctx_rel   = sum(all_ctx_rel)   / len(all_ctx_rel)

print()
print("=" * 70)
print("  SUMMARY")
print("=" * 70)
print(f"  Average Precision         : {avg_precision:.0%}  (target > 60%)")
print(f"  Average Recall            : {avg_recall:.0%}  (target > 50%)")
print(f"  Average Context Relevance : {avg_ctx_rel:.0%}  (target > 70%)")
print("=" * 70)
print()

# Interpretation
print("INTERPRETATION:")
if avg_precision >= 0.6:
    print("  ✅ Precision OK — retrieval is finding relevant suppliers")
else:
    print("  ❌ Precision LOW — retrieval is returning too many irrelevant suppliers")

if avg_recall >= 0.5:
    print("  ✅ Recall OK — retrieval is covering most relevant suppliers")
else:
    print("  ⚠️  Recall LOW — some relevant suppliers are being missed (expected with k=5)")

if avg_ctx_rel >= 0.7:
    print("  ✅ Context Relevance OK — retrieved docs contain the right information")
else:
    print("  ⚠️  Context Relevance LOW — retrieved docs may not fully answer the question")

print()
print("Note: These scores go in your report under 'Evaluation Methodology'.")
print("      Label as: Manual evaluation (RAGAS unavailable on CPU-only setup)")
