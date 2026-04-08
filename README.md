# Supply Chain Resilience and Demand Sensing Agent for Pharmaceutical Companies

**MediCare Pharmaceuticals India — AI-Powered Supply Chain Management System**

> A GenAI course project by Yuvika T, Yogitha A S, and Godha Kanchi Mandya  
> Department of CSE, PES University, Bengaluru, India

---

## What This Project Does

MediCare Pharmaceuticals India is a mid-sized pharmaceutical distributor in Bangalore serving 2,500 hospitals and 8,000 pharmacies across India. Managing the supply chain for critical drugs like insulin, vaccines, and antibiotics is extremely complex — prices change, suppliers go out of stock, geopolitical events (like China banning API exports) disrupt supply, and CDSCO regulations must be followed at all times.

This project builds an **AI agent** that helps procurement officers at MediCare:
- Find the right suppliers instantly from a database of 100 pharmaceutical companies
- Get warned about supply disruptions via real pharmaceutical news (GDELT)
- Check if a supplier is CDSCO/GMP compliant before placing an order
- Monitor warehouse inventory levels and know when to reorder
- Compare supplier prices and generate draft purchase orders

Instead of a procurement officer spending hours searching spreadsheets and calling suppliers, they type a question in plain English and the AI gives them a ranked list of suppliers with contact details, prices, compliance status, and a recommended action — in under 2 seconds.

---

## System Architecture

```
User Question (Streamlit UI)
         │
         ▼
   Orchestrator (orchestrator.py)
   ── Keyword routing decides which tools to call ──
         │
         ├──► RAG Agent (simple_agent_improved.py)
         │         │
         │         ├── FAISS Vector Search (top-10 semantic results)
         │         ├── BM25 Sparse Search  (top-5 keyword results)
         │         ├── Merge + Deduplicate (up to 15 candidates)
         │         ├── Blocked Country Pre-filter
         │         ├── Category + Compliance Filter
         │         ├── Cross-Encoder Reranking (ms-marco-MiniLM-L-6-v2)
         │         └── LLaMA-3-8B generates 1-sentence recommendation
         │
         ├──► inventory_check()   — reads current_inventory.json
         ├──► news_monitor()      — searches GDELT CSV
         ├──► compliance_check()  — reads pharma_suppliers.csv
         ├──► price_compare()     — weighted ranking formula
         └──► draft_order()       — generates PO for human approval
```

**Key design principle:** The LLM is only used to write one recommendation sentence. All supplier facts (price, location, reliability, compliance) come directly from the database — this eliminates hallucination.

---

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Language Model | LLaMA-3-8B (GGUF 4-bit) | Runs on CPU, no GPU needed, private/on-premise |
| Fine-tuning | 300 pharma JSONL scenarios | Domain-specific responses |
| Vector Database | FAISS (Facebook AI Similarity Search) | Sub-millisecond semantic retrieval |
| Sparse Retrieval | BM25 (rank_bm25) | Catches exact keyword matches FAISS misses |
| Reranking | Cross-Encoder (ms-marco-MiniLM-L-6-v2) | Re-scores top candidates for precision |
| Embeddings | sentence-transformers/all-MiniLM-L6-v2 | 384-dim vectors, fast and accurate |
| LLM Runtime | llama-cpp-python | Loads GGUF model on CPU |
| Frontend | Streamlit | Interactive dashboard |
| Visualization | Plotly | Charts for inventory and supplier analytics |
| News Data | GDELT Global Knowledge Graph | Real pharmaceutical news |

---

## Project Structure

```
genai_project/
│
├── app/                              # Core application code
│   ├── streamlit_ui.py               # Main Streamlit dashboard (run this)
│   ├── simple_agent_improved.py      # RAG agent — retrieval, filtering, reranking
│   ├── orchestrator.py               # Routes queries to the right tools
│   └── tools.py                      # 6 standalone tool functions
│
├── scripts/                          # Data pipeline scripts (run once in order)
│   ├── 1_create_company_profile.py   # Creates company_profile.json
│   ├── 1b_create_inventory.py        # Creates current_inventory.json
│   ├── 2_generate_pharma_suppliers.py# Generates 100 supplier profiles CSV
│   ├── 3_create_pharma_training.py   # Creates initial 50 training examples
│   ├── 3.1_pharma_training_250_more.py # Creates additional 250 examples (total = 300)
│   ├── 4_download_pharma_news.py     # Downloads news from GDELT API
│   ├── 5_preprocess_pharma_news.py   # Cleans and categorizes news
│   ├── 7_load_suppliers_to_rag.py    # Embeds suppliers into FAISS
│   ├── 8_test_faiss_search.py        # Tests FAISS retrieval
│   └── manual_eval.py                # Manual precision/recall evaluation
│
├── data/
│   ├── company/
│   │   ├── company_profile.json      # MediCare company details
│   │   └── current_inventory.json    # Live warehouse stock levels (8 products)
│   ├── suppliers/
│   │   └── pharma_suppliers.csv      # 100 supplier profiles with all attributes
│   ├── gdelt/
│   │   ├── gdelt_pharma_raw.csv      # Raw downloaded news
│   │   └── gdelt_pharma_clean.csv    # Cleaned, categorized, scored news
│   └── training/
│       └── pharma_training_300.jsonl # 300 fine-tuning examples (50 base + 250 additional, ChatML format)
│
├── database/
│   └── faiss_suppliers/              # FAISS index files (pre-built)
│       ├── index.faiss
│       └── index.pkl
│
├── models/
│   └── llama-3-8b_300.Q4_K_M.gguf   # Fine-tuned LLaMA model (4-bit quantized)
│
└── requirements.txt
```

---

## How to Run

### Prerequisites
- Python 3.9+
- ~4 GB RAM free (for the GGUF model)
- The model file: `models/llama-3-8b_300.Q4_K_M.gguf`
- The FAISS index: `database/faiss_suppliers/`

### Install dependencies
```bash
pip install -r requirements.txt
```

### Run the Streamlit app
```bash
cd genai_project
streamlit run app/streamlit_ui.py
```

The app opens at `http://localhost:8501`

### Run evaluation
```bash
python3 scripts/manual_eval.py
```

---

## Data Pipeline (Run Once to Rebuild From Scratch)

If you need to regenerate all data, run the scripts in this order:

```bash
python3 scripts/1_create_company_profile.py      # Step 1: Company JSON
python3 scripts/1b_create_inventory.py           # Step 1b: Inventory JSON
python3 scripts/2_generate_pharma_suppliers.py   # Step 2: 100 supplier profiles
python3 scripts/3_create_pharma_training.py         # Step 3a: Initial 50 training examples
python3 scripts/3.1_pharma_training_250_more.py     # Step 3b: Additional 250 examples (total = 300)
python3 scripts/4_download_pharma_news.py           # Step 4: Download GDELT news (needs internet)
python3 scripts/5_preprocess_pharma_news.py         # Step 5: Clean and categorize news
python3 scripts/7_load_suppliers_to_rag.py          # Step 6: Embed suppliers and build FAISS index
```

> **Note:** Step 4 requires internet access. All other steps are fully offline.

---

## The Inventory System

The warehouse tracks 8 drug products across different urgency levels:

| Product | Category | Urgency | Sourcing Strategy |
|---------|----------|---------|-------------------|
| Insulin Glargine | Diabetes | 🔴 HIGH | India domestic (CDSCO approved) |
| Atorvastatin | Cardiac | 🔴 HIGH | International (no India CDSCO supplier) |
| Salbutamol Inhaler | Respiratory | 🔴 HIGH | India domestic |
| Hepatitis B Vaccine | Vaccines | 🔴 HIGH | India domestic (cold chain) |
| Amoxicillin | Antibiotics | 🟡 MEDIUM | India domestic (China API risk) |
| Imatinib | Oncology | 🟢 LOW | India preferred / international backup |
| Paracetamol | Pain Relief | 🟢 LOW | India domestic |
| Methotrexate | Oncology | 🟢 LOW | India preferred |

Urgency is calculated from CDSCO safety stock policy:
- Critical drugs need 90 days target stock
- Below 65% of target → HIGH (reorder now)
- Below 85% of target → MEDIUM (order soon)
- Above 85% of target → LOW (adequate)

The inventory JSON (`current_inventory.json`) is read live by the UI every 60 seconds. When stock levels change, the dashboard buttons and alerts update automatically.

---

## Query Scenarios Covered

The AI agent handles these types of queries:

| Scenario | Example Query | What Happens |
|----------|--------------|--------------|
| Domestic sourcing | "Find insulin suppliers in India urgently" | Filters India + CDSCO, sorts by lead time |
| International sourcing | "Find cardiac drug suppliers globally" | Removes India filter, searches worldwide |
| Geopolitical disruption | "China blocks API exports, find alternatives" | Pre-filters out Chinese suppliers, biases toward India/others |
| Cold chain | "Find vaccine suppliers with cold chain" | Filters cold_chain_capable = True |
| Compliance check | "Check compliance for PHARM-0072" | Reads CSV directly, returns CDSCO/GMP/score |
| Price comparison | "Compare prices across suppliers" | Weighted ranking: 50% price, 30% lead time, 20% reliability |
| Emergency restock | "Port strike, need insulin in 7 days" | Triggers fast-delivery filter |

---

## Evaluation Results

Evaluated on 5 manual test cases using the `manual_eval.py` script:

| Metric | Score | Target | Status |
|--------|-------|--------|--------|
| Precision | **87%** | > 60% | ✅ Pass |
| Recall | **49%** | > 40% | ✅ Pass |
| Context Relevance | **90%** | > 70% | ✅ Pass |
| Avg Response Time | **1.8s** | < 3s | ✅ Pass |

**Why recall is 49% and not higher:** The agent retrieves top-5 suppliers. If 12 suppliers qualify in the database, recall is capped at 5/12 = 42%. This is a deliberate design choice — returning 5 focused results is more useful for a procurement officer than returning 15 options. Precision (how many of the 5 are actually correct) is 87%.

Compared to a keyword-only baseline:
- Precision improved by **+58%**
- Context relevance improved by **+29%**

---

## Key Design Decisions

### Why keyword routing instead of LLM planning?
We tried using the LLM to decide which tools to call (by outputting JSON). It failed constantly — 4-bit quantized models reliably ignore JSON format instructions and output plain English instead. Keyword routing (Python if/else) is 100% reliable, instant, and uses zero tokens.

### Why hybrid FAISS + BM25?
FAISS (semantic search) finds suppliers that are *conceptually* related even if the exact word isn't there (e.g., "diabetes" matches "insulin supplier"). BM25 catches exact keyword matches that FAISS might rank lower. Together they cover more ground than either alone.

### Why pre-filter before cross-encoder reranking?
The cross-encoder scores query-document pairs. If the query says "China blocks exports", the model may actually score Chinese suppliers *higher* (they appear in the same context). Pre-filtering removes them before scoring, so they cannot be ranked back in.

### Why does the LLM only write one sentence?
Every supplier fact (price, location, lead time, reliability, compliance) is read directly from the database and formatted in Python. The LLM only writes the recommendation paragraph. This means even if the model hallucinates, it cannot affect the factual supplier cards — only the recommendation text.

---

## The Streamlit Dashboard

The app has 4 tabs:

1. **AI Agent Chat** — Main interface. Has dynamic buttons for all 8 inventory products (red for critical, yellow for medium, green for adequate). Also has supply risk scenario buttons (China ban, port strike). Type any question or click a button to get supplier recommendations.

2. **News Monitor** — Shows GDELT pharmaceutical news filtered by category, urgency level (1-5), and keyword search. Click "Analyze Impact" on any article to get AI-powered supplier suggestions based on that news event.

3. **Analytics Dashboard** — Visual charts: inventory days-of-supply bar chart with CDSCO target lines, supplier geographic distribution pie chart, supplier product category pie chart, top 10 suppliers by reliability, news category and urgency trend bars.

4. **About** — Project overview, technology stack, academic context, and live system status.

---

## Company Context

**MediCare Pharmaceuticals India Pvt. Ltd.**
- Headquarters: Bangalore, Karnataka
- Annual Revenue: ₹500 crore (~$60M USD)
- Customers: 2,500 hospitals, 8,000 pharmacies, 5,000 clinics
- Warehouses: 5 warehouses, 3 cold storage facilities
- Inventory Value: ₹350 crore
- Regulatory: CDSCO licensed, WHO-GDP certified, ISO 9001:2015

The company sources 40% of Active Pharmaceutical Ingredients (APIs) from China, 45% from domestic Indian manufacturers, and 15% from Europe/USA. This makes it vulnerable to geopolitical disruptions — which is exactly the problem this AI system is designed to help manage.

---

## Authors

| Name | Email |
|------|-------|
| Yuvika T | leisha.yuvi@gmail.com |
| Yogitha A S | yogithaas4@gmail.com |
| Godha Kanchi Mandya | godha.mandya@gmail.com |

Department of Computer Science and Engineering  
PES University, Bengaluru, India

---

*For Academic Use Only — GenAI Course Project*
