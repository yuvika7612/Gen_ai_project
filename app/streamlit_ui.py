"""
Pharmaceutical Supply Chain AI Agent - Streamlit Frontend
Uses your existing PharmaSupplyChainAgent with GGUF model
"""

import streamlit as st
import pandas as pd
import json
import os
from datetime import datetime
import plotly.express as px
import plotly.graph_objects as go

# Import your agent class
import sys, os
sys.path.insert(0, os.path.dirname(__file__))
from simple_agent_improved import PharmaSupplyChainAgent

# Page configuration
st.set_page_config(
    page_title="MediCare Pharma Supply Chain AI",
    page_icon="💊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        padding: 20px 0;
        border-bottom: 3px solid #1f77b4;
        margin-bottom: 30px;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 20px;
        border-radius: 10px;
        border-left: 5px solid #1f77b4;
    }
    .critical-alert {
        background-color: #ffe6e6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #ff4444;
        margin: 10px 0;
    }
    .success-box {
        background-color: #e6ffe6;
        padding: 15px;
        border-radius: 8px;
        border-left: 5px solid #44ff44;
        margin: 10px 0;
    }
    .stButton>button {
        background-color: #1f77b4;
        color: white;
        font-weight: bold;
        border-radius: 8px;
        padding: 10px 25px;
        border: none;
    }
    .stButton>button:hover {
        background-color: #155a8a;
    }
    .chat-message {
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .user-message {
        background-color: #e3f2fd;
        border-left: 4px solid #2196F3;
    }
    .agent-message {
        background-color: #f5f5f5;
        border-left: 4px solid #4CAF50;
    }
</style>
""", unsafe_allow_html=True)

# ============================================================
# INITIALIZATION & CACHING
# ============================================================

@st.cache_resource
def load_agent():
    """Load the PharmaSupplyChainAgent (cached)"""
    try:
        agent = PharmaSupplyChainAgent()
        return agent, None
    except Exception as e:
        return None, str(e)

@st.cache_data(ttl=60)  # re-read inventory every 60 seconds
def load_company_data():
    """Load company profile and inventory"""
    with open('data/company/company_profile.json') as f:
        company = json.load(f)

    with open('data/company/current_inventory.json') as f:
        inventory = json.load(f)

    return company, inventory

@st.cache_data
def load_news_data():
    """Load GDELT pharmaceutical news"""
    if os.path.exists('data/gdelt/gdelt_test_set.csv'):
        return pd.read_csv('data/gdelt/gdelt_test_set.csv')
    return None

@st.cache_data
def load_suppliers_data():
    """Load supplier database"""
    if os.path.exists('data/suppliers/pharma_suppliers.csv'):
        return pd.read_csv('data/suppliers/pharma_suppliers.csv')
    return None

# Initialize session state
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_question' not in st.session_state:
    st.session_state.current_question = ''

if 'active_tab' not in st.session_state:
    st.session_state.active_tab = 0

# Load data
company, inventory = load_company_data()
news_df = load_news_data()
suppliers_df = load_suppliers_data()

# Load agent
agent, error = load_agent()

# ============================================================
# SIDEBAR - COMPANY INFO & INVENTORY STATUS
# ============================================================

st.sidebar.markdown("# 🏥 MediCare Pharmaceuticals India")
st.sidebar.markdown(f"📍 {company['headquarters']}")
st.sidebar.markdown(f"💰 Annual Revenue: {company['annual_revenue']}")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📦 Warehouse Stock Alert")
st.sidebar.markdown("*How many days of stock remain at current consumption rate*")

# Count alerts for summary
critical_count = sum(1 for d in inventory['products'].values() if d['urgency'] == 'HIGH')
if critical_count:
    st.sidebar.error(f"⚠️ {critical_count} product(s) need urgent reorder")

for product_id, details in inventory['products'].items():
    name    = details['product_name']
    days    = details['days_of_supply']
    stock   = details.get('current_stock', 0)
    demand  = details.get('monthly_demand', 0)
    unit    = details.get('unit', 'units')
    urgency = details['urgency']
    category = details.get('category', product_id)

    if urgency == 'HIGH':
        color = "🔴"
        label = "REORDER NOW"
        bg_color     = "#ffe6e6"
        border_color = "#dc3545"
    elif urgency == 'MEDIUM':
        color = "🟡"
        label = "ORDER SOON"
        bg_color     = "#fff3cd"
        border_color = "#ffc107"
    else:
        color = "🟢"
        label = "ADEQUATE"
        bg_color     = "#e6ffed"
        border_color = "#28a745"

    st.sidebar.markdown(f"""
<div style="background:{bg_color}; padding:10px; border-radius:8px;
     margin-bottom:4px;
     border-left: 4px solid {border_color}">
<b>{color} {name}</b><br>
<span style="font-size:1.3em; font-weight:bold">{days} days left</span>
&nbsp;<span style="color:grey; font-size:0.85em">{stock:,} {unit} in stock</span><br>
<span style="font-size:0.8em">Monthly demand: {demand:,} {unit} &nbsp;|&nbsp; <b>{label}</b></span>
</div>
""", unsafe_allow_html=True)

    st.sidebar.markdown("<div style='margin-bottom:8px'></div>",
                        unsafe_allow_html=True)

st.sidebar.markdown("---")
st.sidebar.markdown("## 🔧 System Status")

if agent:
    st.sidebar.success("✅ AI Agent: Online")
else:
    st.sidebar.error("❌ AI Agent: Offline")
    if error:
        st.sidebar.error(f"Error: {error[:100]}...")

col_a, col_b = st.sidebar.columns(2)
with col_a:
    st.metric("Suppliers", len(suppliers_df) if suppliers_df is not None else 0)
with col_b:
    st.metric("News Articles", len(news_df) if news_df is not None else 0)

st.sidebar.markdown("---")
st.sidebar.caption(f"Last updated: {inventory['last_updated']}")

# ============================================================
# MAIN CONTENT
# ============================================================

st.markdown('<div class="main-header">💊 Pharmaceutical Supply Chain AI Assistant</div>', unsafe_allow_html=True)

# Create tabs — active_tab=0 forces AI Agent tab open when sidebar button clicked
tab_names = ["🤖 AI Agent Chat", "📰 News Monitor", "📊 Analytics Dashboard", "ℹ️ About"]
tab1, tab2, tab3, tab4 = st.tabs(tab_names)

# Auto-switch to AI Agent tab if triggered from sidebar
if st.session_state.active_tab == 0 and st.session_state.current_question:
    # Reset so it doesn't keep switching on every rerun
    st.session_state.active_tab = 0

# ============================================================
# TAB 1: AI AGENT CHAT
# ============================================================

with tab1:
    st.header("🤖 AI Supply Chain Agent")

    if not agent:
        st.error("❌ AI Agent failed to load. Please check the error in the sidebar.")
        st.info("""
        **Common issues:**
        - GGUF model file not found at `models/llama-3-8b_300.Q4_K_M.gguf`
        - FAISS database not loaded
        - Missing dependencies (`pip install llama-cpp-python`)
        """)
    else:
        col1, col2 = st.columns([2, 1])

        with col1:
            st.markdown("""
            Ask the AI about supply chain disruptions, inventory decisions, or emergency scenarios.

            **The agent will:**
            - Search 100+ pharmaceutical suppliers in the database
            - Provide specific recommendations
            - Consider regulatory compliance (CDSCO)
            - Calculate costs and timelines
            """)

        with col2:
            # Show active critical alerts from inventory
            critical_alerts = inventory.get('critical_alerts', [])
            if critical_alerts:
                alert_text = "\n".join(
                    f"- 🔴 {a['message']}" for a in critical_alerts[:3]
                )
                st.warning(f"**Active Stock Alerts:**\n{alert_text}")

        # ── Dynamic inventory action buttons ─────────────────────────────
        # Query is built from the 'sourcing' field in current_inventory.json:
        #
        #   india_available     → "in India" added → domestic first
        #   international_required → no "India" → global search (no India+CDSCO supplier exists)
        #   geopolitical_risk   → mentions China API ban scenario
        #   india_preferred     → tries India first, mentions international fallback
        #
        # emoji per category for button display
        CATEGORY_EMOJI = {
            "Diabetes": "🩸", "Cardiac": "💊", "Respiratory": "🫁",
            "Vaccines": "💉", "Antibiotics": "💊", "Oncology": "🎗️",
            "Pain Relief": "💊",
        }

        def build_query(name, days, category, sourcing):
            """Build agent query based on sourcing strategy from inventory JSON."""
            s = sourcing or "india_available"
            if s == "india_available":
                return (
                    f"Find CDSCO approved {category.lower()} suppliers in India urgently. "
                    f"We have only {days} days of {name} stock remaining."
                )
            elif s == "international_required":
                return (
                    f"Find CDSCO approved {category.lower()} suppliers. "
                    f"Global sourcing required — no domestic CDSCO-approved supplier "
                    f"available for this category. "
                    f"We have only {days} days of {name} stock remaining."
                )
            elif s == "geopolitical_risk":
                return (
                    f"Find CDSCO approved {category.lower()} suppliers in India urgently. "
                    f"China blocks API exports — need non-China alternative. "
                    f"We have only {days} days of {name} stock remaining."
                )
            elif s == "india_preferred":
                return (
                    f"Find CDSCO approved {category.lower()} suppliers. "
                    f"Prefer India but consider international if unavailable. "
                    f"We have only {days} days of {name} stock remaining."
                )
            return (
                f"Find {category.lower()} suppliers urgently. "
                f"We have only {days} days of {name} stock remaining."
            )

        # Build button list — HIGH and MEDIUM urgency only
        action_products = [
            (pid, details)
            for pid, details in inventory['products'].items()
            if details['urgency'] in ('HIGH', 'MEDIUM')
        ]
        action_products.sort(key=lambda x: (
            0 if x[1]['urgency'] == 'HIGH' else 1,
            x[1]['days_of_supply']
        ))

        if action_products:
            st.markdown("### 🚨 Current Inventory Actions")
            st.caption(
                "Dynamically generated from your warehouse stock levels. "
                "India suppliers preferred where available; "
                "international sourcing shown where no domestic CDSCO-approved supplier exists."
            )

            for row_start in range(0, len(action_products), 3):
                row   = action_products[row_start:row_start + 3]
                cols  = st.columns(3)
                for col, (pid, details) in zip(cols, row):
                    name          = details['product_name']
                    days          = details['days_of_supply']
                    category      = details.get('category', '')
                    urgency_level = details['urgency']
                    sourcing      = details.get('sourcing', 'india_available')

                    emoji = CATEGORY_EMOJI.get(category, "💊")
                    flag  = "🔴" if urgency_level == 'HIGH' else "🟡"
                    label = "CRITICAL" if urgency_level == 'HIGH' else "LOW"

                    # Sourcing hint shown on button
                    src_hint = {
                        "india_available"      : "🇮🇳 Domestic",
                        "international_required": "🌍 International",
                        "geopolitical_risk"    : "⚠️ Alt Source",
                        "india_preferred"      : "🇮🇳 India/Global",
                    }.get(sourcing, "")

                    btn_text = f"{flag} {name[:20]}\n{days}d — {label} | {src_hint}"
                    query    = build_query(name, days, category, sourcing)

                    with col:
                        if st.button(btn_text, key=f"action_{pid}",
                                     use_container_width=True):
                            st.session_state.current_question = query
                            st.rerun()
        else:
            st.success("✅ All inventory levels are adequate — no urgent actions needed.")

        # ── Adequate stock products — routine supply queries ──────────────
        adequate_products = [
            (pid, details)
            for pid, details in inventory['products'].items()
            if details['urgency'] == 'LOW'
        ]
        adequate_products.sort(key=lambda x: x[1]['days_of_supply'])

        if adequate_products:
            st.markdown("### 📊 Routine Supply Queries")
            st.caption(
                "Products currently at adequate stock levels — "
                "use these to identify and lock in suppliers before a shortage occurs."
            )
            for row_start in range(0, len(adequate_products), 3):
                row   = adequate_products[row_start:row_start + 3]
                cols  = st.columns(3)
                for col, (pid, details) in zip(cols, row):
                    name     = details['product_name']
                    days     = details['days_of_supply']
                    category = details.get('category', '')
                    sourcing = details.get('sourcing', 'india_available')
                    emoji    = CATEGORY_EMOJI.get(category, "💊")
                    query    = build_query(name, days, category, sourcing)
                    with col:
                        if st.button(f"🟢 {name[:22]}\n{days}d — Adequate",
                                     key=f"routine_{pid}",
                                     use_container_width=True):
                            st.session_state.current_question = query
                            st.rerun()

        # Geopolitical disruption — standing supply risk scenarios
        st.markdown("#### 🌐 Supply Risk Scenarios")
        st.caption("Simulate geopolitical or supply disruption events")

        geo_col1, geo_col2 = st.columns(2)

        with geo_col1:
            if st.button("🇨🇳 China API Ban — Find Alternative Antibiotic Suppliers",
                         use_container_width=True, key="geo_china_ban"):
                amox_days = inventory['products'].get(
                    'amoxicillin', {}
                ).get('days_of_supply', '')
                days_note = f" Our Amoxicillin stock is also low ({amox_days} days)." \
                            if amox_days else ""
                st.session_state.current_question = (
                    f"China blocks API exports for antibiotics. "
                    f"Find alternative suppliers urgently.{days_note}"
                )
                st.rerun()

        with geo_col2:
            if st.button("⚡ Port Strike — Emergency Insulin Restock",
                         use_container_width=True, key="geo_port_strike"):
                insulin_days = inventory['products'].get(
                    'insulin_glargine', {}
                ).get('days_of_supply', '')
                days_note = f" Only {insulin_days} days of insulin stock remaining." \
                            if insulin_days else ""
                st.session_state.current_question = (
                    f"Major port strike disrupting imports. "
                    f"Find domestic CDSCO approved diabetes suppliers in India "
                    f"who can deliver within 7 days. It is an emergency.{days_note}"
                )
                st.rerun()

        st.markdown("---")

        # Chat interface
        st.markdown("### 💬 Ask Your Question")

        # User input
        user_question = st.text_area(
            "Type your supply chain question:",
            value=st.session_state.current_question,
            height=100,
            placeholder="Example: Find insulin suppliers in India with cold chain and CDSCO approval",
            key="question_input"
        )

        col_submit, col_clear, col_history = st.columns([1, 1, 3])

        with col_submit:
            submit_button = st.button("🚀 Ask Agent", type="primary", use_container_width=True)

        with col_clear:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.current_question = ''
                st.rerun()

        with col_history:
            if st.button("📜 Clear Chat History", use_container_width=True):
                st.session_state.chat_history = []
                st.rerun()

        # Process query
        if submit_button and user_question:
            with st.spinner("🔍 Searching supplier database and generating response..."):
                try:
                    # Call your agent's ask method
                    response = agent.ask(user_question)

                    # Add to chat history
                    st.session_state.chat_history.append({
                        'question': user_question,
                        'answer': response,
                        'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    })

                    # Clear current question
                    st.session_state.current_question = ''

                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
                    st.info("Make sure the GGUF model is loaded correctly and the path is correct.")

        # Display chat history (most recent first)
        if st.session_state.chat_history:
            st.markdown("---")
            st.markdown("## 💬 Conversation History")

            for i, chat in enumerate(reversed(st.session_state.chat_history)):
                with st.container():
                    # User question
                    st.markdown(f'<div class="chat-message user-message">', unsafe_allow_html=True)
                    st.markdown(f"**👤 You asked:** ({chat['timestamp']})")
                    st.markdown(chat['question'])
                    st.markdown('</div>', unsafe_allow_html=True)

                    # Agent response
                    st.markdown(f'<div class="chat-message agent-message">', unsafe_allow_html=True)
                    st.markdown(f"**🤖 Agent Response:**")
                    st.markdown(chat['answer'])
                    st.markdown('</div>', unsafe_allow_html=True)

                    st.markdown("---")

# ============================================================
# TAB 2: NEWS MONITOR
# ============================================================

with tab2:
    st.header("📰 Pharmaceutical Supply Chain News Monitor")

    if news_df is None:
        st.warning("⚠️ News data not available. Run GDELT download scripts first.")
        st.code("""
# Download and process GDELT news:
python scripts/4_download_pharma_news.py
python scripts/5_preprocess_pharma_news.py
python scripts/6_create_test_set.py
        """)
    else:
        # Stats
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Articles", len(news_df))

        with col2:
            st.metric("Categories", news_df['category'].nunique())

        with col3:
            critical_news = len(news_df[news_df['urgency_score'] >= 4])
            st.metric("Critical Alerts", critical_news)

        with col4:
            st.metric("Sources", news_df['source_domain'].nunique())

        st.markdown("---")

        # Filters
        col1, col2, col3 = st.columns(3)

        with col1:
            categories = st.multiselect(
                "Filter by Category:",
                options=sorted(news_df['category'].unique()),
                default=sorted(news_df['category'].unique())[:3]
            )

        with col2:
            urgency = st.slider(
                "Minimum Urgency Level:",
                min_value=1,
                max_value=5,
                value=3
            )

        with col3:
            search_term = st.text_input("Search Headlines:", "")

        # Filter data
        filtered_df = news_df[
            (news_df['category'].isin(categories)) &
            (news_df['urgency_score'] >= urgency)
        ]

        if search_term:
            filtered_df = filtered_df[
                filtered_df['headline'].str.contains(search_term, case=False, na=False)
            ]

        st.markdown(f"### 📋 Showing {len(filtered_df)} articles")

        # Display news
        for idx, row in filtered_df.head(20).iterrows():
            # Urgency indicator
            if row['urgency_score'] >= 4:
                urgency_color = "🔴 CRITICAL"
            elif row['urgency_score'] >= 3:
                urgency_color = "🟡 MEDIUM"
            else:
                urgency_color = "🟢 LOW"

            with st.expander(f"{urgency_color} | [{row['category']}] {row['headline'][:100]}..."):
                col1, col2 = st.columns([3, 1])

                with col1:
                    st.markdown(f"**Full Headline:** {row['headline']}")
                    st.markdown(f"**Date:** {row['date']}")
                    st.markdown(f"**Source:** {row['source_domain']}")
                    if 'countries_mentioned' in row:
                        st.markdown(f"**Countries:** {row['countries_mentioned']}")

                with col2:
                    st.markdown(f"**Category:** {row['category']}")
                    st.markdown(f"**Urgency:** {row['urgency_score']}/5")

                    if agent and st.button(f"🔍 Analyze Impact", key=f"analyze_{idx}"):
                        with st.spinner("Analyzing..."):
                            try:
                                analysis = agent.ask(f"Analyze this pharmaceutical news and suggest suppliers: {row['headline']}")
                                st.markdown("**AI Analysis:**")
                                st.info(analysis)
                            except Exception as e:
                                st.error(f"Analysis failed: {str(e)}")

                st.markdown(f"[📰 Read Article]({row['url']})")

# ============================================================
# TAB 3: ANALYTICS DASHBOARD
# ============================================================

with tab3:
    st.header("📊 Supply Chain Analytics Dashboard")

    # Inventory visualization
    st.subheader("📦 Inventory Status Overview")

    # Prepare data
    inventory_data = []
    for product_id, details in inventory['products'].items():
        inventory_data.append({
            'Product': details['product_name'][:20] + '...',
            'Days of Supply': details['days_of_supply'],
            'Monthly Demand': details['monthly_demand'],
            'Current Stock': details['current_stock'],
            'Status': details['status'],
            'Urgency': details['urgency']
        })

    df_inv = pd.DataFrame(inventory_data)

    # Bar chart
    fig1 = px.bar(
        df_inv,
        x='Product',
        y='Days of Supply',
        color='Status',
        color_discrete_map={'CRITICAL': '#ff4444', 'BELOW_TARGET': '#ffaa44', 'ADEQUATE': '#44ff44'},
        title='Inventory Days of Supply vs Targets',
        labels={'Days of Supply': 'Days of Supply'}
    )

    # Add target lines
    fig1.add_hline(y=90, line_dash="dash", line_color="red",
                   annotation_text="Critical Drug Target (90 days)",
                   annotation_position="right")
    fig1.add_hline(y=60, line_dash="dash", line_color="orange",
                   annotation_text="Essential Drug Target (60 days)",
                   annotation_position="right")

    st.plotly_chart(fig1, use_container_width=True)

    # Supplier analytics
    if suppliers_df is not None:
        st.markdown("---")
        st.subheader("🏭 Supplier Analytics")

        col1, col2 = st.columns(2)

        with col1:
            # Geographic distribution
            fig2 = px.pie(
                suppliers_df,
                names='country',
                title='Supplier Geographic Distribution',
                hole=0.4
            )
            st.plotly_chart(fig2, use_container_width=True)

        with col2:
            # Product category distribution
            fig3 = px.pie(
                suppliers_df,
                names='product_category',
                title='Supplier Product Categories',
                hole=0.4
            )
            st.plotly_chart(fig3, use_container_width=True)

        # Supplier metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.metric("Total Suppliers", len(suppliers_df))

        with col2:
            st.metric("Cold Chain Capable",
                     suppliers_df['cold_chain_capable'].sum())

        with col3:
            st.metric("CDSCO Approved",
                     suppliers_df['cdsco_approved'].sum())

        with col4:
            avg_reliability = suppliers_df['reliability_score'].mean()
            st.metric("Avg Reliability", f"{avg_reliability:.1f}%")

        # Top suppliers table
        st.markdown("---")
        st.subheader("🏆 Top Suppliers by Reliability")

        top_suppliers = suppliers_df.nlargest(10, 'reliability_score')[
            ['company_name', 'country', 'product_category', 'reliability_score',
             'cdsco_approved', 'cold_chain_capable']
        ]

        st.dataframe(
            top_suppliers,
            use_container_width=True,
            hide_index=True
        )

    # News analytics
    if news_df is not None:
        st.markdown("---")
        st.subheader("📰 News Trends")

        col1, col2 = st.columns(2)

        with col1:
            # Category distribution
            category_counts = news_df['category'].value_counts()
            fig5 = px.bar(
                x=category_counts.index,
                y=category_counts.values,
                title='News by Category',
                labels={'x': 'Category', 'y': 'Count'}
            )
            st.plotly_chart(fig5, use_container_width=True)

        with col2:
            # Urgency distribution
            urgency_counts = news_df['urgency_score'].value_counts().sort_index()
            fig6 = px.bar(
                x=urgency_counts.index,
                y=urgency_counts.values,
                title='News by Urgency Level',
                labels={'x': 'Urgency (1-5)', 'y': 'Count'},
                color=urgency_counts.index,
                color_continuous_scale=['green', 'yellow', 'orange', 'red', 'darkred']
            )
            st.plotly_chart(fig6, use_container_width=True)

# ============================================================
# TAB 4: ABOUT
# ============================================================

with tab4:
    st.header("ℹ️ About This System")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        ## 🎯 Project Overview

        AI-powered Supply Chain Management System for pharmaceutical distribution in India.

        **Key Features:**
        - 🤖 Fine-tuned Llama-3-8B model (GGUF format)
        - 📚 RAG with 100+ pharmaceutical suppliers
        - 📰 Real-time news from GDELT
        - 📊 Analytics dashboard

        ## 🏥 Company Profile

        **MediCare Pharmaceuticals India Pvt. Ltd.**
        - Revenue: ₹500 crore annually
        - Customers: 2,500 hospitals, 8,000 pharmacies
        - Products: Antibiotics, Insulin, Vaccines, Cardiac drugs
        - Location: Bangalore, Karnataka
        - Regulatory: CDSCO licensed, WHO-GDP certified
        """)

    with col2:
        st.markdown("""
        ## 🔧 Technology Stack

        **AI Model:**
        - Base: Llama-3-8B
        - Format: GGUF (q4_k_m quantization)
        - Fine-tuned on 300 pharmaceutical scenarios

        **Vector Database:**
        - FAISS (Facebook AI Similarity Search)
        - 100 supplier profiles
        - Embeddings: sentence-transformers/all-MiniLM-L6-v2

        **Data Sources:**
        - Training: 300 synthetic pharma scenarios
        - RAG: 100 pharmaceutical suppliers
        - News: GDELT Global Knowledge Graph

        **Frontend:**
        - Framework: Streamlit
        - Visualization: Plotly
        - Deployment: Local
        """)

    st.markdown("---")

    st.markdown("""
    ## 📚 Features

    ### 1. AI Agent Chat
    - Ask questions about supply chain disruptions
    - Get supplier recommendations from database
    - Regulatory compliance guidance (CDSCO)
    - Cost analysis and timelines

    ### 2. News Monitor
    - Real-time pharmaceutical news (GDELT)
    - Categorized by event type
    - Urgency scoring (1-5)
    - AI-powered impact analysis

    ### 3. Analytics Dashboard
    - Inventory visualization
    - Supplier distribution analysis
    - Geographic diversity metrics
    - News trend tracking

    ## 🎓 Academic Context

    **Project Type:** GenAI Course Project

    **Datasets:**
    - Training: 300 examples (team-created)
    - Suppliers: 100 profiles (generated)
    - Testing: GDELT news (external source)

    **Literature:**
    - LLMs for supply chain optimization
    - RAG for business intelligence
    - Pharmaceutical inventory management
    """)

    st.markdown("---")

    # System status
    st.subheader("🔧 System Status")

    col1, col2, col3 = st.columns(3)

    with col1:
        if agent:
            st.success("✅ AI Agent: Online")
            st.info("Model: GGUF (llama-cpp-python)")
        else:
            st.error("❌ AI Agent: Offline")
            if error:
                st.error(f"Error: {error}")

    with col2:
        if news_df is not None:
            st.success(f"✅ News: {len(news_df)} articles loaded")
        else:
            st.warning("⚠️ News: Not loaded")

    with col3:
        if suppliers_df is not None:
            st.success(f"✅ Suppliers: {len(suppliers_df)} in database")
        else:
            st.warning("⚠️ Suppliers: Not loaded")

# ============================================================
# FOOTER
# ============================================================

st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 20px;">
    <p><strong>MediCare Pharmaceuticals India Pvt. Ltd.</strong></p>
    <p>Supply Chain Intelligence System | Powered by AI | Real-time Data</p>
    <p>© 2026 | For Academic Use Only</p>
</div>
""", unsafe_allow_html=True)
