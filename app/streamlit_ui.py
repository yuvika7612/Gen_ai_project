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
from simple_agent import PharmaSupplyChainAgent

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

@st.cache_data
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

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

if 'current_question' not in st.session_state:
    st.session_state.current_question = ''

# Load data
company, inventory = load_company_data()
news_df = load_news_data()
suppliers_df = load_suppliers_data()

# Load agent
agent, error = load_agent()

# ============================================================
# SIDEBAR - COMPANY INFO & INVENTORY STATUS
# ============================================================

st.sidebar.markdown("# 🏥 MediCare Pharmaceuticals")
st.sidebar.markdown(f"**{company['business_type']}**")
st.sidebar.markdown(f"📍 {company['headquarters']}")
st.sidebar.markdown(f"💰 Revenue: {company['annual_revenue']}")

st.sidebar.markdown("---")
st.sidebar.markdown("## 📦 Current Inventory Status")

for product_id, details in inventory['products'].items():
    # Status emoji
    if details['urgency'] == 'HIGH':
        emoji = "🔴"
    elif details['urgency'] == 'MEDIUM':
        emoji = "🟡"
    else:
        emoji = "🟢"

    # Display metric
    st.sidebar.metric(
        label=f"{emoji} {details['product_name'][:25]}...",
        value=f"{details['days_of_supply']} days",
        delta=f"{details['status']}"
    )

st.sidebar.markdown("---")

# System status
st.sidebar.markdown("## 🔧 System Status")

if agent:
    st.sidebar.success("✅ AI Agent: Online")
else:
    st.sidebar.error("❌ AI Agent: Error")
    if error:
        st.sidebar.error(f"Error: {error[:100]}...")

if news_df is not None:
    st.sidebar.success(f"✅ News Data: {len(news_df)} articles")
else:
    st.sidebar.warning("⚠️ News Data: Not loaded")

if suppliers_df is not None:
    st.sidebar.success(f"✅ Suppliers: {len(suppliers_df)} in database")

st.sidebar.markdown("---")
st.sidebar.markdown("**Last Updated:** " + inventory['last_updated'])

# ============================================================
# MAIN CONTENT
# ============================================================

st.markdown('<div class="main-header">💊 Pharmaceutical Supply Chain AI Assistant</div>', unsafe_allow_html=True)

# Create tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "🤖 AI Agent Chat",
    "📰 News Monitor",
    "📊 Analytics Dashboard",
    "ℹ️ About"
])

# ============================================================
# TAB 1: AI AGENT CHAT
# ============================================================

with tab1:
    st.header("🤖 AI Supply Chain Agent")

    if not agent:
        st.error("❌ AI Agent failed to load. Please check the error in the sidebar.")
        st.info("""
        **Common issues:**
        - GGUF model file not found at `models/llama-3-8b.Q4_K_M.gguf`
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
            st.info("""
            **Example Questions:**
            - "Find insulin suppliers in India"
            - "Who has cold chain capability?"
            - "Recommend CDSCO approved manufacturers"
            """)

        # Predefined scenarios
        st.markdown("### 📝 Quick Scenarios")

        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🇨🇳 China API Ban", use_container_width=True):
                st.session_state.current_question = "China blocks API exports for antibiotics. Find alternative suppliers urgently."

        with col2:
            if st.button("❄️ Cold Chain Emergency", use_container_width=True):
                st.session_state.current_question = "Power failure threatens vaccine storage. Find suppliers with emergency cold chain delivery."

        with col3:
            if st.button("💊 CDSCO Compliance", use_container_width=True):
                st.session_state.current_question = "Find CDSCO approved insulin manufacturers in India with fast delivery."

        # More scenarios
        col1, col2, col3 = st.columns(3)

        with col1:
            if st.button("🏭 Generic Alternative", use_container_width=True):
                st.session_state.current_question = "Find low-cost generic drug manufacturers to replace expensive imports."

        with col2:
            if st.button("🚨 Vaccine Shortage", use_container_width=True):
                st.session_state.current_question = "Need COVID vaccine suppliers urgently. Who has stock available now?"

        with col3:
            if st.button("🌡️ Temperature Control", use_container_width=True):
                st.session_state.current_question = "Find suppliers with ultra-cold chain capability for specialized vaccines."

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
        - Fine-tuned on 50 pharmaceutical scenarios

        **Vector Database:**
        - FAISS (Facebook AI Similarity Search)
        - 100 supplier profiles
        - Embeddings: sentence-transformers/all-MiniLM-L6-v2

        **Data Sources:**
        - Training: 50 synthetic pharma scenarios
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
    - Training: 50 examples (team-created)
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
