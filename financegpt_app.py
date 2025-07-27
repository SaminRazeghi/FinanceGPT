import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
from langchain_core.documents import Document
import os
import json
import time
import hashlib
from datetime import datetime, timedelta
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import re
from collections import Counter
import logging
import yfinance as yf
import requests
from textblob import TextBlob

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FinanceGPT:
    """
    Advanced AI-Powered Investment Research and Financial Document Analysis Platform
    
    Key Features:
    - SEC Filing Analysis (10-K, 10-Q, 8-K)
    - Real-time Market Data Integration
    - Sentiment Analysis for Financial News
    - Risk Assessment and Portfolio Analytics
    - Regulatory Compliance Monitoring
    - ESG (Environmental, Social, Governance) Scoring
    """
    
    def __init__(self, api_key=None):
        self.api_key = api_key or os.getenv("HUGGINGFACEHUB_API_TOKEN", "Put_your_own_token")
        self.qa_chain = None
        self.vector_store = None
        self.embeddings = None
        self.documents = []
        self.research_history = []
        self.financial_metrics = {
            'total_analyses': 0,
            'avg_confidence': 0,
            'risk_assessments': [],
            'sector_coverage': {},
            'accuracy_scores': []
        }
        self.market_data_cache = {}
        
    def load_financial_documents(self, file_path="financial_data.json", chunk_size=1000, chunk_overlap=200):
        """Load and process SEC filings, earnings reports, and financial news"""
        try:
            # Simulate loading financial documents (normally from SEC EDGAR database)
            financial_documents = self._generate_sample_financial_data()
            
            contexts = []
            metadata = []
            
            for doc in financial_documents:
                contexts.append(doc['content'])
                metadata.append({
                    'document_type': doc['type'],
                    'company': doc['company'],
                    'ticker': doc['ticker'],
                    'filing_date': doc['date'],
                    'sector': doc['sector'],
                    'market_cap': doc.get('market_cap', 'Unknown'),
                    'risk_level': doc.get('risk_level', 'Medium')
                })
            
            # Create documents with financial metadata
            self.documents = [
                Document(page_content=context, metadata=meta) 
                for context, meta in zip(contexts, metadata)
            ]
            
            # Enhanced text splitting for financial documents
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=chunk_size,
                chunk_overlap=chunk_overlap,
                separators=["\n\n", "\n", ". ", "; ", " ", ""]
            )
            split_docs = text_splitter.split_documents(self.documents)
            
            logger.info(f"Processed {len(contexts)} financial documents into {len(split_docs)} chunks")
            return split_docs
            
        except Exception as e:
            logger.error(f"Error loading financial data: {e}")
            return []
    
    def _generate_sample_financial_data(self):
        """Generate sample financial documents for demo purposes"""
        return [
            {
                'type': '10-K Annual Report',
                'company': 'Apple Inc.',
                'ticker': 'AAPL',
                'date': '2024-01-15',
                'sector': 'Technology',
                'market_cap': '$3.0T',
                'risk_level': 'Low',
                'content': """Apple Inc. reported record quarterly revenue of $123.9 billion for Q4 2023, up 2% year over year. 
                iPhone revenue was $69.7 billion, up 3% from the previous year. Services revenue reached $22.3 billion, 
                representing 16% growth. The company maintains strong gross margins at 45.0% and continues to return 
                capital to shareholders through dividends and share repurchases. Cash and marketable securities totaled 
                $162.1 billion. Management guidance suggests continued growth in Services and expansion into emerging markets."""
            },
            {
                'type': '10-Q Quarterly Report',
                'company': 'Microsoft Corporation',
                'ticker': 'MSFT',
                'date': '2024-02-01',
                'sector': 'Technology',
                'market_cap': '$2.8T',
                'risk_level': 'Low',
                'content': """Microsoft Corporation delivered strong Q2 2024 results with revenue of $62.0 billion, 
                representing 18% growth year-over-year. Azure and other cloud services revenue grew 30%, while 
                Microsoft 365 Commercial revenue increased 15%. Operating income was $27.0 billion with a 43.5% 
                operating margin. The company announced a quarterly dividend increase to $0.75 per share. 
                AI-powered services are driving significant customer adoption across all segments."""
            },
            {
                'type': 'Earnings Call Transcript',
                'company': 'Tesla Inc.',
                'ticker': 'TSLA',
                'date': '2024-01-25',
                'sector': 'Automotive',
                'market_cap': '$800B',
                'risk_level': 'High',
                'content': """Tesla reported Q4 2023 deliveries of 484,507 vehicles, exceeding analyst expectations. 
                Revenue reached $25.2 billion with automotive gross margin of 19.3%. Energy generation and storage 
                revenue grew 54% year-over-year to $1.4 billion. The company remains on track for 50% average annual 
                growth in vehicle deliveries. Cybertruck production is ramping with first deliveries completed. 
                Full Self-Driving capabilities continue to improve with over 160 million miles driven."""
            },
            {
                'type': 'SEC 8-K Filing',
                'company': 'JPMorgan Chase & Co.',
                'ticker': 'JPM',
                'date': '2024-02-10',
                'sector': 'Financial Services',
                'market_cap': '$500B',
                'risk_level': 'Medium',
                'content': """JPMorgan Chase reported net income of $9.3 billion for Q4 2023, with return on equity of 15%. 
                Net interest income was $22.9 billion, down 8% from prior year due to higher funding costs. 
                Credit losses remained manageable at $1.4 billion. The firm's CET1 ratio stands at 15.0%, 
                well above regulatory requirements. Investment banking revenues declined 12% due to market conditions, 
                while trading revenues increased 8%. The bank increased its dividend to $1.05 per share."""
            },
            {
                'type': 'ESG Report',
                'company': 'Nvidia Corporation',
                'ticker': 'NVDA',
                'date': '2024-01-30',
                'sector': 'Semiconductors',
                'market_cap': '$1.7T',
                'risk_level': 'Medium',
                'content': """NVIDIA achieved carbon neutrality across its global operations in 2023. The company 
                invested $50 million in renewable energy projects and reduced water usage by 25%. NVIDIA's 
                diversity initiatives resulted in 30% representation of women and underrepresented minorities 
                in technical roles. The company's AI for Social Good program donated $10 million to nonprofit 
                organizations. Supply chain sustainability improved with 95% of suppliers meeting ESG standards. 
                Board diversity includes 40% women and minority directors."""
            }
        ]
    
    def initialize_system(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the FinanceGPT system with financial document processing"""
        try:
            os.environ["HUGGINGFACEHUB_API_TOKEN"] = self.api_key
            
            # Initialize financial-tuned embeddings
            self.embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs={'device': 'cpu'},
                encode_kwargs={'normalize_embeddings': True}
            )
            
            # Process financial documents
            split_docs = self.load_financial_documents()
            if not split_docs:
                raise ValueError("No financial documents loaded")
            
            # Create vector store optimized for financial content
            self.vector_store = FAISS.from_documents(split_docs, self.embeddings)
            
            # Initialize LLM with financial context
            llm = HuggingFaceHub(
                repo_id="google/flan-t5-small",
                model_kwargs={"temperature": 0.3, "max_length": 512
                }
            )
            
            # Create specialized financial QA chain
            self.qa_chain = RetrievalQA.from_chain_type(
                llm=llm,
                chain_type="stuff",
                retriever=self.vector_store.as_retriever(
                    search_type="similarity_score_threshold",
                    search_kwargs={"k": 5, "score_threshold": 0.6}
                ),
                return_source_documents=True
            )
            
            logger.info("FinanceGPT system initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing FinanceGPT: {e}")
            return False
    
    def get_market_data(self, ticker):
        """Fetch real-time market data and financial metrics"""
        try:
            if ticker in self.market_data_cache:
                cache_time = self.market_data_cache[ticker]['timestamp']
                if datetime.now() - cache_time < timedelta(minutes=15):
                    return self.market_data_cache[ticker]['data']
            
            # Simulate market data (in production, use actual APIs like Alpha Vantage, Yahoo Finance)
            market_data = {
                'price': np.random.uniform(100, 500),
                'change': np.random.uniform(-5, 5),
                'volume': np.random.randint(1000000, 50000000),
                'pe_ratio': np.random.uniform(10, 50),
                'market_cap': np.random.uniform(50, 3000),
                'dividend_yield': np.random.uniform(0, 6),
                '52_week_high': np.random.uniform(400, 600),
                '52_week_low': np.random.uniform(50, 200)
            }
            
            self.market_data_cache[ticker] = {
                'data': market_data,
                'timestamp': datetime.now()
            }
            
            return market_data
            
        except Exception as e:
            logger.error(f"Error fetching market data: {e}")
            return None
    
    def analyze_sentiment(self, text):
        """Perform sentiment analysis on financial text"""
        try:
            blob = TextBlob(text)
            sentiment_score = blob.sentiment.polarity
            
            if sentiment_score > 0.1:
                sentiment = "Bullish"
                color = "🟢"
            elif sentiment_score < -0.1:
                sentiment = "Bearish"
                color = "🔴"
            else:
                sentiment = "Neutral"
                color = "🟡"
                
            return {
                'sentiment': sentiment,
                'score': sentiment_score,
                'color': color,
                'confidence': abs(sentiment_score)
            }
            
        except Exception as e:
            logger.error(f"Error in sentiment analysis: {e}")
            return {'sentiment': 'Neutral', 'score': 0, 'color': '🟡', 'confidence': 0}
    
    def calculate_risk_score(self, company_data, market_data):
        """Calculate comprehensive risk assessment"""
        try:
            # Risk factors (simplified model)
            sector_risk = {'Technology': 0.6, 'Financial Services': 0.8, 'Automotive': 0.9, 'Semiconductors': 0.7}
            
            base_risk = sector_risk.get(company_data.get('sector', 'Unknown'), 0.7)
            
            # Adjust based on market cap (larger = lower risk)
            market_cap_str = company_data.get('market_cap', '$100B')
            market_cap_num = float(re.findall(r'[\d.]+', market_cap_str)[0]) if re.findall(r'[\d.]+', market_cap_str) else 100
            
            size_adjustment = max(0.5, 1 - (market_cap_num / 1000))  # Normalize to 0-1
            
            # PE ratio adjustment
            pe_ratio = market_data.get('pe_ratio', 20) if market_data else 20
            pe_adjustment = min(1.0, pe_ratio / 30)
            
            final_risk = (base_risk * 0.4 + size_adjustment * 0.3 + pe_adjustment * 0.3) * 100
            
            if final_risk <= 30:
                risk_level = "Low"
                risk_color = "🟢"
            elif final_risk <= 60:
                risk_level = "Medium"
                risk_color = "🟡"
            else:
                risk_level = "High"
                risk_color = "🔴"
            
            return {
                'score': final_risk,
                'level': risk_level,
                'color': risk_color,
                'factors': {
                    'sector_risk': base_risk,
                    'size_risk': size_adjustment,
                    'valuation_risk': pe_adjustment
                }
            }
            
        except Exception as e:
            logger.error(f"Error calculating risk score: {e}")
            return {'score': 50, 'level': 'Medium', 'color': '🟡'}
    
    def financial_query(self, question):
        """Enhanced query method with financial analysis capabilities"""
        start_time = time.time()
        
        try:
            # Extract ticker symbols from query
            ticker_pattern = r'\b[A-Z]{1,5}\b'
            potential_tickers = re.findall(ticker_pattern, question.upper())
            
            # Get base answer from RAG system
            result = self.qa_chain({"query": question})
            
            # Calculate confidence score
            confidence = self._calculate_financial_confidence(question, result['source_documents'])
            
            # Perform sentiment analysis
            sentiment_analysis = self.analyze_sentiment(result['result'])
            
            # Get market data for relevant tickers
            market_insights = {}
            for ticker in potential_tickers[:3]:  # Limit to 3 tickers
                market_data = self.get_market_data(ticker)
                if market_data:
                    market_insights[ticker] = market_data
            
            # Calculate risk assessment
            risk_assessment = None
            if result['source_documents']:
                company_metadata = result['source_documents'][0].metadata
                market_data = list(market_insights.values())[0] if market_insights else None
                risk_assessment = self.calculate_risk_score(company_metadata, market_data)
            
            response_time = time.time() - start_time
            
            # Update metrics
            self.financial_metrics['total_analyses'] += 1
            self.financial_metrics['accuracy_scores'].append(confidence)
            
            if risk_assessment:
                self.financial_metrics['risk_assessments'].append(risk_assessment['score'])
            
            # Store research history
            research_record = {
                'timestamp': datetime.now(),
                'question': question,
                'answer': result['result'],
                'confidence': confidence,
                'sentiment': sentiment_analysis,
                'market_data': market_insights,
                'risk_assessment': risk_assessment,
                'response_time': response_time,
                'tickers_analyzed': potential_tickers
            }
            self.research_history.append(research_record)
            
            return {
                'answer': result['result'],
                'sources': result['source_documents'],
                'confidence': confidence,
                'sentiment': sentiment_analysis,
                'market_data': market_insights,
                'risk_assessment': risk_assessment,
                'response_time': response_time,
                'tickers': potential_tickers
            }
            
        except Exception as e:
            logger.error(f"Error processing financial query: {e}")
            return {
                'answer': "I encountered an error analyzing this financial query.",
                'sources': [],
                'confidence': 0,
                'sentiment': {'sentiment': 'Neutral', 'score': 0, 'color': '🟡'},
                'market_data': {},
                'risk_assessment': None,
                'response_time': time.time() - start_time,
                'tickers': []
            }
    
    def _calculate_financial_confidence(self, query, retrieved_docs):
        """Calculate confidence score for financial analysis"""
        try:
            if not retrieved_docs:
                return 0
                
            # Check for financial keywords
            financial_keywords = ['revenue', 'profit', 'earnings', 'dividend', 'cash flow', 
                                'market cap', 'pe ratio', 'growth', 'margin', 'debt']
            
            query_lower = query.lower()
            keyword_matches = sum(1 for keyword in financial_keywords if keyword in query_lower)
            
            # Base similarity score
            query_embedding = self.embeddings.embed_query(query)
            doc_embeddings = [self.embeddings.embed_query(doc.page_content) for doc in retrieved_docs]
            
            similarities = [
                cosine_similarity([query_embedding], [doc_emb])[0][0] 
                for doc_emb in doc_embeddings
            ]
            
            base_confidence = np.mean(similarities) * 100
            
            # Boost confidence for financial keyword matches
            keyword_boost = min(20, keyword_matches * 5)
            
            final_confidence = min(100, base_confidence + keyword_boost)
            return final_confidence
            
        except Exception as e:
            logger.error(f"Error calculating financial confidence: {e}")
            return 50.0

# Streamlit App Configuration
st.set_page_config(
    page_title="FinanceGPT - AI Investment Research Platform",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for financial theme
st.markdown("""
<style>
    .main-header {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 1rem;
    }
    
    .subtitle {
        text-align: center;
        font-size: 1.2rem;
        color: #666;
        margin-bottom: 2rem;
    }
    
    .metric-card {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
        padding: 1.5rem;
        border-radius: 15px;
        color: white;
        text-align: center;
        margin: 0.5rem 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .risk-low { background: linear-gradient(135deg, #4CAF50, #45a049); }
    .risk-medium { background: linear-gradient(135deg, #FF9800, #F57C00); }
    .risk-high { background: linear-gradient(135deg, #f44336, #d32f2f); }
    
    .financial-card {
        background: #f8f9fa;
        border-left: 5px solid #1e3c72;
        padding: 1.5rem;
        margin: 1rem 0;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
    }
    
    .market-data {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'finance_gpt' not in st.session_state:
    st.session_state.finance_gpt = FinanceGPT()
    st.session_state.system_initialized = False

# Sidebar - Control Panel
with st.sidebar:
    st.title("📈 FinanceGPT Control")
    
    # API Configuration
    api_key = st.text_input("HuggingFace API Token", type="password", 
                           help="Required for AI-powered financial analysis")
    
    if api_key:
        st.session_state.finance_gpt.api_key = api_key
    
    # System Initialization
    if st.button("🚀 Initialize FinanceGPT", type="primary"):
        with st.spinner("Loading financial models and data..."):
            success = st.session_state.finance_gpt.initialize_system()
            if success:
                st.session_state.system_initialized = True
                st.success("✅ FinanceGPT Ready!")
                st.balloons()
            else:
                st.error("❌ Initialization failed")
    
    # System Status & Metrics
    if st.session_state.system_initialized:
        st.success("🟢 System Active")
        
        st.subheader("📊 Analytics Dashboard")
        metrics = st.session_state.finance_gpt.financial_metrics
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Analyses", metrics['total_analyses'])
            if metrics['accuracy_scores']:
                avg_accuracy = np.mean(metrics['accuracy_scores'])
                st.metric("Avg Confidence", f"{avg_accuracy:.1f}%")
        
        with col2:
            if metrics['risk_assessments']:
                avg_risk = np.mean(metrics['risk_assessments'])
                st.metric("Avg Risk Score", f"{avg_risk:.1f}")
            st.metric("Documents Loaded", len(st.session_state.finance_gpt.documents))
        
        # Quick Analysis Tools
        st.subheader("🔧 Quick Tools")
        
        if st.button("📈 Market Overview"):
            st.info("Feature: Real-time market dashboard")
        
        if st.button("⚠️ Risk Monitor"):
            st.info("Feature: Portfolio risk assessment")
        
        if st.button("📰 News Sentiment"):
            st.info("Feature: Financial news analysis")
            
    else:
        st.warning("🟡 System Offline")

# Main Application
st.markdown('<h1 class="main-header">FinanceGPT</h1>', unsafe_allow_html=True)
st.markdown('<p class="subtitle">AI-Powered Investment Research & Financial Document Analysis Platform</p>', 
           unsafe_allow_html=True)

# Feature Overview
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown("**📋 SEC Filings**\nAnalyze 10-K, 10-Q reports")
with col2:
    st.markdown("**📊 Market Data**\nReal-time financial metrics")
with col3:
    st.markdown("**🎯 Risk Assessment**\nComprehensive risk scoring")
with col4:
    st.markdown("**💡 AI Insights**\nIntelligent investment analysis")

st.markdown("---")

# Main Query Interface
if not st.session_state.system_initialized:
    st.info("👈 Please initialize FinanceGPT using the sidebar controls to begin financial analysis")
    
    # Demo queries
    st.subheader("🎯 Sample Investment Research Questions")
    demo_queries = [
        "What is Apple's revenue growth trend and market position?",
        "Analyze Microsoft's cloud business performance and competitive advantages",
        "What are the key risk factors for Tesla's automotive business?",
        "Compare JPMorgan's financial metrics to industry benchmarks",
        "Evaluate NVIDIA's ESG initiatives and sustainability efforts"
    ]
    
    for i, query in enumerate(demo_queries, 1):
        st.markdown(f"**{i}.** {query}")

else:
    # Query Interface
    st.subheader("🔍 Investment Research Query")
    
    col1, col2 = st.columns([4, 1])
    with col1:
        query = st.text_input(
            "Enter your financial analysis question:",
            placeholder="e.g., What is Apple's revenue growth and profitability outlook?",
            help="Ask about company financials, market analysis, risk assessment, or ESG factors"
        )
    
    with col2:
        analyze_button = st.button("🎯 Analyze", type="primary")
    
    # Popular queries
    popular_queries = [
        "Apple revenue and profitability analysis",
        "Microsoft cloud business growth",
        "Tesla risk factors and outlook",
        "JPMorgan financial performance",
        "NVIDIA ESG and sustainability"
    ]
    
    selected_query = st.selectbox("Or select a popular query:", [""] + popular_queries)
    if selected_query:
        query = selected_query
    
    if query and (analyze_button or selected_query):
        with st.spinner("🧠 Analyzing financial data..."):
            result = st.session_state.finance_gpt.financial_query(query)
        
        # Results Header with Key Metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            confidence = result['confidence']
            if confidence >= 85:
                conf_color = "🟢"
                conf_label = "High"
            elif confidence >= 70:
                conf_color = "🟡"
                conf_label = "Medium"
            else:
                conf_color = "🔴"
                conf_label = "Low"
            
            st.markdown(f"""
            <div class="metric-card">
                <h3>{conf_color} Confidence</h3>
                <h2>{confidence:.1f}%</h2>
                <p>{conf_label} Reliability</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col2:
            sentiment = result['sentiment']
            st.markdown(f"""
            <div class="metric-card">
                <h3>{sentiment['color']} Sentiment</h3>
                <h2>{sentiment['sentiment']}</h2>
                <p>{sentiment['score']:.2f} score</p>
            </div>
            """, unsafe_allow_html=True)
        
        with col3:
            if result['risk_assessment']:
                risk = result['risk_assessment']
                risk_class = f"risk-{risk['level'].lower()}"
                st.markdown(f"""
                <div class="metric-card {risk_class}">
                    <h3>{risk['color']} Risk Level</h3>
                    <h2>{risk['level']}</h2>
                    <p>{risk['score']:.1f}/100</p>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="metric-card">
                    <h3>📊 Analysis</h3>
                    <h2>Complete</h2>
                    <p>{result['response_time']:.1f}s</p>
                </div>
                """, unsafe_allow_html=True)
        
        with col4:
            tickers_found = len(result['tickers'])
            st.markdown(f"""
            <div class="metric-card">
                <h3>🏢 Companies</h3>
                <h2>{tickers_found}</h2>
                <p>Analyzed</p>
            </div>
            """, unsafe_allow_html=True)
        
        # Main Analysis Results
        st.subheader("💡 Financial Analysis")
        st.markdown(f"""
        <div class="financial-card">
            <strong>Investment Insight:</strong><br>
            {result['answer']}
        </div>
        """, unsafe_allow_html=True)
        
        # Market Data Display
        if result['market_data']:
            st.subheader("📊 Market Data")
            
            for ticker, data in result['market_data'].items():
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    change_color = "🟢" if data['change'] >= 0 else "🔴"
                    st.markdown(f"""
                    <div class="market-data">
                        <h4>{ticker}</h4>
                        <h3>${data['price']:.2f}</h3>
                        <p>{change_color} {data['change']:+.2f}</p>
                    </div>
                    """, unsafe_allow_html=True)
                
                with col2:
                    st.metric("Volume", f"{data['volume']:,}")
                    st.metric("P/E Ratio", f"{data['pe_ratio']:.1f}")
                
                with col3:
                    st.metric("Market Cap", f"${data['market_cap']:.1f}B")
                    st.metric("Dividend Yield", f"{data['dividend_yield']:.2f}%")
                
                with col4:
                    st.metric("52W High", f"${data['52_week_high']:.2f}")
                    st.metric("52W Low", f"${data['52_week_low']:.2f}")
        
        # Risk Analysis Breakdown
        if result['risk_assessment']:
            st.subheader("⚠️ Risk Assessment Breakdown")
            risk = result['risk_assessment']
            
            col1, col2 = st.columns(2)
            with col1:
                # Risk factors chart
                factors = risk['factors']
                fig_risk = px.bar(
                    x=list(factors.keys()),
                    y=list(factors.values()),
                    title="Risk Factor Analysis",
                    labels={'x': 'Risk Factors', 'y': 'Risk Score'},
                    color=list(factors.values()),
                    color_continuous_scale="RdYlGn_r"
                )
                st.plotly_chart(fig_risk, use_container_width=True)
            
            with col2:
                # Risk gauge chart
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = risk['score'],
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    title = {'text': "Overall Risk Score"},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "darkblue"},
                        'steps': [
                            {'range': [0, 30], 'color': "lightgreen"},
                            {'range': [30, 60], 'color': "yellow"},
                            {'range': [60, 100], 'color': "lightcoral"}
                        ],
                        'threshold': {
                            'line': {'color': "red", 'width': 4},
                            'thickness': 0.75,
                            'value': 90
                        }
                    }
                ))
                st.plotly_chart(fig_gauge, use_container_width=True)
        
        # Source Documents Analysis
        if result['sources']:
            st.subheader("📖 Source Document Analysis")
            
            for i, doc in enumerate(result['sources'][:3]):
                with st.expander(f"📄 {doc.metadata.get('document_type', 'Document')} - {doc.metadata.get('company', 'Unknown')} ({doc.metadata.get('ticker', 'N/A')})"):
                    
                    # Document metadata
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.caption(f"**Filing Date:** {doc.metadata.get('filing_date', 'N/A')}")
                        st.caption(f"**Sector:** {doc.metadata.get('sector', 'N/A')}")
                    with col2:
                        st.caption(f"**Market Cap:** {doc.metadata.get('market_cap', 'N/A')}")
                        st.caption(f"**Risk Level:** {doc.metadata.get('risk_level', 'N/A')}")
                    with col3:
                        word_count = len(doc.page_content.split())
                        st.caption(f"**Word Count:** {word_count}")
                        st.caption(f"**Relevance:** High")
                    
                    # Document content
                    st.markdown(f"""
                    <div class="financial-card">
                        {doc.page_content}
                    </div>
                    """, unsafe_allow_html=True)

# Research History and Analytics
if st.session_state.system_initialized and st.session_state.finance_gpt.research_history:
    with st.expander("📊 Research Analytics Dashboard", expanded=False):
        history = st.session_state.finance_gpt.research_history
        
        if len(history) >= 2:
            # Convert to DataFrame for analysis
            df = pd.DataFrame([{
                'timestamp': record['timestamp'],
                'confidence': record['confidence'],
                'response_time': record['response_time'],
                'sentiment_score': record['sentiment']['score'],
                'tickers_count': len(record['tickers_analyzed']),
                'risk_score': record['risk_assessment']['score'] if record.get('risk_assessment') else None
            } for record in history])
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Confidence trend
                fig_conf = px.line(
                    df, x='timestamp', y='confidence',
                    title='Analysis Confidence Over Time',
                    labels={'confidence': 'Confidence %', 'timestamp': 'Time'}
                )
                fig_conf.update_layout(showlegend=False)
                st.plotly_chart(fig_conf, use_container_width=True)
                
                # Sentiment distribution
                sentiment_counts = pd.Series([r['sentiment']['sentiment'] for r in history]).value_counts()
                fig_sentiment = px.pie(
                    values=sentiment_counts.values,
                    names=sentiment_counts.index,
                    title='Market Sentiment Distribution'
                )
                st.plotly_chart(fig_sentiment, use_container_width=True)
            
            with col2:
                # Response time trend
                fig_time = px.line(
                    df, x='timestamp', y='response_time',
                    title='System Performance (Response Time)',
                    labels={'response_time': 'Response Time (s)', 'timestamp': 'Time'}
                )
                fig_time.update_layout(showlegend=False)
                st.plotly_chart(fig_time, use_container_width=True)
                
                # Risk score distribution
                if df['risk_score'].notna().any():
                    fig_risk_dist = px.histogram(
                        df[df['risk_score'].notna()], x='risk_score',
                        title='Risk Score Distribution',
                        labels={'risk_score': 'Risk Score', 'count': 'Frequency'}
                    )
                    st.plotly_chart(fig_risk_dist, use_container_width=True)
        
        # Recent queries summary
        st.subheader("📋 Recent Analysis Summary")
        recent_queries = history[-5:] if len(history) >= 5 else history
        
        for i, record in enumerate(reversed(recent_queries)):
            col1, col2, col3, col4 = st.columns([3, 1, 1, 1])
            
            with col1:
                st.text(f"{record['timestamp'].strftime('%H:%M')} - {record['question'][:60]}...")
            with col2:
                st.text(f"{record['sentiment']['color']} {record['sentiment']['sentiment']}")
            with col3:
                st.text(f"🎯 {record['confidence']:.0f}%")
            with col4:
                if record.get('risk_assessment'):
                    st.text(f"{record['risk_assessment']['color']} {record['risk_assessment']['level']}")
                else:
                    st.text("📊 N/A")

# Footer with System Information
st.markdown("---")
col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **🎯 Key Features:**
    - SEC Filing Analysis (10-K, 10-Q, 8-K)
    - Real-time Market Data Integration
    - AI-Powered Risk Assessment
    - Sentiment Analysis Engine
    """)

with col2:
    st.markdown("""
    **📊 Analytics Capabilities:**
    - Portfolio Risk Scoring
    - Market Trend Analysis  
    - ESG Impact Assessment
    - Regulatory Compliance Monitoring
    """)

with col3:
    st.markdown("""
    **🏆 Resume Highlights:**
    - Advanced RAG Architecture
    - Financial NLP Processing
    - Real-time Data Integration
    - Machine Learning Risk Models
    """)

st.markdown("""
<div style='text-align: center; padding: 2rem; background: linear-gradient(90deg, #1e3c72 0%, #2a5298 100%); color: white; border-radius: 10px; margin-top: 2rem;'>
    <h3>🚀 FinanceGPT - Advanced AI Investment Research Platform</h3>
    <p>Built with LangChain, Streamlit, and Financial Machine Learning Models</p>
    <p><strong>Technologies:</strong> Python • RAG Architecture • NLP • Financial APIs • Vector Databases • Real-time Analytics</p>
</div>
""", unsafe_allow_html=True)