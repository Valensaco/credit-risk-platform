import streamlit as st

st.set_page_config(
    page_title="Credit Risk Intelligence Platform",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Outfit for headers, Inter for body , chosen for readability in data-heavy interfaces
st.markdown("""
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@400;600;700&family=Inter:wght@400;500;600&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #0A1F44 0%, #1E293B 100%);
        padding: 2rem;
        border-radius: 12px;
        margin-bottom: 2rem;
        color: white;
    }
    
    .risk-low {
        background: #D1FAE5;
        color: #065F46;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .risk-medium {
        background: #FEF3C7;
        color: #92400E;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .risk-high {
        background: #FEE2E2;
        color: #991B1B;
        padding: 0.5rem 1rem;
        border-radius: 8px;
        font-weight: 600;
    }
    
    .stButton>button {
        background: #10B981;
        color: white;
        border-radius: 8px;
        padding: 0.5rem 2rem;
        font-weight: 600;
        border: none;
        width: 100%;
    }
    
    .stButton>button:hover {
        background: #059669;
    }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="main-header">
        <h1 style="margin:0; font-family: Outfit; font-size: 2rem;">🏦 Credit Risk Intelligence Platform</h1>
        <p style="margin:0.5rem 0 0 0; opacity:0.8;">Production-grade ML platform for real-time credit risk assessment & fraud detection</p>
    </div>
""", unsafe_allow_html=True)

# Metrics are hardcoded here based on test set evaluation
# Check 06_hyperparameter_tuning.ipynb for full evaluation details
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Model AUC-ROC", "0.7192", "+0.0036 vs baseline")
with col2:
    st.metric("Avg Processing Time", "127ms", "-23ms")
with col3:
    st.metric("Approval Rate", "76.2%", "+2.1%")
with col4:
    st.metric("Fraud Detected", "82%", "+5%")

st.markdown("---")

st.markdown("### 🚀 Quick Navigation")
col1, col2, col3 = st.columns(3)

with col1:
    st.info("📝 **Single Prediction**\nAssess individual loan applications in real-time")
    st.info("📊 **Batch Processing**\nProcess multiple applications at once via CSV upload")

with col2:
    st.info("📈 **Analytics Dashboard**\nView model performance and trends over time")
    st.info("🔍 **Model Monitoring**\nTrack drift detection and system health")

with col3:
    st.info("⚖️ **Fairness Report**\nBias detection and compliance metrics")
    st.info("🧪 **A/B Testing**\nCompare model versions and performance")

st.markdown("---")
st.markdown("**Built with:** Python • XGBoost • LightGBM • Isolation Forest • SHAP • FastAPI • Streamlit")
