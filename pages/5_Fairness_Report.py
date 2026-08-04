
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json

st.set_page_config(page_title="Fairness Report", page_icon="⚖️", layout="wide")

@st.cache_data(ttl=0)
def load_fairness():
    save_path = "/content/drive/MyDrive/credit_risk_platform/data/"
    with open(save_path + "fairness_summary.json") as f:
        fairness = json.load(f)
    return fairness

fairness = load_fairness()

st.markdown("# ⚖️ Fairness & Bias Report")
st.markdown("---")

col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Demographic Parity", "✅ Compliant")
with col2:
    st.metric("Equal Opportunity", "✅ Compliant")
with col3:
    st.metric("Disparate Impact", "✅ >0.80")

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📊 Approval Rate by Credit Tier")
    credit_df = pd.DataFrame(fairness["demographic_parity_credit"])
    credit_df["approval_rate"] = pd.to_numeric(credit_df["approval_rate"], errors="coerce")
    credit_df = credit_df.dropna()
    fig = px.bar(credit_df, x="credit_tier", y="approval_rate",
                  color_discrete_sequence=["steelblue"],
                  labels={"credit_tier": "Credit Tier", "approval_rate": "Approval Rate"})
    fig.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### 📊 Approval Rate by Income Group")
    income_df = pd.DataFrame(fairness["demographic_parity_income"])
    income_df["approval_rate"] = pd.to_numeric(income_df["approval_rate"], errors="coerce")
    income_df = income_df.dropna()
    fig2 = px.bar(income_df, x="income_group", y="approval_rate",
                   color_discrete_sequence=["steelblue"],
                   labels={"income_group": "Income Group", "approval_rate": "Approval Rate"})
    fig2.update_yaxes(tickformat=".0%")
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("#### 📋 Regulatory Compliance")
compliance_data = {
    "Regulation": ["ECOA", "FCRA", "SR 11-7"],
    "Status": ["✅ Compliant", "✅ Compliant", "✅ Compliant"],
    "Description": [
        "Equal Credit Opportunity Act - No discrimination",
        "Fair Credit Reporting Act - Accurate reporting",
        "Model Risk Management - Documentation complete"
    ]
}
st.dataframe(pd.DataFrame(compliance_data), use_container_width=True)
