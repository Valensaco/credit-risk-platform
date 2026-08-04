
import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import json

st.set_page_config(page_title="Model Monitoring", page_icon="🔍", layout="wide")

FEATURE_LABELS = {
    "int_rate": "Interest Rate",
    "term": "Loan Term",
    "fico_score": "FICO Credit Score",
    "dti": "Debt-to-Income Ratio",
    "annual_inc": "Annual Income",
    "loan_amnt": "Loan Amount",
    "revol_util": "Credit Utilization",
    "inq_last_6mths": "Recent Credit Inquiries",
    "delinq_2yrs": "Recent Delinquencies",
    "pub_rec": "Public Records",
    "pub_rec_bankruptcies": "Bankruptcies",
    "open_acc": "Open Accounts",
    "total_acc": "Total Accounts",
    "revol_bal": "Revolving Balance",
    "mort_acc": "Mortgage Accounts",
    "emp_length": "Employment Length",
    "grade": "Credit Grade",
    "sub_grade": "Credit Sub Grade",
    "home_ownership": "Home Ownership",
    "verification_status": "Income Verification",
    "purpose": "Loan Purpose",
    "installment": "Monthly Installment",
    "funded_amnt": "Funded Amount",
    "loan_to_income": "Loan to Income Ratio",
    "installment_to_income": "Installment to Income Ratio",
    "dti_adjusted": "Adjusted DTI Ratio",
    "has_delinq": "Has Delinquency",
    "has_pub_rec": "Has Public Record",
    "has_bankruptcy": "Has Bankruptcy",
    "high_inq": "High Inquiry Count",
    "revol_util_band": "Utilization Band",
    "fico_band": "FICO Score Band",
    "loan_amnt_band": "Loan Amount Band",
    "initial_list_status": "Listing Status",
    "application_type": "Application Type"
}

@st.cache_data
def load_monitoring():
    save_path = "/content/drive/MyDrive/credit_risk_platform/data/"
    monitoring_df = pd.read_parquet(save_path + "monitoring_data.parquet")
    drift_df = pd.read_parquet(save_path + "drift_results.parquet")
    with open(save_path + "model_health.json") as f:
        health = json.load(f)
    return monitoring_df, drift_df, health

monitoring_df, drift_df, health = load_monitoring()

st.markdown("# 🔍 Model Monitoring")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Avg AUC-ROC", health["avg_auc"])
with col2:
    st.metric("Min AUC-ROC", health["min_auc"])
with col3:
    st.metric("Total Predictions", f"{health['total_predictions']:,}")
with col4:
    st.metric("Drifted Features", health["drift_features_detected"])

st.markdown("---")

col1, col2 = st.columns(2)
with col1:
    st.markdown("#### 📈 AUC Over Time")
    fig = px.line(monitoring_df, x="day", y="auc")
    fig.add_hline(y=0.65, line_dash="dash", line_color="red")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### 📊 Approval Rate Over Time")
    fig2 = px.line(monitoring_df, x="day", y="approval_rate",
                    color_discrete_sequence=["green"])
    st.plotly_chart(fig2, use_container_width=True)

st.markdown("#### 🌊 Data Drift Detection")
drift_display = drift_df.copy()
drift_display["Feature"] = drift_display["feature"].map(
    lambda x: FEATURE_LABELS.get(x, x)
)
drift_display["KS Statistic"] = drift_display["ks_statistic"]
drift_display["P-Value"] = drift_display["p_value"]
drift_display["Status"] = drift_display["drift_detected"].map({True: "⚠️ Drift Detected", False: "✅ Normal"})
st.dataframe(drift_display[["Feature", "KS Statistic", "P-Value", "Status"]], use_container_width=True)

st.markdown("#### 🚨 Alerts")
for alert in health["alerts"]:
    if alert["type"] == "CRITICAL":
        st.error(f"🔴 {alert['message']}")
    elif alert["type"] == "WARNING":
        st.warning(f"🟡 {alert['message']}")
    else:
        st.success(f"🟢 {alert['message']}")
