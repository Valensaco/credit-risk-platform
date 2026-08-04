
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Analytics", page_icon="📈", layout="wide")

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
def load_data():
    save_path = "data/"
    model_path = "models/"

    X_test = pd.read_parquet(save_path + "X_test_best.parquet")
    y_test = pd.read_parquet(save_path + "y_test_best.parquet").squeeze()
    monitoring_df = pd.read_parquet(save_path + "monitoring_data.parquet")

    with open(model_path + "xgboost_best.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(model_path + "lightgbm_best.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(save_path + "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)

    y_prob = (xgb.predict_proba(X_test)[:, 1] + lgbm.predict_proba(X_test)[:, 1]) / 2
    return X_test, y_test, y_prob, monitoring_df, feature_names, xgb

X_test, y_test, y_prob, monitoring_df, feature_names, xgb = load_data()

st.markdown("# 📈 Analytics Dashboard")
st.markdown("---")

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("AUC-ROC", "0.7192", "+0.0036")
with col2:
    st.metric("Approval Rate", f"{(y_prob < 0.35).mean():.2%}")
with col3:
    st.metric("Avg Default Prob", f"{y_prob.mean():.2%}")
with col4:
    st.metric("Total Applications", f"{len(y_prob):,}")

st.markdown("---")

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📊 Default Probability Distribution")
    fig = px.histogram(x=y_prob, nbins=50,
                        labels={"x": "Default Probability"},
                        color_discrete_sequence=["steelblue"])
    fig.add_vline(x=0.35, line_dash="dash", line_color="red",
                   annotation_text="Decision Threshold")
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.markdown("#### 🎯 Risk Distribution")
    risk_counts = pd.cut(y_prob, bins=[0, 0.25, 0.60, 1.0],
                          labels=["LOW", "MEDIUM", "HIGH"]).value_counts()
    fig2 = px.pie(values=risk_counts.values, names=risk_counts.index,
                   color_discrete_map={"LOW": "#10B981", "MEDIUM": "#F59E0B", "HIGH": "#EF4444"})
    st.plotly_chart(fig2, use_container_width=True)

col1, col2 = st.columns(2)

with col1:
    st.markdown("#### 📈 AUC Over Time")
    fig3 = px.line(monitoring_df, x="day", y="auc",
                    labels={"day": "Day", "auc": "AUC-ROC"},
                    color_discrete_sequence=["steelblue"])
    fig3.add_hline(y=0.65, line_dash="dash", line_color="red",
                    annotation_text="Alert Threshold")
    st.plotly_chart(fig3, use_container_width=True)

with col2:
    st.markdown("#### 📊 Feature Importance")
    importances = pd.Series(xgb.feature_importances_, index=feature_names)
    importances.index = [FEATURE_LABELS.get(f, f) for f in importances.index]
    importances = importances.sort_values(ascending=True).tail(15)
    fig4 = px.bar(x=importances.values, y=importances.index,
                   orientation="h",
                   labels={"x": "Importance", "y": "Feature"},
                   color_discrete_sequence=["steelblue"])
    st.plotly_chart(fig4, use_container_width=True)
