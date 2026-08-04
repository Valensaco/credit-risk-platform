import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Single Prediction", page_icon="📝", layout="wide")

# Raw feature names mapped to labels a non-technical user can understand
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

@st.cache_resource
def load_models():
    model_path = "models/"
    data_path = "data/"
    with open(model_path + "xgboost_best.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(model_path + "lightgbm_best.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(model_path + "isolation_forest.pkl", "rb") as f:
        iso = pickle.load(f)
    # XGBoost chosen for SHAP because TreeExplainer is faster and more accurate
    # with gradient boosted trees than with LightGBM in this setup
    with open(model_path + "shap_explainer_xgb.pkl", "rb") as f:
        explainer = pickle.load(f)
    with open(data_path + "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return xgb, lgbm, iso, explainer,
