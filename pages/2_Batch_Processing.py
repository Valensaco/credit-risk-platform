
import streamlit as st
import pandas as pd
import numpy as np
import pickle

st.set_page_config(page_title="Batch Processing", page_icon="📊", layout="wide")

@st.cache_resource
def load_models():
    model_path = "/content/drive/MyDrive/credit_risk_platform/models/"
    data_path = "/content/drive/MyDrive/credit_risk_platform/data/"
    with open(model_path + "xgboost_best.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(model_path + "lightgbm_best.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(data_path + "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return xgb, lgbm, feature_names

xgb, lgbm, feature_names = load_models()

def prepare_features(df):
    df = df.copy()
    df["loan_to_income"] = df["loan_amnt"] / (df["annual_inc"] + 1)
    df["installment_to_income"] = df["installment"] / (df["annual_inc"] / 12 + 1)
    df["dti_adjusted"] = df["dti"] / (df["annual_inc"] / 12 + 1)
    df["has_delinq"] = (df["delinq_2yrs"] > 0).astype(int)
    df["has_pub_rec"] = (df["pub_rec"] > 0).astype(int)
    df["has_bankruptcy"] = (df["pub_rec_bankruptcies"] > 0).astype(int)
    df["high_inq"] = (df["inq_last_6mths"] > 2).astype(int)
    df["revol_util_band"] = (df["revol_util"] / 25).clip(0, 3).astype(int)
    df["fico_band"] = ((df["fico_score"] - 580) / 54).clip(0, 4).astype(int)
    df["loan_amnt_band"] = (df["loan_amnt"] / 8000).clip(0, 4).astype(int)
    df["funded_amnt"] = df["loan_amnt"]
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

display_columns = {
    "loan_amnt": "Loan Amount ($)",
    "term": "Term (months)",
    "int_rate": "Interest Rate (%)",
    "installment": "Monthly Installment ($)",
    "grade": "Credit Grade",
    "sub_grade": "Sub Grade",
    "emp_length": "Employment Length",
    "home_ownership": "Home Ownership",
    "annual_inc": "Annual Income ($)",
    "verification_status": "Verification Status",
    "purpose": "Purpose",
    "dti": "DTI Ratio",
    "delinq_2yrs": "Delinquencies (2yr)",
    "fico_score": "FICO Score",
    "inq_last_6mths": "Inquiries (6mo)",
    "open_acc": "Open Accounts",
    "pub_rec": "Public Records",
    "revol_bal": "Revolving Balance ($)",
    "revol_util": "Revolving Utilization (%)",
    "total_acc": "Total Accounts",
    "mort_acc": "Mortgage Accounts",
    "pub_rec_bankruptcies": "Bankruptcies"
}

st.markdown("# 📊 Batch Processing")
st.markdown("Upload a CSV file to process multiple loan applications at once.")
st.markdown("---")

sample_data = pd.DataFrame([{
    "loan_amnt": 15000, "term": 36, "int_rate": 12.5, "installment": 500,
    "grade": 2, "sub_grade": 10, "emp_length": 5, "home_ownership": 1,
    "annual_inc": 65000, "verification_status": 1, "purpose": 1, "dti": 18.5,
    "delinq_2yrs": 0, "fico_score": 710, "inq_last_6mths": 1, "open_acc": 8,
    "pub_rec": 0, "revol_bal": 12000, "revol_util": 45.0, "total_acc": 15,
    "mort_acc": 1, "pub_rec_bankruptcies": 0
}])

st.download_button(
    "📥 Download Sample CSV Template",
    sample_data.to_csv(index=False),
    "sample_applications.csv",
    "text/csv"
)

uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

if uploaded_file:
    df_input = pd.read_csv(uploaded_file)
    st.success(f"✅ Loaded {len(df_input)} applications")
    st.dataframe(df_input.rename(columns=display_columns).head(), use_container_width=True)

    if st.button("🚀 Process All Applications"):
        with st.spinner(f"Processing {len(df_input)} applications..."):
            progress = st.progress(0)
            df_features = prepare_features(df_input)

            xgb_prob = xgb.predict_proba(df_features)[:, 1]
            lgbm_prob = lgbm.predict_proba(df_features)[:, 1]
            default_prob = (xgb_prob + lgbm_prob) / 2

            risk_label = pd.cut(default_prob,
                                 bins=[0, 0.25, 0.60, 1.0],
                                 labels=["🟢 Low", "🟡 Medium", "🔴 High"])
            decision = np.where(default_prob < 0.35, "✅ Approved",
                       np.where(default_prob > 0.60, "❌ Denied", "⚠️ Manual Review"))

            df_results = pd.DataFrame({
                "Application #": range(1, len(df_input) + 1),
                "Loan Amount ($)": df_input["loan_amnt"],
                "Annual Income ($)": df_input["annual_inc"],
                "FICO Score": df_input["fico_score"],
                "Default Probability": [f"{p:.2%}" for p in default_prob],
                "Risk Level": risk_label,
                "Decision": decision
            })
            progress.progress(100)

        st.markdown("### 📋 Results Summary")
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Applications", len(df_results))
        with col2:
            st.metric("Approved", (decision == "✅ Approved").sum())
        with col3:
            st.metric("Denied", (decision == "❌ Denied").sum())
        with col4:
            st.metric("Manual Review", (decision == "⚠️ Manual Review").sum())

        st.markdown("### 📊 Results Table")
        st.dataframe(df_results, use_container_width=True)

        st.download_button(
            "📥 Download Results CSV",
            df_results.to_csv(index=False),
            "batch_results.csv",
            "text/csv"
        )
