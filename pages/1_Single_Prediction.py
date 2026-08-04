
import streamlit as st
import pandas as pd
import numpy as np
import pickle
import plotly.graph_objects as go

st.set_page_config(page_title="Single Prediction", page_icon="📝", layout="wide")

# Human readable feature names
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
    model_path = "/content/drive/MyDrive/credit_risk_platform/models/"
    data_path = "/content/drive/MyDrive/credit_risk_platform/data/"
    with open(model_path + "xgboost_best.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(model_path + "lightgbm_best.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(model_path + "isolation_forest.pkl", "rb") as f:
        iso = pickle.load(f)
    with open(model_path + "shap_explainer_xgb.pkl", "rb") as f:
        explainer = pickle.load(f)
    with open(data_path + "feature_names.pkl", "rb") as f:
        feature_names = pickle.load(f)
    return xgb, lgbm, iso, explainer, feature_names

xgb, lgbm, iso, explainer, feature_names = load_models()

def prepare_features(inputs):
    data = inputs.copy()
    data["loan_to_income"] = data["loan_amnt"] / (data["annual_inc"] + 1)
    data["installment_to_income"] = data["installment"] / (data["annual_inc"] / 12 + 1)
    data["dti_adjusted"] = data["dti"] / (data["annual_inc"] / 12 + 1)
    data["has_delinq"] = int(data["delinq_2yrs"] > 0)
    data["has_pub_rec"] = int(data["pub_rec"] > 0)
    data["has_bankruptcy"] = int(data["pub_rec_bankruptcies"] > 0)
    data["high_inq"] = int(data["inq_last_6mths"] > 2)
    data["revol_util_band"] = min(int(data["revol_util"] / 25), 3)
    data["fico_band"] = min(int((data["fico_score"] - 580) / 54), 4) if data["fico_score"] > 580 else 0
    data["loan_amnt_band"] = min(int(data["loan_amnt"] / 8000), 4)
    data["funded_amnt"] = data["loan_amnt"]
    df = pd.DataFrame([data])
    for col in feature_names:
        if col not in df.columns:
            df[col] = 0
    return df[feature_names]

st.markdown("# 📝 Single Application Assessment")
st.markdown("Enter loan application details below to get an instant risk assessment.")
st.markdown("---")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("#### 💰 Loan Details")
    loan_amnt = st.slider("Loan Amount ($)", 1000, 40000, 15000, 500)
    term = st.selectbox("Term (months)", [36, 60])
    int_rate = st.slider("Interest Rate (%)", 5.0, 30.0, 12.5, 0.1)
    installment = st.slider("Monthly Installment ($)", 50, 1500, 500, 10)
    purpose = st.selectbox("Purpose", [0, 1, 2, 3, 4],
                            format_func=lambda x: ["Debt Consolidation", "Credit Card",
                                                    "Home Improvement", "Other", "Medical"][x])

with col2:
    st.markdown("#### 👤 Applicant Info")
    annual_inc = st.slider("Annual Income ($)", 20000, 300000, 65000, 1000)
    emp_length = st.slider("Employment Length (years)", 0, 10, 5)
    home_ownership = st.selectbox("Home Ownership", [0, 1, 2],
                                   format_func=lambda x: ["Rent", "Own", "Mortgage"][x])
    verification_status = st.selectbox("Verification Status", [0, 1, 2],
                                        format_func=lambda x: ["Not Verified", "Verified", "Source Verified"][x])
    dti = st.slider("Debt-to-Income Ratio", 0.0, 40.0, 18.5, 0.1)

with col3:
    st.markdown("#### 📊 Credit History")
    fico_score = st.slider("FICO Score", 580, 850, 710, 1)
    grade = st.selectbox("Credit Grade", [0,1,2,3,4,5,6],
                          format_func=lambda x: ["A","B","C","D","E","F","G"][x])
    sub_grade = st.slider("Sub Grade", 0, 34, 10)
    inq_last_6mths = st.slider("Inquiries Last 6 Months", 0, 10, 1)
    delinq_2yrs = st.slider("Delinquencies (2 years)", 0, 10, 0)
    pub_rec = st.slider("Public Records", 0, 5, 0)
    pub_rec_bankruptcies = st.slider("Bankruptcies", 0, 3, 0)
    revol_bal = st.slider("Revolving Balance ($)", 0, 100000, 12000, 500)
    revol_util = st.slider("Revolving Utilization (%)", 0.0, 100.0, 45.0, 0.1)
    open_acc = st.slider("Open Accounts", 0, 30, 8)
    total_acc = st.slider("Total Accounts", 0, 60, 15)
    mort_acc = st.slider("Mortgage Accounts", 0, 10, 1)

st.markdown("---")

if st.button("🔍 Analyze Risk", use_container_width=True):
    inputs = {
        "loan_amnt": loan_amnt, "term": term, "int_rate": int_rate,
        "installment": installment, "grade": grade, "sub_grade": sub_grade,
        "emp_length": emp_length, "home_ownership": home_ownership,
        "annual_inc": annual_inc, "verification_status": verification_status,
        "purpose": purpose, "dti": dti, "delinq_2yrs": delinq_2yrs,
        "fico_score": fico_score, "inq_last_6mths": inq_last_6mths,
        "open_acc": open_acc, "pub_rec": pub_rec, "revol_bal": revol_bal,
        "revol_util": revol_util, "total_acc": total_acc, "mort_acc": mort_acc,
        "pub_rec_bankruptcies": pub_rec_bankruptcies
    }

    with st.spinner("Analyzing application..."):
        df = prepare_features(inputs)
        xgb_prob = xgb.predict_proba(df)[0][1]
        lgbm_prob = lgbm.predict_proba(df)[0][1]
        default_prob = (xgb_prob + lgbm_prob) / 2

        risk_label = "LOW" if default_prob < 0.25 else "HIGH" if default_prob > 0.60 else "MEDIUM"
        decision = "✅ APPROVED" if default_prob < 0.35 else "❌ DENIED" if default_prob > 0.60 else "⚠️ MANUAL REVIEW"
        fraud_score = float(np.clip(-iso.score_samples(df)[0] * 10, 0, 100))

        shap_vals = explainer.shap_values(df)
        shap_df = pd.DataFrame({
            "feature": feature_names,
            "label": [FEATURE_LABELS.get(f, f) for f in feature_names],
            "impact": shap_vals[0],
            "abs_impact": np.abs(shap_vals[0])
        }).sort_values("abs_impact", ascending=False).head(10)

    st.markdown("## 📊 Assessment Results")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Default Probability", f"{default_prob:.2%}")
    with col2:
        st.metric("Risk Label", risk_label)
    with col3:
        st.metric("Decision", decision)
    with col4:
        st.metric("Fraud Score", f"{fraud_score:.1f}/100")

    st.markdown("---")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 🎯 Risk Gauge")
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=default_prob * 100,
            title={"text": "Default Risk %"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": "red" if default_prob > 0.6 else "orange" if default_prob > 0.35 else "green"},
                "steps": [
                    {"range": [0, 35], "color": "#D1FAE5"},
                    {"range": [35, 60], "color": "#FEF3C7"},
                    {"range": [60, 100], "color": "#FEE2E2"}
                ]
            }
        ))
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 🔍 Top Risk Factors")
        colors = ["red" if v > 0 else "green" for v in shap_df["impact"]]
        fig2 = go.Figure(go.Bar(
            x=shap_df["impact"],
            y=shap_df["label"],
            orientation="h",
            marker_color=colors
        ))
        fig2.update_layout(
            xaxis_title="Impact on Risk",
            yaxis_title="",
            height=400
        )
        st.plotly_chart(fig2, use_container_width=True)

    st.markdown("#### 📋 Recommended Actions")
    if default_prob < 0.35:
        st.success(f"✅ **Approve** - Suggested Rate: {int_rate * (1 + default_prob * 0.5):.2f}% APR")
    elif default_prob > 0.60:
        st.error("❌ **Deny** - High default risk. Consider requesting additional documentation.")
    else:
        st.warning("⚠️ **Manual Review** - Borderline case. Recommend human underwriter review.")
