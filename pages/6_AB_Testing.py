
import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import pickle
from scipy import stats

st.set_page_config(page_title="A/B Testing", page_icon="🧪", layout="wide")

@st.cache_data
def load_data():
    save_path = "data/"
    model_path = "models/"
    X_test = pd.read_parquet(save_path + "X_test_best.parquet")
    y_test = pd.read_parquet(save_path + "y_test_best.parquet").squeeze()
    with open(model_path + "xgboost_best.pkl", "rb") as f:
        xgb = pickle.load(f)
    with open(model_path + "lightgbm_best.pkl", "rb") as f:
        lgbm = pickle.load(f)
    with open(model_path + "xgboost_fair.pkl", "rb") as f:
        xgb_fair = pickle.load(f)
    return X_test, y_test, xgb, lgbm, xgb_fair

X_test, y_test, xgb, lgbm, xgb_fair = load_data()

from sklearn.metrics import roc_auc_score, f1_score

prob_a = (xgb.predict_proba(X_test)[:, 1] + lgbm.predict_proba(X_test)[:, 1]) / 2
prob_b = xgb_fair.predict_proba(X_test)[:, 1]
pred_a = (prob_a >= 0.35).astype(int)
pred_b = (prob_b >= 0.35).astype(int)

st.markdown("# 🧪 A/B Testing")
st.markdown("Comparing **Model A** (Original Ensemble) vs **Model B** (Fair Model)")
st.markdown("---")

col1, col2 = st.columns(2)
metrics = {
    "AUC-ROC": [roc_auc_score(y_test, prob_a), roc_auc_score(y_test, prob_b)],
    "F1 Score": [f1_score(y_test, pred_a), f1_score(y_test, pred_b)],
    "Approval Rate": [(pred_a == 0).mean(), (pred_b == 0).mean()],
    "Avg Default Prob": [prob_a.mean(), prob_b.mean()]
}

results_df = pd.DataFrame(metrics, index=["Model A (Original)", "Model B (Fair)"]).T
results_df.columns = ["Model A", "Model B"]
results_df["Difference"] = (results_df["Model B"] - results_df["Model A"]).round(4)
results_df["Winner"] = results_df.apply(
    lambda row: "🟢 B" if row["Model B"] > row["Model A"] else "🔴 A", axis=1
)

st.markdown("#### 📊 Model Comparison")
st.dataframe(results_df.round(4))

# Statistical significance
t_stat, p_value = stats.ttest_ind(prob_a, prob_b)
st.markdown("#### 📐 Statistical Significance")
st.metric("P-Value", f"{p_value:.6f}")
if p_value < 0.05:
    st.success("✅ Difference is statistically significant (p < 0.05)")
else:
    st.warning("⚠️ Difference is NOT statistically significant")

st.markdown("#### 🎯 Recommendation")
if roc_auc_score(y_test, prob_b) >= roc_auc_score(y_test, prob_a):
    st.success("✅ **Deploy Model B** - Better or equal performance with improved fairness")
else:
    st.info("ℹ️ **Keep Model A** - Higher accuracy, but consider fairness trade-off")
