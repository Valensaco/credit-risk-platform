## Credit risk and fraud detection project

Built by Valentin Rivadeo — incoming FinTech MS student at UT Dallas.

I built this project to understand how machine learning models actually work in a financial context. I wanted to learn the system around them, and see how the data from loans can be used in a deployed product that it is possible ti make real time decisions, explain those decisions, and also monitors itself over time.

## What does it do ?

This project is aimed to use a platform that takes loan applications and predicts the probability of default using a combination of ML models. It can also run a fraud check, and it can explain the decision using SHAP values. It can also flag which factors drove the risk score up or down. 

Beyond single predictions, it supports batch processing, tracks performance over time, analyzes fairness across income and credit score groups, and compares two model versions side by side.

## Live Demo

https://credit-risk-platform-aryxrhltgspmdvn4zav5vy.streamlit.app/ 

## Pages

- Home -  platform overview and key metrics
- Single Prediction - assess one loan application in real time
- Batch Processing - upload a CSV and process multiple applications at once
- Analytics - model performance, risk distribution, feature importance
- Model Monitoring - drift detection and daily performance tracking
- Fairness Report - approval rates by credit tier and income group, regulatory compliance
- A/B Testing - compare the original model vs the fairness-adjusted version

## Tech Stack

- ML Models: XGBoost, LightGBM, Random Forest, Logistic Regression, 
Stacked Ensemble, Isolation Forest, One-Class SVM
-Explainability: SHAP
- API: FastAPI 
 -Frontend: Streamlit 
- Data: Lending Club loan dataset (2007–2018), 300k sample
- Other: Docker, scikit-learn, Plotly, Python


## Key Design Decisions

Why 0.71 AUC and not higher?

I did this because most notebooks on this dataset show 0.90+ AUC because they use data that only exist after the loan is already active such as payment history or recovery amounts, which could be considered data leakage. I only used information available at the moment of application for a more real outcome, which gives a 0.71 AUC +. 

Why ensemble?

XGBoost and LightGBM individually give about the same results, but averaging their prediction gives more stable probability estimates. 

Why fairness analysis?
Banks and lenders are not allowed by federal law such as ECOA to discriminate people based on race, religion,etc, when making lending decisions. I also implemented adverse action codes which are requirements under the federal law FCRA, which give specific reasons why an applicant was denied.

## What I learned

- How to build a full ML system end to end.
- Why data leakage is a real problem and how to avoid it.
- How SHAP values work and why explainability matters in regulated industries.
- How model monitoring and drift detection work in production.
- How fairness metrics are calculated and what regulatory compliance looks like.
