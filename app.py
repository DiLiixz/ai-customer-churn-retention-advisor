import os
import time
import joblib
import numpy as np
import pandas as pd
import shap
import streamlit as st
import plotly.graph_objects as go
from dotenv import load_dotenv
from google import genai

# 1. Load GEMINI_API_KEY from .env file
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# 2. Page Configuration
st.set_page_config(
    page_title="AI Customer Churn & Retention Advisor",
    page_icon="📊",
    layout="wide"
)

# 3. Load Saved ML Pipeline
@st.cache_resource
def load_model():
    return joblib.load("models/churn_pipeline.pkl")

pipeline = load_model()

# 4. Main Header
st.title("📊 AI-Powered Customer Churn Predictor & Retention Advisor")
st.markdown(
    "Predict customer churn probability using **Machine Learning**, uncover key risk drivers with **SHAP Explainable AI**, "
    "and generate actionable retention strategies powered by **Gemini AI**."
)
st.divider()

# 5. Sidebar - Customer Profile Inputs
st.sidebar.header("👤 Customer Profile")

tenure = st.sidebar.slider("Tenure (Months)", min_value=0, max_value=72, value=12)
monthly_charges = st.sidebar.slider("Monthly Charges ($)", min_value=18.0, max_value=120.0, value=75.0, step=0.5)

default_total = float(tenure * monthly_charges) if tenure > 0 else float(monthly_charges)
total_charges = st.sidebar.number_input("Total Charges ($)", min_value=0.0, max_value=10000.0, value=default_total)

contract = st.sidebar.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])
internet_service = st.sidebar.selectbox("Internet Service", ["Fiber optic", "DSL", "No"])
online_security = st.sidebar.selectbox("Online Security", ["No", "Yes", "No internet service"])
tech_support = st.sidebar.selectbox("Tech Support", ["No", "Yes", "No internet service"])
payment_method = st.sidebar.selectbox(
    "Payment Method",
    ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"]
)

# Convert inputs to DataFrame
input_data = pd.DataFrame([{
    "tenure": tenure,
    "MonthlyCharges": monthly_charges,
    "TotalCharges": total_charges,
    "Contract": contract,
    "InternetService": internet_service,
    "OnlineSecurity": online_security,
    "TechSupport": tech_support,
    "PaymentMethod": payment_method
}])

# 6. Prediction & SHAP Explanation
churn_prob = pipeline.predict_proba(input_data)[0][1]
churn_percent = round(churn_prob * 100, 1)

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

X_transformed = preprocessor.transform(input_data)
if hasattr(X_transformed, "toarray"):
    X_transformed = X_transformed.toarray()

feature_names = preprocessor.get_feature_names_out()
clean_feature_names = [name.replace("num__", "").replace("cat__", "") for name in feature_names]

explainer = shap.TreeExplainer(classifier)
shap_values = explainer.shap_values(X_transformed)

if isinstance(shap_values, list):
    sv = shap_values[1][0]
elif len(shap_values.shape) == 3:
    sv = shap_values[0, :, 1]
else:
    sv = shap_values[0]

shap_df = pd.DataFrame({
    "Feature": clean_feature_names,
    "Impact": sv
}).sort_values(by="Impact", ascending=False)

top_risk_factors = shap_df[shap_df["Impact"] > 0].head(3)

# 7. Dashboard Layout (2 Columns)
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("🎯 Churn Risk Assessment")
    
    if churn_percent >= 65:
        risk_label = "High Risk"
        bar_color = "#FF4B4B"
    elif churn_percent >= 35:
        risk_label = "Medium Risk"
        bar_color = "#FFA500"
    else:
        risk_label = "Low Risk"
        bar_color = "#00CC96"

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=churn_percent,
        number={'suffix': "%"},
        title={'text': risk_label, 'font': {'size': 22}},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': bar_color},
            'steps': [
                {'range': [0, 35], 'color': "rgba(0, 204, 150, 0.15)"},
                {'range': [35, 65], 'color': "rgba(255, 165, 0, 0.15)"},
                {'range': [65, 100], 'color': "rgba(255, 75, 75, 0.15)"}
            ]
        }
    ))
    fig.update_layout(height=300, margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

with col2:
    st.subheader("🔍 Key Risk Drivers (SHAP Explainability)")
    st.caption("Top features driving this customer's churn risk:")
    
    if not top_risk_factors.empty:
        for _, row in top_risk_factors.iterrows():
            st.warning(f"**{row['Feature']}** (Risk Impact Score: +{row['Impact']:.3f})")
    else:
        st.success("No significant churn risk factors detected for this customer.")

    with st.expander("View All Feature SHAP Impacts"):
        st.dataframe(shap_df, use_container_width=True, hide_index=True)

st.divider()

# 8. Gemini AI Retention Advisor
st.subheader("🤖 Gemini AI Retention Advisor")
st.markdown("Generate a personalized customer retention plan based on the ML prediction and SHAP risk factors above.")

if st.button("✨ Generate AI Retention Strategy", type="primary"):
    if not api_key:
        st.error("GEMINI_API_KEY not found! Please check your .env file.")
    else:
        with st.spinner("Analyzing customer profile and generating retention strategy..."):
            try:
                client = genai.Client(api_key=api_key)

                reasons_text = ", ".join(top_risk_factors["Feature"].tolist()) if not top_risk_factors.empty else "None"

                prompt = f"""
                You are a Senior Telecom Customer Retention Strategist. Analyze this customer's profile and provide an actionable retention plan.

                Customer Profile:
                - Tenure: {tenure} months
                - Monthly Charges: ${monthly_charges}
                - Total Charges: ${total_charges}
                - Contract Type: {contract}
                - Internet Service: {internet_service}
                - Online Security: {online_security}
                - Tech Support: {tech_support}
                - Payment Method: {payment_method}

                Machine Learning Prediction:
                - Churn Probability: {churn_percent}% ({risk_label})
                - Top Risk Factors identified by SHAP: {reasons_text}

                Provide a clear, structured response with:
                1. **Root Cause Summary**: Why is this specific customer likely to leave (or stay)?
                2. **Recommended Retention Offer**: Exact discount, contract upgrade, or free add-on (e.g., free Tech Support/Online Security) to offer.
                3. **Actionable Script/Message**: A short, polite 2-sentence SMS or call script for the retention agent to use.
                """

                response = None
                for attempt in range(3):
                    try:
                        response = client.models.generate_content(
                            model="gemini-3.8-flash",
                            contents=prompt
                        )
                        break
                    except Exception as retry_err:
                        if "503" in str(retry_err) and attempt < 2:
                            time.sleep(2)
                        else:
                            raise retry_err

                st.success("AI Retention Strategy Generated!")
                st.markdown(response.text)

            except Exception as e:
                st.error(f"Gemini API Error: {e}")