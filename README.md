# 📊 AI-Powered Customer Churn Predictor & Retention Advisor

An end-to-end Machine Learning and Generative AI web application that predicts customer churn probability, uncovers key risk drivers using **SHAP (Explainable AI)**, and generates personalized customer retention strategies using **Google Gemini AI**.

## 🚀 Key Features
- **Real-Time Churn Risk Prediction:** Uses a Scikit-learn preprocessing pipeline (`StandardScaler` + `OneHotEncoder`) and a balanced `RandomForestClassifier` trained on the Telco Customer Churn dataset to calculate churn probability.
- **Interactive Risk Gauge:** Visualizes Low, Medium, and High churn risk levels dynamically using Plotly.
- **Explainable AI (SHAP):** Identifies the top customer-specific risk factors (e.g., month-to-month contract, high monthly charges, lack of tech support) driving each prediction.
- **AI Retention Strategist:** Integrates the `google-genai` SDK to generate tailored root-cause summaries, retention offers, and ready-to-use agent outreach scripts.

## 🛠️ Tech Stack
- **Language:** Python
- **Frontend & Deployment:** Streamlit, Streamlit Community Cloud
- **Machine Learning & Explainability:** Scikit-learn, SHAP, Pandas, NumPy, Joblib
- **Data Visualization:** Plotly
- **Generative AI:** Google Gemini API (`google-genai`)

## 📂 Project Structure
```text
├── data/               # Telco Customer Churn dataset
├── models/             # Serialized Scikit-learn pipeline (churn_pipeline.pkl)
├── notebooks/          # Data cleaning, EDA, and model training notebook
├── app.py              # Main Streamlit web application
└── requirements.txt    # Project dependencies
