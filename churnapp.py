# app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib
import boto3
from io import BytesIO
import shap
import matplotlib.pyplot as plt

# ----------------------------
# DATA PREPROCESSING FUNCTIONS
# ----------------------------
def clean_data(df):
    df = df.copy()
    df['International plan'] = df['International plan'].map({'Yes':1,'No':0})
    df['Voice mail plan'] = df['Voice mail plan'].map({'Yes':1,'No':0})
    return df

def engineer_features(df):
    df = df.copy()
    df['total_minutes'] = df['Total day minutes'] + df['Total eve minutes'] + df['Total night minutes'] + df['Total intl minutes']
    df['total_calls'] = df['Total day calls'] + df['Total eve calls'] + df['Total night calls'] + df['Total intl calls']
    df['cost_per_minute'] = (df['Total day charge'] + df['Total eve charge'] + df['Total night charge'] + df['Total intl charge']) / df['total_minutes']
    df['day_night_ratio'] = df['Total day minutes'] / (df['Total night minutes'] + 1)
    df['intl_ratio'] = df['Total intl minutes'] / (df['total_minutes'] + 1)
    df['high_service_calls'] = (df['Customer service calls'] > 3).astype(int)
    return df

def preprocess(df, feature_names, scaler):
    df = clean_data(df)
    df = engineer_features(df)
    df = df[feature_names]  # enforce correct feature order
    scaled = scaler.transform(df)
    return pd.DataFrame(scaled, columns=feature_names)

# ----------------------------
# LOAD MODELS AND FEATURE NAMES FROM R2
# ----------------------------
@st.cache_resource
def load_assets_from_r2():
    try:
        R2_ENDPOINT = st.secrets["R2_ENDPOINT_URL"]
        R2_ACCESS_KEY = st.secrets["R2_ACCESS_KEY_ID"]
        R2_SECRET_KEY = st.secrets["R2_SECRET_ACCESS_KEY"]
        R2_BUCKET = st.secrets["R2_BUCKET_NAME"]

        s3 = boto3.client(
            "s3",
            endpoint_url=R2_ENDPOINT,
            aws_access_key_id=R2_ACCESS_KEY,
            aws_secret_access_key=R2_SECRET_KEY
        )

        # Load model
        model_obj = s3.get_object(Bucket=R2_BUCKET, Key="models/best_model.pkl")
        model = joblib.load(BytesIO(model_obj["Body"].read()))

        # Load scaler
        scaler_obj = s3.get_object(Bucket=R2_BUCKET, Key="models/scaler.pkl")
        scaler = joblib.load(BytesIO(scaler_obj["Body"].read()))

        # Load feature names
        features_obj = s3.get_object(Bucket=R2_BUCKET, Key="models/feature_names.pkl")
        feature_names = joblib.load(BytesIO(features_obj["Body"].read()))

        # Load metrics (optional)
        metrics_obj = s3.get_object(Bucket=R2_BUCKET, Key="models/metrics.pkl")
        metrics = joblib.load(BytesIO(metrics_obj["Body"].read()))

        st.success("Model, scaler, and feature names loaded from R2")
        return model, scaler, feature_names, metrics, s3, R2_BUCKET

    except Exception as e:
        st.warning(f"Could not load model assets from R2: {e}")
        st.stop()

model, scaler, feature_names, metrics, s3, R2_BUCKET = load_assets_from_r2()

# ----------------------------
# LOAD DEFAULT DATA FROM R2
# ----------------------------
@st.cache_data
def load_default_data_from_r2():
    try:
        # Get first CSV from data folder
        objects = s3.list_objects_v2(Bucket=R2_BUCKET, Prefix="data/")
        file_name = next((obj['Key'] for obj in objects['Contents'] if obj['Key'].endswith('.csv')), None)
        if file_name is None:
            raise FileNotFoundError("No CSV file found in data folder.")

        obj = s3.get_object(Bucket=R2_BUCKET, Key=file_name)
        data_df = pd.read_csv(BytesIO(obj['Body'].read()))

        st.success(f"Dataset loaded from Cloudflare R2: {file_name}")
        return data_df

    except Exception as e:
        st.warning(f"Could not load dataset from Cloudflare R2: {e}")
        st.stop()

# ----------------------------
# STREAMLIT UI
# ----------------------------
st.title("Customer Churn Batch Prediction App")
st.write("Upload a CSV file or use the default test dataset from Cloudflare R2.")

uploaded_file = st.file_uploader("Upload CSV", type=["csv"])

if uploaded_file:
    raw_df = pd.read_csv(uploaded_file)
    st.success("Using uploaded dataset")
else:
    raw_df = load_default_data_from_r2()
    st.info("Using default dataset from R2")

st.subheader("Preview Data")
st.dataframe(raw_df.head())

# ----------------------------
# PREDICTION
# ----------------------------
if st.button("Run Prediction"):
    processed_df = preprocess(raw_df, feature_names, scaler)
    
    # Predictions
    preds = model.predict(processed_df)
    probs = model.predict_proba(processed_df)[:,1]
    
    results = raw_df.copy()
    results["churn_prediction"] = preds
    results["churn_probability"] = probs
    
    st.subheader("Prediction Results")
    st.dataframe(results.head())
    
    # ----------------------------
    # SHAP EXPLAINABILITY
    # ----------------------------
    st.subheader("Model Explainability (SHAP)")

    explainer = shap.Explainer(model, processed_df)
    shap_values = explainer(processed_df)

    fig = plt.figure()
    shap.plots.beeswarm(shap_values, show=False)
    st.pyplot(fig)
    
    # ----------------------------
    # BUSINESS INTERPRETATION
    # ----------------------------
    st.subheader("Top 3 Drivers of Churn")
    
    shap_importance = np.abs(shap_values.values).mean(axis=0)
    importance_df = pd.DataFrame({
        "feature": feature_names,
        "importance": shap_importance
    }).sort_values(by="importance", ascending=False)
    
    top3 = importance_df.head(3)
    
    for _, row in top3.iterrows():
        feature = row["feature"]
        if "service" in feature.lower():
            st.warning(f" High customer service interactions ({feature}) are a major driver of churn — indicates dissatisfaction.")
        elif "minutes" in feature.lower():
            st.info(f" Usage pattern ({feature}) strongly affects churn - pricing or plan mismatch likely.")
        elif "intl" in feature.lower():
            st.info(f"International usage behavior ({feature}) influences churn - consider tailored plans.")
        elif "cost" in feature.lower():
            st.warning(f"Cost efficiency ({feature}) is a churn driver - customers may feel overcharged.")
        else:
            st.write(f"{feature} is an important driver of churn.")
    
    # ----------------------------
    # DOWNLOAD RESULTS
    # ----------------------------
    st.download_button(
        label="Download Predictions",
        data=results.to_csv(index=False),
        file_name="churn_predictions.csv",
        mime="text/csv"
    )