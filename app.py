import streamlit as st
import pandas as pd
import joblib
import numpy as np

# 1. Load the exported model and scaler
try:
    model = joblib.load('rf_model.pkl')
    scaler = joblib.load('scaler.pkl')
except FileNotFoundError:
    st.error("Error: 'best_model.pkl' or 'scaler.pkl' not found. Please run the save code in your notebook first.")

st.set_page_config(page_title="Breast Cancer Diagnostic Tool", layout="centered")
st.title("Breast Cancer Diagnostic Assistant")
st.write("Adjust clinical features in the sidebar to predict tumor type.")

# 2. Define all 30 features in the exact order they appear in the dataset
def get_user_input():
    st.sidebar.header("Tumor Measurements")
    
    # We group inputs into a dictionary to maintain order
    data = {
        'radius_mean': st.sidebar.slider('Radius Mean', 6.0, 30.0, 14.1),
        'texture_mean': st.sidebar.slider('Texture Mean', 9.0, 40.0, 19.2),
        'perimeter_mean': st.sidebar.slider('Perimeter Mean', 43.0, 190.0, 91.9),
        'area_mean': st.sidebar.slider('Area Mean', 143.0, 2500.0, 654.8),
        'smoothness_mean': st.sidebar.slider('Smoothness Mean', 0.05, 0.16, 0.09),
        'compactness_mean': st.sidebar.slider('Compactness Mean', 0.01, 0.35, 0.10),
        'concavity_mean': st.sidebar.slider('Concavity Mean', 0.0, 0.45, 0.08),
        'concave points_mean': st.sidebar.slider('Concave Points Mean', 0.0, 0.20, 0.04),
        'symmetry_mean': st.sidebar.slider('Symmetry Mean', 0.1, 0.3, 0.18),
        'fractal_dimension_mean': st.sidebar.slider('Fractal Dimension Mean', 0.05, 0.1, 0.06),
        'radius_se': st.sidebar.slider('Radius SE', 0.1, 3.0, 0.4),
        'texture_se': st.sidebar.slider('Texture SE', 0.3, 5.0, 1.2),
        'perimeter_se': st.sidebar.slider('Perimeter SE', 0.7, 22.0, 2.8),
        'area_se': st.sidebar.slider('Area SE', 6.0, 542.0, 40.3),
        'smoothness_se': st.sidebar.slider('Smoothness SE', 0.001, 0.03, 0.007),
        'compactness_se': st.sidebar.slider('Compactness SE', 0.002, 0.13, 0.025),
        'concavity_se': st.sidebar.slider('Concavity SE', 0.0, 0.4, 0.03),
        'concave points_se': st.sidebar.slider('Concave Points SE', 0.0, 0.05, 0.01),
        'symmetry_se': st.sidebar.slider('Symmetry SE', 0.007, 0.08, 0.02),
        'fractal_dimension_se': st.sidebar.slider('Fractal Dimension SE', 0.0008, 0.03, 0.003),
        'radius_worst': st.sidebar.slider('Radius Worst', 7.0, 36.0, 16.2),
        'texture_worst': st.sidebar.slider('Texture Worst', 12.0, 50.0, 25.6),
        'perimeter_worst': st.sidebar.slider('Perimeter Worst', 50.0, 250.0, 107.2),
        'area_worst': st.sidebar.slider('Area Worst', 185.0, 4250.0, 880.5),
        'smoothness_worst': st.sidebar.slider('Smoothness Worst', 0.07, 0.22, 0.13),
        'compactness_worst': st.sidebar.slider('Compactness Worst', 0.02, 1.0, 0.25),
        'concavity_worst': st.sidebar.slider('Concavity Worst', 0.0, 1.2, 0.27),
        'concave points_worst': st.sidebar.slider('Concave Points Worst', 0.0, 0.3, 0.11),
        'symmetry_worst': st.sidebar.slider('Symmetry Worst', 0.15, 0.6, 0.29),
        'fractal_dimension_worst': st.sidebar.slider('Fractal Dimension Worst', 0.05, 0.2, 0.08)
    }
    return pd.DataFrame(data, index=[0])

input_df = get_user_input()

# 3. Prediction Logic
if st.button('Analyze Tumor'):
    # Scale input using the saved scaler
    input_scaled = scaler.transform(input_df)
    
    # Predict using the Random Forest model
    prediction = model.predict(input_scaled)
    prediction_proba = model.predict_proba(input_scaled)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Diagnosis")
        if prediction[0] == 1:
            st.error("Malignant")
        else:
            st.success("Benign")
            
    with col2:
        st.subheader("Confidence Level")
        st.write(f"{np.max(prediction_proba) * 100:.2f}%")

st.info("Note: This tool is for educational purposes as part of the Data Science Project Life Cycle.")