import streamlit as st
import pandas as pd
import numpy as np
import pickle
from model_utils import preprocess_input, create_model
import os

# Page configuration
st.set_page_config(
    page_title="Liver Disease Prediction",
    page_icon="",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Title and description
st.title("Liver Disease Prediction System")
st.markdown("""
This application uses machine learning to predict the likelihood of liver disease based on medical parameters.
Please enter the patient's medical information below.
""")

# Medical disclaimer
st.warning("""
**Medical Disclaimer**: This tool is for educational purposes only and should not be used as a substitute for professional medical advice, diagnosis, or treatment. Always consult with a qualified healthcare provider for medical decisions.
""")

# Load or create model
@st.cache_resource
def load_model():
    model_path = "liver_disease_model.pkl"
    if os.path.exists(model_path):
        try:
            with open(model_path, 'rb') as f:
                return pickle.load(f)
        except:
            pass
    # Create and save new model if not exists
    model_data = create_model()
    with open(model_path, 'wb') as f:
        pickle.dump(model_data, f)
    return model_data

# Create two columns for better layout
col1, col2 = st.columns(2)

with col1:
    st.subheader("Patient Demographics")
    
    age = st.number_input(
        "Age (years)",
        min_value=1,
        max_value=120,
        value=45,
        help="Patient's age in years"
    )
    
    gender = st.selectbox(
        "Gender",
        options=["Male", "Female"],
        help="Patient's biological gender"
    )

with col2:
    st.subheader("Liver Function Tests")
    
    total_bilirubin = st.number_input(
        "Total Bilirubin (mg/dL)",
        min_value=0.1,
        max_value=100.0,
        value=1.0,
        step=0.1,
        help="Normal range: 0.2-1.2 mg/dL"
    )
    
    direct_bilirubin = st.number_input(
        "Direct Bilirubin (mg/dL)",
        min_value=0.0,
        max_value=50.0,
        value=0.3,
        step=0.1,
        help="Normal range: 0.0-0.3 mg/dL"
    )

# Second row of inputs
col3, col4 = st.columns(2)

with col3:
    st.subheader("Enzyme Levels")
    
    alkaline_phosphotase = st.number_input(
        "Alkaline Phosphatase (IU/L)",
        min_value=10,
        max_value=3000,
        value=200,
        help="Normal range: 44-147 IU/L"
    )
    
    alamine_aminotransferase = st.number_input(
        "Alanine Aminotransferase (ALT) (IU/L)",
        min_value=1,
        max_value=3000,
        value=35,
        help="Normal range: 7-56 IU/L"
    )
    
    aspartate_aminotransferase = st.number_input(
        "Aspartate Aminotransferase (AST) (IU/L)",
        min_value=1,
        max_value=5000,
        value=40,
        help="Normal range: 10-40 IU/L"
    )

with col4:
    st.subheader("Protein Levels")
    
    total_protiens = st.number_input(
        "Total Proteins (g/dL)",
        min_value=1.0,
        max_value=15.0,
        value=6.5,
        step=0.1,
        help="Normal range: 6.0-8.3 g/dL"
    )
    
    albumin = st.number_input(
        "Albumin (g/dL)",
        min_value=0.5,
        max_value=10.0,
        value=3.1,
        step=0.1,
        help="Normal range: 3.5-5.0 g/dL"
    )
    
    albumin_and_globulin_ratio = st.number_input(
        "Albumin and Globulin Ratio",
        min_value=0.1,
        max_value=5.0,
        value=0.9,
        step=0.1,
        help="Normal range: 0.8-2.0"
    )

# Prediction button
st.markdown("---")
col_predict, col_space = st.columns([1, 3])

with col_predict:
    predict_button = st.button(
        "Predict Liver Disease",
        type="primary",
        use_container_width=True
    )

if predict_button:
    # Create input dataframe
    input_data = pd.DataFrame({
        'age': [age],
        'gender': [gender],
        'total_bilirubin': [total_bilirubin],
        'direct_bilirubin': [direct_bilirubin],
        'alkaline_phosphotase': [alkaline_phosphotase],
        'alamine_aminotransferase': [alamine_aminotransferase],
        'aspartate_aminotransferase': [aspartate_aminotransferase],
        'total_protiens': [total_protiens],
        'albumin': [albumin],
        'albumin_and_globulin_ratio': [albumin_and_globulin_ratio]
    })
    
    try:
        # Load model
        model_data = load_model()
        
        # Preprocess input
        processed_input = preprocess_input(input_data, model_data)
        
        # Make prediction
        prediction = model_data['model'].predict(processed_input)[0]
        prediction_proba = model_data['model'].predict_proba(processed_input)[0]
        
        # Display results
        st.markdown("---")
        st.subheader("Prediction Results")
        
        # Create result columns
        result_col1, result_col2 = st.columns(2)
        
        with result_col1:
            if prediction == 2:
                st.error("**Liver Disease Detected**")
                st.markdown(f"**Confidence:** {prediction_proba[1]:.1%}")
            else:
                st.success("**No Liver Disease Detected**")
                st.markdown(f"**Confidence:** {prediction_proba[0]:.1%}")
        
        with result_col2:
            st.subheader("Probability Distribution")
            prob_df = pd.DataFrame({
                'Outcome': ['No Disease', 'Disease Present'],
                'Probability': [prediction_proba[0], prediction_proba[1]]
            })
            st.bar_chart(prob_df.set_index('Outcome'))
        
        
    except Exception as e:
        st.error(f"**Error during prediction:** {str(e)}")
        st.info("Please check your input values and try again.")

