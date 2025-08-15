import streamlit as st
import joblib
from pathlib import Path

# Base directory of the current file
BASE_DIR = Path(__file__).resolve().parent

# Load model and encoders safely
model = joblib.load(BASE_DIR / "student_mental_health_model.pkl")
pressure_encoder = joblib.load(BASE_DIR / "pressure_encoder.pkl")
social_encoder = joblib.load(BASE_DIR / "social_encoder.pkl")
activity_encoder = joblib.load(BASE_DIR / "activity_encoder.pkl")
diet_encoder = joblib.load(BASE_DIR / "diet_encoder.pkl")
target_encoder = joblib.load(BASE_DIR / "target_encoder.pkl")

# App title and description
st.title("Student Mental Health Prediction")
st.write(
    "This app predicts whether a student may face mental health issues "
    "based on lifestyle inputs."
)

# Input fields
academic_pressure = st.selectbox("Academic Pressure", pressure_encoder.classes_)
social_life = st.selectbox("Social Life", social_encoder.classes_)
physical_activity = st.selectbox("Physical Activity", activity_encoder.classes_)
diet = st.selectbox("Diet", diet_encoder.classes_)

# Predict button
if st.button("Predict"):
    try:
        # Transform inputs using encoders
        ap_encoded = pressure_encoder.transform([academic_pressure])[0]
        sl_encoded = social_encoder.transform([social_life])[0]
        pa_encoded = activity_encoder.transform([physical_activity])[0]
        d_encoded = diet_encoder.transform([diet])[0]
        
        # Create input array
        input_data = [[ap_encoded, sl_encoded, pa_encoded, d_encoded]]
        
        # Make prediction
        prediction = model.predict(input_data)[0]
        result = target_encoder.inverse_transform([prediction])[0]
        
        # Display result
        st.success(f"Prediction: {result}")
    except Exception as e:
        st.error(f"Error making prediction: {e}")
