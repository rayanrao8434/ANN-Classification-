import streamlit as st
import numpy as np
import joblib
from tensorflow.keras.models import load_model
import time

st.set_page_config(
    page_title="Predictive Diabetes Intelligence",
    page_icon="🧬",
    layout="wide",
)

st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Plus+Jakarta+Sans:wght@700;800&display=swap');

    /* Global Fonts */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    /* Headings Font */
    h1, h2, h3, .header-title {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    
    /* Hide the top header added by streamlit */
    header {visibility: hidden;}
    
    .header-box {
        background-color: #111827;
        padding: 1rem 2rem;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        margin-top: -3rem;
        margin-bottom: 1.5rem;
        text-align: center;
        border-top: 4px solid #3B82F6;
    }
    .header-title {
        margin: 0;
        color: #F8FAFC;
        font-size: 2.2rem;
        font-weight: 700;
        letter-spacing: -0.5px;
    }
    .header-box p {
        margin: 0.5rem 0 0 0;
        color: #94A3B8;
        font-size: 1.1rem;
    }
    
    /* Button styling - Reddish theme */
    .stButton>button {
        background: linear-gradient(135deg, #ef233c, #d90429);
        color: #F8FAFC !important;
        border-radius: 8px; /* Classic SaaS shape */
        height: 60px;
        width: 100%;
        font-family: 'Inter', sans-serif;
        border: none;
        box-shadow: 0 4px 15px rgba(239, 35, 60, 0.3);
        transition: all 0.3s ease;
        display: flex;
        justify-content: center;
        align-items: center;
    }
    .stButton>button p {
        font-size: 1.4rem !important; /* Made the text bigger */
        font-weight: 700 !important;
        margin: 0;
        color: white !important;
    }
    .stButton>button:hover {
        background: linear-gradient(135deg, #d90429, #bc0321);
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(239, 35, 60, 0.4);
    }
    
    .result-container {
        display: flex;
        flex-direction: column;
        justify-content: center;
        align-items: center;
        background-color: #111827;
        border-radius: 12px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
        padding: 2rem;
        text-align: center;
        min-height: 420px; /* Adjusted height to align better with left side */
        margin-top: 0;
    }
    .result-high { color: #F59E0B; font-weight: 700; font-size: 1.8rem; margin-bottom: 0;} /* Amber Warning */
    .result-low { color: #10B981; font-weight: 700; font-size: 1.8rem; margin-bottom: 0;} /* Emerald Success */
    
    .prob-circle {
        display: inline-flex;
        justify-content: center;
        align-items: center;
        width: 130px;
        height: 130px;
        border-radius: 50%;
        font-size: 2.2rem;
        font-weight: 800;
        margin: 1.5rem 0;
        color: #F8FAFC;
        font-family: 'Inter', sans-serif;
    }
    .circle-high { background: linear-gradient(135deg, #F59E0B, #D97706); box-shadow: 0 8px 20px rgba(245,158,11,0.2); }
    .circle-low { background: linear-gradient(135deg, #10B981, #059669); box-shadow: 0 8px 20px rgba(16,185,129,0.2); }
    
    .empty-state {
        color: #F8FAFC;
        font-size: 1.2rem;
        font-weight: 500;
    }
    .empty-icon {
        font-size: 4rem;
        color: #334155;
        margin-bottom: 1rem;
    }
    
    /* Subheaders inside cards */
    h4 {
        color: #F8FAFC !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
    }
    p {
        color: #94A3B8;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_assets():
    model = load_model("diabetes_model.h5")
    scaler = joblib.load("scaler.joblib")
    return model, scaler

try:
    model, scaler = load_assets()
except Exception as e:
    st.error(f"Error loading model or scaler: {e}")
    st.stop()

st.markdown("""
<div class="header-box">
    <h1 class="header-title">🧬 Predictive Diabetes Intelligence</h1>
    <p>Real-time neural network inference using patient vitals</p>
</div>
""", unsafe_allow_html=True)

# Layout: Inputs & Button (Left) | Result (Right)
c_in, c_res = st.columns([1, 1], gap="large")

with c_in:
    st.markdown("<h4 style='margin-top:0; margin-bottom:1rem;'>Patient Vitals</h4>", unsafe_allow_html=True)
    
    # Divide features into 2 columns for compactness
    in1, in2 = st.columns(2)
    with in1:
        age = st.number_input("Age (Years)", 1, 120, 30, 1)
        bmi = st.number_input("BMI", 10.0, 60.0, 25.0, 0.1)
        pregnancies = st.number_input("Pregnancies", 0, 20, 0, 1)
        pedigree = st.number_input("Pedigree Function", 0.0, 3.0, 0.5, 0.01)
    with in2:
        glucose = st.number_input("Glucose", 0.0, 300.0, 120.0, 1.0)
        blood_pressure = st.number_input("Blood Pressure", 0.0, 150.0, 70.0, 1.0)
        skin_thickness = st.number_input("Skin Thickness", 0.0, 100.0, 20.0, 1.0)
        insulin = st.number_input("Insulin", 0.0, 900.0, 80.0, 1.0)
        
    st.markdown("<br>", unsafe_allow_html=True) # A bit of spacing
    predict_btn = st.button("Predict ➔", use_container_width=True)

with c_res:
    # Invisible header to horizontally align the result container with the inputs on the left
    st.markdown("<h4 style='margin-top:0; margin-bottom:1rem; visibility:hidden;'>Result</h4>", unsafe_allow_html=True)
    
    if predict_btn:
        features = np.array([[pregnancies, glucose, blood_pressure, skin_thickness, insulin, bmi, pedigree, age]])
        features_scaled = scaler.transform(features)
        prediction_prob = model.predict(features_scaled)[0][0]
        prediction = (prediction_prob >= 0.5).astype(int)
        
        if prediction == 1:
            st.markdown(f"""
            <div class="result-container">
                <p class="result-high">High Risk Detected</p>
                <div class="prob-circle circle-high">{prediction_prob:.1%}</div>
                <p style="color: #F8FAFC; font-size: 1.1rem; margin-bottom: 0;">The model indicates a high probability of diabetes.</p>
                <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0.5rem;">Consult a medical professional for clinical assessment.</p>
            </div>
            """, unsafe_allow_html=True)
        else:
            st.markdown(f"""
            <div class="result-container">
                <p class="result-low">Low Risk Detected</p>
                <div class="prob-circle circle-low">{(1-prediction_prob):.1%}</div>
                <p style="color: #F8FAFC; font-size: 1.1rem; margin-bottom: 0;">The model indicates a low probability of diabetes.</p>
                <p style="color: #94A3B8; font-size: 0.95rem; margin-top: 0.5rem;">Maintain healthy habits to preserve this status.</p>
            </div>
            """, unsafe_allow_html=True)
    else:
        st.markdown("""
        <div class="result-container">
            <div class="empty-icon">📊</div>
            <p class="empty-state">Awaiting Input</p>
            <p style="color: #94A3B8; font-size: 0.95rem;">Enter patient vitals on the left and click predict.</p>
        </div>
        """, unsafe_allow_html=True)
