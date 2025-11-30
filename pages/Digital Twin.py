import streamlit as st
from Components.navbar import navbar
from Components.footer import footer
from utils.style_loader import load_css

navbar()
load_css()

# Title Card
st.markdown("""
<div class="card" style="margin-bottom: 2rem; background-color: #F3E5F5;">
    <h1 style="margin-bottom: 0.5rem;">🧬 Digital Health Twin</h1>
    <p style="font-size: 1.1rem; color: #555;">Enter your basic health stats to generate a personalized health twin.</p>
</div>
""", unsafe_allow_html=True)
st.warning("Digital Twin System under development.")
st.write("This will become the flagship feature.")
# --- Basic Profile ---
st.subheader("Basic Information")
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120)
with col2:
    gender = st.selectbox("Gender", ["Male", "Female", "Other"])
with col3:
    weight = st.number_input("Weight (kg)", min_value=1.0)

# --- Medical values ---
st.subheader("Medical Inputs")
bp = st.number_input("Blood Pressure (Systolic / Diastolic)", min_value=70)
chol = st.number_input("Cholesterol Level", min_value=50)
glucose = st.number_input("Glucose Level", min_value=50)

st.divider()

st.button("Generate Digital Twin (Coming Soon)")

st.info("Your digital twin visualization will appear here.")

footer()
