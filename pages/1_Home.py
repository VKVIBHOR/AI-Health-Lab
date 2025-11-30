import streamlit as st
from Components.navbar import navbar
from Components.footer import footer


st.set_page_config(page_title="AI Health Lab", page_icon="🧠", layout="wide")

from utils.style_loader import load_css

navbar()

# Load CSS
load_css()

st.title("Welcome to AI Health Lab 🧠")
st.write("Choose a feature from the sidebar or explore below.")

# --- Feature Cards ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("🩺 Disease Predictors")
    st.write("Run AI-based models for major diseases.")
    st.button("Open", key="predictors")

with col2:
    st.subheader("🧬 Digital Twin")
    st.write("Generate a personalized health representation.")
    st.button("Open", key="digital_twin")

with col3:
    st.subheader("📊 Health Analytics")
    st.write("Explore your health data with smart visualizations.")
    st.button("Open", key="analytics")

footer()
