import streamlit as st
import pandas as pd
import numpy as np
from Components.navbar import navbar
from Components.footer import footer

# --- PAGE CONFIG (must be first!) ---
st.set_page_config(
    page_title="AI Health Lab",
    page_icon="🧠",
    layout="wide"
)

# --- NAVBAR ---
navbar()


from utils.style_loader import load_css

from utils.auth import login, logout

# --- AUTHENTICATION CHECK ---
if not login():
    st.stop()

# --- SIDEBAR LOGOUT ---
with st.sidebar:
    st.write(f"Logged in as: **{st.session_state.get('username', 'User')}**")
    if st.button("Logout"):
        logout()

# --- PAGE ROUTER ---
def router():
    page = st.query_params.get("page", ["home"])[0]

    if page == "home":
        # Load CSS
        load_css()

        # Main Content with Animation
        st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)
        
        # Title Card
        st.markdown("""
        <div class="card" style="margin-bottom: 2rem; text-align: center; background-color: #E3F2FD;">
            <h1 style="margin-bottom: 0.5rem;">AI Health Lab 🧠</h1>
            <p style="font-size: 1.1rem; color: #555;">Your unified platform for advanced medical diagnostics and health monitoring.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("<br>", unsafe_allow_html=True)

        # 3-Column Layout for Feature Cards
        col1, col2, col3 = st.columns(3)

        with col1:
            st.markdown("""
            <div class="card">
                <h3>🩺 Disease Predictors</h3>
                <p>Run AI-based models for major diseases like Heart Disease, Diabetes, and Cancer.</p>
                <a href="?page=predictors" target="_self">
                    <button style="background-color: #007AFF; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer; font-weight: 600;">Open Predictors</button>
                </a>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown("""
            <div class="card">
                <h3>🧬 Digital Twin</h3>
                <p>Generate a personalized health representation and simulate health scenarios.</p>
                <a href="?page=digital-twin" target="_self">
                    <button style="background-color: #007AFF; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer; font-weight: 600;">View Twin</button>
                </a>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown("""
            <div class="card">
                <h3>📊 Health Analytics</h3>
                <p>Explore your health data with smart visualizations and historical trends.</p>
                <a href="?page=analytics" target="_self">
                    <button style="background-color: #007AFF; color: white; border: none; padding: 10px 20px; border-radius: 20px; cursor: pointer; font-weight: 600;">View Analytics</button>
                </a>
            </div>
            """, unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # --- Map Section ---
        st.divider()
        st.subheader("📍 Find Care Nearby")
        st.write("Locate the nearest Clinics, Hospitals, and Diagnostic Labs.")
        
        # Real Hospital Data for Vasant Kunj, New Delhi
        hospital_data = pd.DataFrame({
            'lat': [28.5293, 28.5298, 28.5424, 28.5260],
            'lon': [77.1484, 77.1531, 77.1393, 77.1550],
            'Hospital Name': [
                "Fortis Flt. Lt. Rajan Dhall Hospital",
                "Institute of Liver and Biliary Sciences (ILBS)",
                "Indian Spinal Injuries Centre",
                "Sukhmani Hospital"
            ]
        })
        
        st.map(hospital_data)
        
        st.markdown("### 🏥 Nearby Healthcare Centers")
        st.dataframe(hospital_data[['Hospital Name']], use_container_width=True, hide_index=True)

    elif page == "predictors":
        st.title("Disease Predictors")
        st.write("Models for heart disease, diabetes, cancer, etc.")
        st.info("Please select a specific model from the sidebar or use the navigation above.")

    elif page == "digital-twin":
        st.title("Digital Health Twin")
        st.write("Personalized multi-modal health simulation.")

    elif page == "analytics":
        st.title("Health Analytics")
        st.write("Explore your health data with smart visualizations.")
        st.bar_chart({"Data": [10, 20, 30, 40]})

    elif page == "datasets":
        st.title("Datasets")
        st.write("Upload / manage datasets for training.")

    elif page == "about":
        st.title("About AI Health Lab")
        st.write("A unified AI-powered medical research platform.")


# Render router
router()

# --- FOOTER ---
footer()
