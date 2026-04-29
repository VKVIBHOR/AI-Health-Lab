import streamlit as st
import pandas as pd
import numpy as np
from Components.navbar import navbar
from Components.footer import footer

# --- PAGE CONFIG (must be first!) ---
st.set_page_config(
    page_title="AI Health Lab",
    page_icon="🔬",
    layout="wide"
)

from utils.style_loader import load_css

# --- GLOBAL STYLES ---
load_css()

# --- NAVBAR ---
navbar()


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
        st.html("""
        <div class="page-heading animate-fade-in">
            <h1>Patient Overview</h1>
            <p>Real-time health insights and AI-driven predictive modeling.</p>
        </div>
        """)

        st.html("""
        <div class="clinical-grid">
            <section class="clinical-card span-8">
                <div class="card-header">
                    <div>
                        <h2 class="card-title">Overall Health Score</h2>
                        <div class="card-subtitle">AI Confidence: High</div>
                    </div>
                    <span class="status-badge status-normal">Optimal</span>
                </div>
                <div class="metric-display">
                    <div class="metric-value">92</div>
                    <div class="metric-unit">/ 100</div>
                </div>
                <div class="metric-row">
                    <div>
                        <div class="data-label">Heart Rate</div>
                        <div class="data-value">72 bpm</div>
                    </div>
                    <div>
                        <div class="data-label">Blood Pressure</div>
                        <div class="data-value">118/76</div>
                    </div>
                    <div>
                        <div class="data-label">Oxygen</div>
                        <div class="data-value">98%</div>
                    </div>
                </div>
            </section>

            <section class="clinical-card ai-insight span-4">
                <div class="card-header" style="margin-bottom: 14px;">
                    <div style="display:flex;align-items:center;gap:8px;">
                        <span class="material-symbols-outlined">auto_awesome</span>
                        <div class="card-subtitle" style="margin:0;color:#d85c63;">AI Insight</div>
                    </div>
                </div>
                <p style="font-size:15px;line-height:1.65;margin:0 0 24px;">
                    Your current profile shows strong baseline markers. Run a new diagnostic analysis to refresh
                    disease-specific risk scores and update your digital twin.
                </p>
                <a class="clinical-button" href="/Disease_Predictors" target="_self">View Full Analysis</a>
            </section>

            <section class="clinical-card risk-card span-4">
                <div class="risk-title">
                    <span class="material-symbols-outlined">monitor_heart</span>
                    <strong>Cardio Risk</strong>
                </div>
                <div class="risk-bar"><div class="risk-fill" style="width:15%;"></div></div>
                <div class="risk-footer">
                    <span class="data-value" style="color:#334155;">15% Risk</span>
                    <span class="compact-badge">Low</span>
                </div>
            </section>

            <section class="clinical-card risk-card span-4">
                <div class="risk-title">
                    <span class="material-symbols-outlined">glucose</span>
                    <strong>Metabolic</strong>
                </div>
                <div class="risk-bar"><div class="risk-fill salmon-soft" style="width:28%;"></div></div>
                <div class="risk-footer">
                    <span class="data-value" style="color:#334155;">28% Risk</span>
                    <span class="compact-badge">Optimal</span>
                </div>
            </section>

            <section class="clinical-card risk-card span-4">
                <div class="risk-title">
                    <span class="material-symbols-outlined">pulmonology</span>
                    <strong>Respiratory</strong>
                </div>
                <div class="risk-bar"><div class="risk-fill error" style="width:65%;"></div></div>
                <div class="risk-footer">
                    <span class="data-value" style="color:#334155;">65% Risk</span>
                    <span class="compact-badge critical">Elevated</span>
                </div>
            </section>
        </div>
        """)

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

        st.html("""
        <div class="clinical-card facility-card" style="margin-top:24px;">
            <div class="facility-panel">
                <h2 class="card-title">Nearby Facilities</h2>
                <p style="color:#334155;line-height:1.6;margin:10px 0 0;">
                    Specialized care centers in New Delhi / Vasant Kunj matching patient risk profile.
                </p>
                <ul class="facility-list">
                    <li>
                        <span class="material-symbols-outlined" style="color:#d85c63;font-size:18px;">local_hospital</span>
                        <div>
                            <div class="data-value">Fortis Hospital, Vasant Kunj</div>
                            <div class="data-label">2.4 km • Pulmonology Unit</div>
                        </div>
                    </li>
                    <li>
                        <span class="material-symbols-outlined" style="color:#d85c63;font-size:18px;">local_hospital</span>
                        <div>
                            <div class="data-value">Vasant Kunj Medical Center</div>
                            <div class="data-label">3.1 km • General Clinic</div>
                        </div>
                    </li>
                </ul>
                <a class="clinical-button" style="width:100%;margin-top:88px;" href="#facility-directory">View Directory</a>
            </div>
            <div class="map-panel">
                <div class="map-pin"></div>
            </div>
        </div>
        """)

        st.html('<div id="facility-directory" class="page-heading" style="margin-top:24px;"><h2>Facility Directory</h2><p>Live map and healthcare center list from the app dataset.</p></div>')
        st.map(hospital_data, use_container_width=True)
        st.dataframe(hospital_data[['Hospital Name']], use_container_width=True, hide_index=True)

    elif page == "predictors":
        st.html("""
        <div class="page-heading">
            <h1>Disease Predictors</h1>
            <p>Select a diagnostic model from the Predictors page to run an AI-assisted analysis.</p>
        </div>
        """)
        st.page_link("pages/2_Disease Predictors.py", label="Open Disease Predictors")

    elif page == "digital-twin":
        st.html("""
        <div class="page-heading">
            <h1>Digital Health Twin</h1>
            <p>Generate a personalized health representation and risk simulation.</p>
        </div>
        """)
        st.page_link("pages/Digital Twin.py", label="Open Digital Health Twin")

    elif page == "analytics":
        st.html("""
        <div class="page-heading">
            <h1>AI Health Assistant</h1>
            <p>Explore symptom guidance and health analytics.</p>
        </div>
        """)
        st.page_link("pages/3_Health Analytics.py", label="Open AI Assistant")

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
