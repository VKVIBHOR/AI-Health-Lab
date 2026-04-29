import streamlit as st
import pandas as pd
try:
    import plotly.graph_objects as go
except ModuleNotFoundError:
    go = None
from Components.navbar import navbar
from Components.footer import footer
from utils.style_loader import load_css
from utils.health_score import get_diabetes_risk, get_heart_risk, calculate_health_score, generate_recommendations

load_css()
navbar(active="twin")

# Fade-in wrapper
st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)

st.markdown("""
<div class="page-heading">
    <h1>Digital Health Twin</h1>
    <p>An intelligent, multi-modal simulation of your current health state relying on AI predictions.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="clinical-card ai-insight" style="margin-bottom:24px;">
    <div class="card-subtitle" style="color:#d85c63;">AI Simulation Input</div>
    <p style="margin:10px 0 0;line-height:1.6;">Fill out the clinical parameters below to generate your personalized health score and risk analysis.</p>
</div>
""", unsafe_allow_html=True)

st.divider()

# --- Input Form ---
st.markdown('<div class="page-heading" style="margin-bottom:12px;"><h2>Clinical Parameters</h2><p>Patient demographics, vitals, and cardiac indicators.</p></div>', unsafe_allow_html=True)
with st.container():
    st.markdown('<div class="medical-card animate-slide-up">', unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        age = st.number_input("Age", 18, 120, 30)
        gender = st.selectbox("Biological Sex", ["Male", "Female"])
        sex_encoded = 1 if gender == "Male" else 0
        weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)
        height = st.number_input("Height (cm)", 100.0, 250.0, 175.0)
        bmi = weight / ((height/100)**2)

    with col2:
        bp_systolic = st.number_input("Resting Blood Pressure (Systolic)", 80, 200, 120)
        chol = st.number_input("Cholesterol Level (mg/dl)", 100, 600, 180)
        glucose = st.number_input("Glucose Level", 50, 300, 90)
        insulin = st.number_input("Insulin Level", 0, 900, 80)

    with col3:
        pregnancies = st.number_input("Pregnancies (if female)", 0, 20, 0)
        skin_thickness = st.number_input("Skin Thickness (mm)", 0, 100, 20)
        dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)

    st.markdown('<div class="label-caps" style="margin:20px 0 16px;">Advanced Cardiac Indicators - Optional defaults applied</div>', unsafe_allow_html=True)
    col4, col5, col6, col7 = st.columns(4)
    with col4:
        cp = st.selectbox("Chest Pain Type (0-3)", [0, 1, 2, 3])
        thalach = st.number_input("Max Heart Rate", 60, 250, 150)
    with col5:
        fbs = st.selectbox("Fasting Blood Sugar > 120 mg/dl?", [1, 0])
        restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
    with col6:
        exang = st.selectbox("Exercise Induced Angina?", [1, 0])
        oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
    with col7:
        slope = st.selectbox("Slope of Peak Exercise (0-2)", [0, 1, 2])
        ca = st.selectbox("Major Vessels Colored by Flourosopy (0-3)", [0, 1, 2, 3])
        thal = st.selectbox("Thal (3=normal, 6=fixed, 7=reversable)", [3, 6, 7])

    st.markdown('</div>', unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

if st.button("🧬 Generate Digital Twin & Health Score", type="primary", use_container_width=True):
    with st.spinner("Running integrated AI models..."):
        # Run Models
        diabetes_prob = get_diabetes_risk(pregnancies, glucose, bp_systolic, skin_thickness, insulin, bmi, dpf, age)
        heart_prob = get_heart_risk(age, sex_encoded, cp, bp_systolic, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)

        # Calculate Logic
        health_score = calculate_health_score(diabetes_prob, heart_prob, bmi, bp_systolic)
        recommendations = generate_recommendations(health_score, diabetes_prob, heart_prob, bmi, bp_systolic, chol)

        st.divider()
        st.markdown('<div class="page-heading"><h2>Your Digital Health Twin Analysis</h2><p>Integrated model output and personalized action planning.</p></div>', unsafe_allow_html=True)

        col_score, col_radar = st.columns(2)

        # 1. Gauge Chart for Overall Health Score
        with col_score:
            st.markdown('<div class="label-caps">Overall Health Score</div>', unsafe_allow_html=True)
            if go is not None:
                fig_gauge = go.Figure(go.Indicator(
                    mode = "gauge+number",
                    value = health_score,
                    domain = {'x': [0, 1], 'y': [0, 1]},
                    gauge = {
                        'axis': {'range': [None, 100]},
                        'bar': {'color': "#d85c63"},
                        'steps': [
                            {'range': [0, 50], 'color': "#ffdad6"},
                            {'range': [50, 75], 'color': "#fff3d6"},
                            {'range': [75, 100], 'color': "#e4f3ee"}
                        ],
                    }
                ))
                fig_gauge.update_layout(height=350, margin=dict(l=20, r=20, t=30, b=20))
                st.plotly_chart(fig_gauge, use_container_width=True)
            else:
                st.markdown(f"""
                <div class="clinical-card">
                    <div class="metric-display" style="margin:0 0 18px;">
                        <div class="metric-value">{health_score}</div>
                        <div class="metric-unit">/ 100</div>
                    </div>
                    <div class="risk-bar"><div class="risk-fill" style="width:{health_score}%;"></div></div>
                </div>
                """, unsafe_allow_html=True)

            # Text context under gauge
            if health_score >= 80:
                st.success("Your health score is excellent.")
            elif health_score >= 60:
                st.warning("Your health score is moderate. See recommendations below.")
            else:
                st.error("Your health score is critically low. Please consult a professional.")

        # 2. Radar Chart for Risk Distribution
        with col_radar:
            st.markdown('<div class="label-caps">Risk Distribution Profile</div>', unsafe_allow_html=True)

            # Normalize BMI and BP to 0-1 scale visually for the radar chart
            bmi_risk = min(abs(22 - bmi) / 15, 1.0)
            bp_risk = min(max((bp_systolic - 120) / 60, 0), 1.0)

            categories = ['Diabetes Risk', 'Heart Disease Risk', 'BMI Risk Factor', 'Blood Pressure Risk']

            risk_values = [diabetes_prob, heart_prob, bmi_risk, bp_risk]
            if go is not None:
                fig_radar = go.Figure()
                fig_radar.add_trace(go.Scatterpolar(
                    r=risk_values,
                    theta=categories,
                    fill='toself',
                    name='Patient Risk Profile'
                ))
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 1])
                    ),
                    showlegend=False,
                    height=350,
                    margin=dict(l=40, r=40, t=30, b=20)
                )
                st.plotly_chart(fig_radar, use_container_width=True)
            else:
                fallback_df = pd.DataFrame({"Risk": risk_values}, index=categories)
                st.bar_chart(fallback_df)

        # 3. Recommendations block
        st.markdown('<div class="page-heading" style="margin-top:24px;margin-bottom:12px;"><h2>Personalized Action Plan</h2><p>AI-generated recommendations based on risk and vitals.</p></div>', unsafe_allow_html=True)
        with st.container():
            st.markdown('<div class="medical-card">', unsafe_allow_html=True)
            for rec in recommendations:
                st.write(rec)
            st.markdown('</div>', unsafe_allow_html=True)

st.markdown('</div>', unsafe_allow_html=True) # Close fade

footer()
