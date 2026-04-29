import streamlit as st
from Components.navbar import navbar
from Components.footer import footer


st.set_page_config(page_title="AI Health Lab", page_icon="🧠", layout="wide")

from utils.style_loader import load_css

load_css()
navbar(active="dashboard")

st.html("""
<div class="page-heading">
    <h1>Patient Overview</h1>
    <p>Choose a clinical workflow from the workspace navigation.</p>
</div>

<div class="clinical-grid">
    <section class="clinical-card workflow-card span-4">
        <div>
            <h2 class="card-title">Disease Predictors</h2>
            <p>Run AI-based models for heart disease, diabetes, brain tumor MRI, skin cancer, and imaging workflows.</p>
        </div>
        <div class="workflow-card-actions">
            <a class="clinical-button primary" href="/Disease_Predictors" target="_self">Open</a>
        </div>
    </section>

    <section class="clinical-card workflow-card span-4">
        <div>
            <h2 class="card-title">Digital Twin</h2>
            <p>Generate a personalized health representation with integrated risk scoring and recommendations.</p>
        </div>
        <div class="workflow-card-actions">
            <a class="clinical-button primary" href="/Digital_Twin" target="_self">Open</a>
        </div>
    </section>

    <section class="clinical-card workflow-card span-4">
        <div>
            <h2 class="card-title">Health Analytics</h2>
            <p>Explore symptom guidance, recent prediction context, and assistant workflows for care planning.</p>
        </div>
        <div class="workflow-card-actions">
            <a class="clinical-button primary" href="/Health_Analytics" target="_self">Open</a>
        </div>
    </section>
</div>
""")

footer()
