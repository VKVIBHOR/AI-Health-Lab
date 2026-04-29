import streamlit as st
from Components.navbar import navbar
from Components.footer import footer
from utils.style_loader import load_css

load_css()
navbar(active="settings")

st.markdown("""
<div class="page-heading">
    <h1>Settings</h1>
    <p>Manage account details, preferences, and feedback for the clinical workspace.</p>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="clinical-card" style="margin-bottom:24px;">
    <div class="card-title">Theme Settings</div>
    <p style="color:#334155;line-height:1.6;margin-bottom:0;">Theme is controlled natively by your System Settings or by clicking the top right Streamlit menu, then Settings, then Theme.</p>
</div>
""", unsafe_allow_html=True)

st.markdown('<div class="page-heading" style="margin-bottom:12px;"><h2>Profile</h2></div>', unsafe_allow_html=True)
st.text_input("Your Name")
st.text_area("Bio")

st.markdown('<div class="page-heading" style="margin:24px 0 12px;"><h2>Feedback</h2></div>', unsafe_allow_html=True)
st.text_area("Share your feedback")

footer()
