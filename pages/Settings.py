import streamlit as st
from Components.navbar import navbar
from Components.footer import footer
from utils.style_loader import load_css

navbar()
load_css()

# Title Card
st.markdown("""
<div class="card" style="margin-bottom: 2rem; background-color: #FFF3E0;">
    <h1 style="margin-bottom: 0.5rem;">⚙️ Settings</h1>
</div>
""", unsafe_allow_html=True)

st.subheader("Theme")
st.radio("Choose Theme", ["Light", "Dark", "System Default"])

st.subheader("Profile")
st.text_input("Your Name")
st.text_area("Bio")

st.subheader("Feedback")
st.text_area("Share your feedback")

footer()
