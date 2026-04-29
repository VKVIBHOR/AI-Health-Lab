from utils.style_loader import load_css
from Components.navbar import navbar
from Components.footer import footer
from utils.chatbot import get_bot_response
import streamlit as st

load_css()
navbar(active="assistant")

st.markdown("""
<div class="page-heading">
    <h1>AI Health Assistant</h1>
    <p>Structured symptom guidance and care recommendations from the platform knowledge base.</p>
</div>
""", unsafe_allow_html=True)

# --- Chatbot Section ---
st.markdown("""
<div class="clinical-card ai-insight" style="margin-bottom:24px;">
    <div class="card-subtitle" style="color:#d85c63;">Symptom Triage</div>
    <p style="margin:10px 0 0;line-height:1.6;">Describe your symptoms, for example "I have a headache and fever", to get instant advice.</p>
</div>
""", unsafe_allow_html=True)

# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Function to handle chat input
def handle_chat():
    if st.session_state.chat_input:
        prompt = st.session_state.chat_input
        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        
        # Get bot response
        response = get_bot_response(prompt)
        st.session_state.messages.append({"role": "assistant", "content": response})
        
        # Clear input
        st.session_state.chat_input = ""

# Accept user input via text_input
st.text_input(
    "Type your symptoms here...", 
    key="chat_input", 
    on_change=handle_chat, 
    placeholder="Type your symptoms here... (Press Enter to send)",
    label_visibility="collapsed"
)

st.divider()

footer()
