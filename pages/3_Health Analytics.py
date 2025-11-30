from utils.style_loader import load_css
from Components.navbar import navbar
from Components.footer import footer
from utils.chatbot import get_bot_response
import streamlit as st

navbar()
load_css()

# Title Card
st.markdown("""
<div class="card" style="margin-bottom: 2rem; background-color: #E8F5E9;">
    <h1 style="margin-bottom: 0.5rem;">📊 Health Analytics</h1>
    <p style="font-size: 1.1rem; color: #555;"></p>
</div>
""", unsafe_allow_html=True)

# --- Chatbot Section ---
st.subheader("🤖 AI Health Assistant")
st.write("Describe your symptoms (e.g., 'I have a headache and fever') to get instant advice.")

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
