import streamlit as st
import time

# Hardcoded credentials for demonstration
# In production, use a database and hashed passwords
USERS = {
    "admin": "admin123",
    "doctor": "doc123"
}

def check_password(username, password):
    """Returns True if credentials are valid."""
    if username in USERS and USERS[username] == password:
        return True
    return False

def login():
    """Handles the login UI and session state."""
    if 'authenticated' not in st.session_state:
        st.session_state['authenticated'] = False

    if not st.session_state['authenticated']:
        # Center the login box
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.markdown("""
            <div style="background-color: white; padding: 2rem; border-radius: 20px; box-shadow: 0 4px 20px rgba(0,0,0,0.1); text-align: center;">
                <h2 style="color: #333;">🔐 Login</h2>
                <p style="color: #666;">Please sign in to access AI Health Lab</p>
            </div>
            <br>
            """, unsafe_allow_html=True)
            
            username = st.text_input("Username")
            password = st.text_input("Password", type="password")
            
            if st.button("Sign In", use_container_width=True):
                if check_password(username, password):
                    st.session_state['authenticated'] = True
                    st.session_state['username'] = username
                    st.success("Login successful! Redirecting...")
                    time.sleep(1)
                    st.rerun()
                else:
                    st.error("Invalid username or password")
        return False
    
    return True

def logout():
    """Logs the user out."""
    st.session_state['authenticated'] = False
    st.session_state['username'] = None
    st.rerun()
