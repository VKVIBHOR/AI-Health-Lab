import streamlit as st

def navbar(active="dashboard"):
    active_classes = {
        "dashboard": " is-active" if active == "dashboard" else "",
        "predictors": " is-active" if active == "predictors" else "",
        "twin": " is-active" if active == "twin" else "",
        "assistant": " is-active" if active == "assistant" else "",
        "settings": " is-active" if active == "settings" else "",
    }

    st.html(f"""
    <div class="clinical-sidebar">
        <div class="clinical-brand">
            <div class="clinical-brand-icon">
                <span class="material-symbols-outlined">science</span>
            </div>
            <div>
                <span class="clinical-brand-title">Health<br>Intelligence</span>
                <span class="clinical-brand-kicker">Clinical AI Node</span>
            </div>
        </div>

        <a class="clinical-new-analysis" href="/Disease_Predictors" target="_self">
            <span class="material-symbols-outlined">add</span>
            <strong>New Analysis</strong>
        </a>

        <nav class="clinical-nav-links">
            <a class="clinical-nav-link{active_classes["dashboard"]}" href="/Home" target="_self">
                <span class="material-symbols-outlined">dashboard</span>
                Dashboard
            </a>
            <a class="clinical-nav-link{active_classes["predictors"]}" href="/Disease_Predictors" target="_self">
                <span class="material-symbols-outlined">biotech</span>
                Predictors
            </a>
            <a class="clinical-nav-link{active_classes["twin"]}" href="/Digital_Twin" target="_self">
                <span class="material-symbols-outlined">psychology</span>
                Health Twin
            </a>
            <a class="clinical-nav-link{active_classes["assistant"]}" href="/Health_Analytics" target="_self">
                <span class="material-symbols-outlined">smart_toy</span>
                AI Assistant
            </a>
            <a class="clinical-nav-link{active_classes["settings"]}" href="/Settings" target="_self">
                <span class="material-symbols-outlined">settings</span>
                Settings
            </a>
        </nav>

        <div class="clinical-nav-footer">
            <a class="clinical-nav-link" href="/Settings" target="_self">
                <span class="material-symbols-outlined">help</span>
                Support
            </a>
            <a class="clinical-nav-link" href="/" target="_self">
                <span class="material-symbols-outlined">logout</span>
                Logout
            </a>
        </div>
    </div>

    <div class="clinical-mobile-bar">
        <div class="clinical-mobile-brand">
            <span class="clinical-brand-icon" style="width:32px;height:32px;border-radius:8px;">
                <span class="material-symbols-outlined" style="font-size:18px;">science</span>
            </span>
            AI Health Lab
        </div>
        <a class="clinical-button" href="/Disease_Predictors" target="_self">New Analysis</a>
    </div>
    """)
