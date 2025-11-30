import random

# Knowledge Base: Symptoms -> {Keywords, Causes, Remedies}
KNOWLEDGE_BASE = {
    "headache": {
        "keywords": ["headache", "head hurts", "migraine"],
        "causes": ["Dehydration", "Stress", "Lack of sleep", "Eye strain", "Sinus infection"],
        "remedies": ["Drink water", "Rest in a dark room", "Apply a cold or warm compress", "Take over-the-counter pain relief (e.g., Ibuprofen)"]
    },
    "fever": {
        "keywords": ["fever", "high temperature", "chills", "shivering"],
        "causes": ["Viral infection (Flu, Cold)", "Bacterial infection", "Heat exhaustion"],
        "remedies": ["Stay hydrated", "Rest", "Take acetaminophen or ibuprofen", "Keep cool with a damp cloth"]
    },
    "cold": {
        "keywords": ["cold", "runny nose", "sneezing", "congestion"],
        "causes": ["Rhinovirus", "Seasonal changes", "Weak immune system"],
        "remedies": ["Drink warm fluids (tea, soup)", "Use a humidifier", "Rest", "Gargle salt water for sore throat"]
    },
    "cough": {
        "keywords": ["cough", "coughing", "dry cough"],
        "causes": ["Cold/Flu", "Allergies", "Asthma", "Dry air"],
        "remedies": ["Honey and lemon tea", "Steam inhalation", "Stay hydrated", "Cough drops"]
    },
    "stomach ache": {
        "keywords": ["stomach ache", "stomach pain", "abdominal pain", "nausea", "vomiting"],
        "causes": ["Indigestion", "Gas", "Food poisoning", "Gastritis"],
        "remedies": ["Drink ginger tea", "Avoid solid food for a few hours", "Use a heating pad", "Stay hydrated"]
    },
    "fatigue": {
        "keywords": ["fatigue", "tired", "exhausted", "low energy"],
        "causes": ["Lack of sleep", "Anemia", "Stress", "Poor diet"],
        "remedies": ["Improve sleep schedule", "Eat iron-rich foods", "Exercise regularly", "Manage stress"]
    },
    "sore throat": {
        "keywords": ["sore throat", "throat pain", "hurts to swallow"],
        "causes": ["Viral infection", "Dry air", "Allergies"],
        "remedies": ["Salt water gargle", "Honey tea", "Humidifier", "Throat lozenges"]
    },
    # --- SERIOUS DISEASES (APP PREDICTORS) ---
    "heart disease": {
        "keywords": ["chest pain", "heart attack", "palpitations", "shortness of breath", "tightness in chest", "heart disease"],
        "causes": ["High blood pressure", "High cholesterol", "Smoking", "Obesity", "Genetics"],
        "remedies": [
            "⚠️ **SERIOUS CONDITION**: Consult a Cardiologist immediately.",
            "**Required Tests**: ECG (Electrocardiogram), Echocardiogram, Stress Test, Coronary Angiography.",
            "Monitor blood pressure and cholesterol levels.",
            "Adopt a heart-healthy diet (low sodium/saturated fat) and exercise."
        ]
    },
    "diabetes": {
        "keywords": ["diabetes", "high blood sugar", "excessive thirst", "frequent urination", "blurry vision", "slow healing"],
        "causes": ["Insulin resistance", "Genetics", "Obesity", "Sedentary lifestyle"],
        "remedies": [
            "⚠️ **MEDICAL ATTENTION REQUIRED**: Consult an Endocrinologist.",
            "**Required Tests**: HbA1c, Fasting Blood Sugar (FBS), Oral Glucose Tolerance Test (OGTT).",
            "Monitor blood sugar levels regularly.",
            "Maintain a balanced diet and manage weight."
        ]
    },
    "pneumonia": {
        "keywords": ["pneumonia", "difficulty breathing", "chest pain when breathing", "phlegm", "high fever"],
        "causes": ["Bacterial infection", "Viral infection (Flu/COVID-19)", "Fungal infection"],
        "remedies": [
            "⚠️ **SERIOUS INFECTION**: Seek medical attention, especially if breathing is difficult.",
            "**Required Tests**: Chest X-Ray, CT Scan, Blood tests (CBC), Pulse Oximetry.",
            "Treatment often involves antibiotics (if bacterial) or antivirals.",
            "Rest and stay hydrated."
        ]
    },
    "brain tumor": {
        "keywords": ["brain tumor", "seizure", "severe headache", "vision loss", "loss of balance", "confusion"],
        "causes": ["Genetic mutations", "Radiation exposure", "Family history (rare)"],
        "remedies": [
            "🚨 **URGENT**: Consult a Neurologist or Neurosurgeon immediately.",
            "**Required Tests**: MRI (Magnetic Resonance Imaging) of the brain, CT Scan, Biopsy.",
            "Treatment depends on type/location: Surgery, Radiation therapy, Chemotherapy.",
            "Do not ignore persistent headaches, vision changes, or seizures."
        ]
    },
    "skin cancer": {
        "keywords": ["skin cancer", "melanoma", "changing mole", "irregular mole", "new skin growth", "lesion"],
        "causes": ["UV radiation (Sun exposure)", "Tanning beds", "Fair skin", "History of sunburns"],
        "remedies": [
            "⚠️ **MEDICAL CHECK REQUIRED**: Consult a Dermatologist.",
            "**Required Tests**: Dermoscopy, Skin Biopsy (Shave/Punch/Excision).",
            "Perform regular self-exams for changing moles (ABCDE rule).",
            "Use broad-spectrum sunscreen and protective clothing."
        ]
    }
}

FALLBACK_RESPONSES = [
    "I'm not sure about that symptom. Could you try describing it differently? (e.g., 'headache', 'fever', 'chest pain')",
    "I don't have information on that yet. Please consult a doctor for accurate advice.",
    "Could you clarify? I can help with common symptoms like cold, headache, or serious signs like chest pain."
]

def get_bot_response(user_input):
    """
    Analyzes user input for keywords and returns a structured response.
    """
    user_input = user_input.lower()
    
    # Check for greetings
    if any(word in user_input for word in ["hi", "hello", "hey"]):
        return "Hello! I'm your AI Health Assistant. Tell me your symptoms (e.g., 'chest pain', 'fever'), and I'll suggest possible causes and remedies."

    # Check for symptoms in knowledge base using keywords
    detected_conditions = []
    
    for condition, data in KNOWLEDGE_BASE.items():
        # Check if any keyword for this condition is in the user input
        if any(keyword in user_input for keyword in data["keywords"]):
            detected_conditions.append(condition)
    
    if detected_conditions:
        response = ""
        for condition in detected_conditions:
            info = KNOWLEDGE_BASE[condition]
            response += f"### 🩺 Potential Issue: {condition.title()}\n"
            response += f"**Possible Causes:** {', '.join(info['causes'])}\n\n"
            response += "**🏠 Advice / Remedies:**\n"
            for remedy in info['remedies']:
                response += f"- {remedy}\n"
            response += "\n---\n"
        
        response += "\n*Disclaimer: I am an AI assistant, not a doctor. Please consult a professional for serious conditions.*"
        return response

    # Fallback
    return random.choice(FALLBACK_RESPONSES)
