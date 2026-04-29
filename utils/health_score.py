import numpy as np
import joblib

# Load models statically for the twin to use
try:
    diabetes_model = joblib.load("models/diabetes_model.pkl")
    diabetes_scaler = joblib.load("models/diabetes_scaler.pkl")
except:
    diabetes_model = None

try:
    heart_model = joblib.load("models/heart_disease_model.pkl")
    # Heart model doesn't use a saved scaler in the current setup based on 2_Disease Predictors.py
except:
    heart_model = None

def get_diabetes_risk(pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age):
    if not diabetes_model: return 0.0
    input_data = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
    input_scaled = diabetes_scaler.transform([input_data])
    prob = diabetes_model.predict_proba(input_scaled)[0][1]
    return float(prob)

def get_heart_risk(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal):
    if not heart_model: return 0.0
    input_data = [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]
    prob = heart_model.predict_proba([input_data])[0][1]
    return float(prob)

def calculate_health_score(diabetes_prob, heart_prob, bmi, blood_pressure):
    """
    Calculates an overall health score (0-100) based on AI risks and raw vitals.
    """
    base_score = 100
    
    # AI Risk Penalties (Max 40 points)
    ai_penalty = (diabetes_prob * 20) + (heart_prob * 20)
    
    # BMI Penalty
    if bmi < 18.5 or bmi > 24.9:
        bmi_penalty = min(abs(22 - bmi) * 1.5, 15) # Max 15 points
    else:
        bmi_penalty = 0
        
    # BP Penalty (using systolic roughly)
    if blood_pressure > 120:
        bp_penalty = min((blood_pressure - 120) * 0.5, 15)
    elif blood_pressure < 90:
        bp_penalty = min((90 - blood_pressure) * 0.5, 10)
    else:
        bp_penalty = 0

    final_score = base_score - ai_penalty - bmi_penalty - bp_penalty
    return max(0, min(100, round(final_score)))

def generate_recommendations(health_score, diabetes_prob, heart_prob, bmi, bp, chol):
    recs = []
    
    # Overall Score Message
    if health_score >= 85:
        recs.append("🌟 **Overall**: Excellent overall health! Your current lifestyle is working well for you.")
    elif health_score >= 65:
        recs.append("👍 **Overall**: Good health, but there are specific areas for improvement to lower future risks.")
    else:
        recs.append("⚠️ **Overall**: Your health score indicates elevated risks. We strongly recommend consulting a physician.")
        
    # --- Tiered AI Risks ---
    # Diabetes Risk (0 to 1 scale)
    if diabetes_prob < 0.20:
        recs.append("🛡️ **Diabetes Risk (Low)**: Your risk of diabetes is currently low. Continue maintaining a balanced diet, avoiding excessive processed sugars, and staying physically active.")
    elif 0.20 <= diabetes_prob < 0.40:
        recs.append("🟡 **Diabetes Risk (Moderate)**: Elevated risk indicated. Focus on reducing refined carbohydrates, increasing dietary fiber, and engaging in at least 150 minutes of moderate aerobic activity per week.")
    else:
        recs.append("🩸 **Diabetes Risk (High)**: High risk pattern detected. Immediate lifestyle intervention is recommended: drastically reduce added sugars, monitor fasting glucose levels, and consult an endocrinologist.")
        
    # Heart Risk (0 to 1 scale)
    if heart_prob < 0.20:
        recs.append("🛡️ **Cardiovascular Risk (Low)**: Your heart health looks solid. Keep up your cardiovascular exercises and ensure you're consuming healthy fats like Omega-3s.")
    elif 0.20 <= heart_prob < 0.40:
        recs.append("🟡 **Cardiovascular Risk (Moderate)**: Warning signs are present. Prioritize a heart-healthy diet (like the Mediterranean diet), reduce sodium intake to under 2,300mg daily, and aggressively manage stress levels.")
    else:
        recs.append("🫀 **Cardiovascular Risk (High)**: Significant heart risk detected. Please consult a cardiologist immediately. Strictly limit saturated fats, eliminate trans fats, and ensure your blood pressure is controlled through medication or lifestyle.")
        
    # --- Tiered Clinical Vitals ---
    # BMI Ranges
    if bmi < 18.5:
        recs.append("⚖️ **BMI (Underweight)**: You are currently underweight. Consider consulting a nutritionist to safely increase your caloric intake with nutrient-dense foods.")
    elif 18.5 <= bmi <= 24.9:
        recs.append("✅ **BMI (Normal)**: Your body weight is in a healthy range for your height. Maintain your current caloric balance.")
    elif 25.0 <= bmi <= 29.9:
        recs.append("📉 **BMI (Overweight)**: Your weight is slightly above the recommended range. Incremental weight loss (even 5-10% of total body weight) can rapidly decrease metabolic risks.")
    else:
        recs.append("📉 **BMI (Obese)**: Your BMI is placing significant strain on your cardiovascular and endocrine systems. Work with a professional on a structured weight management plan.")
        
    # Blood Pressure (Systolic)
    if bp < 90:
        recs.append("🩺 **Blood Pressure (Low)**: Your systolic pressure is below 90, which may cause dizziness. Ensure adequate hydration and salt intake.")
    elif 90 <= bp <= 120:
        recs.append("✅ **Blood Pressure (Normal)**: Your blood pressure is excellent. Keep sodium intake moderate and stay active.")
    elif 121 <= bp <= 139:
        recs.append("🟡 **Blood Pressure (Elevated)**: Early signs of hypertension. Adopt the DASH diet (high in fruits/vegetables/lean proteins) and reduce alcohol consumption.")
    else:
        recs.append("💓 **Blood Pressure (Hypertension)**: High blood pressure detected. This is a silent driver of heart disease. Limit sodium to <1,500mg daily and consider physician-prescribed interventions.")
        
    # Cholesterol
    if chol < 200:
        recs.append("✅ **Cholesterol (Normal)**: Your cholesterol is within the safe range.")
    elif 200 <= chol <= 239:
        recs.append("🍔 **Cholesterol (Borderline)**: Your cholesterol is creeping up. Swap bad fats (butter, fried foods) for healthy fats (olive oil, avocados, nuts).")
    else:
        recs.append("🍔 **Cholesterol (High)**: High cholesterol puts you at risk for arterial plaque buildup. Significantly restrict saturated fats, increase soluble fiber (oats, beans), and consider a statin consultation with your doctor.")

    return recs
