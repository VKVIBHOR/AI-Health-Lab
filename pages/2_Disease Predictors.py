import streamlit as st
import pandas as pd
import joblib
import os
from PIL import Image
from utils.predict_tabular import predict_diabetes
from utils.predict_image import predict_image, load_model
from utils.explainability import explain_tabular
from utils.gradcam import GradCAM, visualize_cam
from torchvision import transforms
import torch

from Components.navbar import navbar
from Components.footer import footer

from utils.style_loader import load_css
from utils.history import save_prediction, get_history

# UI Navbar
navbar()

# Load CSS
load_css()

# Wrap content in fade-in container
st.markdown('<div class="animate-fade-in">', unsafe_allow_html=True)

# Title Card
st.markdown("""
<div class="card" style="margin-bottom: 2rem; background-color: #FFEBEE;">
    <h1 style="margin-bottom: 0.5rem;">🩺 Disease Predictors</h1>
    <p style="font-size: 1.1rem; color: #555;">Select a disease model below to run a diagnostic analysis.</p>
</div>
""", unsafe_allow_html=True)

# ------------------------------
# Select Model
# ------------------------------
model_choice = st.selectbox(
    "Select the disease model:",
    ["Heart Disease", "Diabetes", "Pneumonia (X-Ray)", "Brain Tumor (MRI)", "Skin Cancer", "Coming Soon..."]
)

st.divider()

# ------------------------------
# Upload Section
# ------------------------------
st.subheader("Upload your data")

uploaded_img = None
uploaded_csv = None

if model_choice in ["Pneumonia (X-Ray)", "Brain Tumor (MRI)", "Skin Cancer"]:
    uploaded_img = st.file_uploader("Upload Image", type=["png", "jpg", "jpeg"])
else:
    uploaded_csv = st.file_uploader("Upload CSV or Enter Values", type=["csv"])


st.divider()
st.subheader("Prediction Output")

# =======================================================
#            HEART DISEASE PREDICTION LOGIC
# =======================================================
if model_choice == "Heart Disease":

    st.subheader("🔧 Heart Disease Input Values")

    # Load model + columns
    model_path = "models/heart_disease_model.pkl"
    columns_path = "models/heart_disease_columns.pkl"

    if not os.path.exists(model_path):
        st.error("❌ Heart Disease model not found in /models folder.")
        footer()
        st.stop()

    model = joblib.load(model_path)
    columns = joblib.load(columns_path)

    # Input Form
    with st.container():
        st.markdown('<div class="medical-card animate-slide-up">', unsafe_allow_html=True)
        st.markdown('<h3 class="medical-header">Patient Data</h3>', unsafe_allow_html=True)
        
        uploaded_csv = st.file_uploader("Upload CSV (Optional)", type=["csv"])
        
        if uploaded_csv:
            try:
                df = pd.read_csv(uploaded_csv)
                df = df[columns]
                if st.button("Predict from CSV"):
                    preds = model.predict(df)
                    probs = model.predict_proba(df)[:, 1]
                    output = pd.DataFrame({"Prediction": ["Disease" if p == 1 else "No Disease" for p in preds], "Probability": probs})
                    st.dataframe(output)
            except Exception as e:
                st.error(f"Error: {e}")
        else:
            st.markdown("#### Patient Vitals")
            col1, col2, col3 = st.columns(3)
            with col1:
                age = st.number_input("Age", 20, 100, 50)
                trestbps = st.number_input("Resting BP", 80, 200, 120)
                exang = st.selectbox("Exercise Angina", [1, 0])
                slope = st.selectbox("Slope (0-2)", [0, 1, 2])
            with col2:
                sex = st.selectbox("Sex (1=M, 0=F)", [1, 0])
                chol = st.number_input("Cholesterol", 100, 600, 200)
                oldpeak = st.number_input("ST Depression", 0.0, 6.0, 1.0)
                ca = st.selectbox("Major Vessels (0-3)", [0, 1, 2, 3])
            with col3:
                cp = st.selectbox("Chest Pain (0-3)", [0, 1, 2, 3])
                thalach = st.number_input("Max Heart Rate", 60, 250, 150)
                restecg = st.selectbox("Resting ECG (0-2)", [0, 1, 2])
                thal = st.selectbox("Thal (3,6,7)", [3, 6, 7])
            
            fbs = st.selectbox("Fasting BS > 120?", [1, 0])

            user_data = pd.DataFrame([[age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal]], columns=columns)

            if st.button("Predict Heart Disease"):
                pred = model.predict(user_data)[0]
                prob = model.predict_proba(user_data)[0][1]
                st.session_state['heart_pred'] = pred
                st.session_state['heart_prob'] = prob
                st.session_state['heart_data'] = user_data
                
                # Save to history
                result_str = "High Risk" if pred == 1 else "Low Risk"
                save_prediction("Heart Disease", result_str, prob)

            # Display Results if available
            if 'heart_pred' in st.session_state:
                pred = st.session_state['heart_pred']
                prob = st.session_state['heart_prob']
                
                if pred == 1:
                    st.error(f"⚠️ High Risk of Heart Disease\nProbability: {prob:.2f}")
                else:
                    st.success(f"✔️ Low Risk of Heart Disease\nProbability: {prob:.2f}")
                
                # XAI Section
                if st.button("Explain Prediction (SHAP)"):
                    with st.spinner("Generating explanation..."):
                        try:
                            fig = explain_tabular(model, st.session_state['heart_data'], columns)
                            st.pyplot(fig)
                        except Exception as e:
                            st.error(f"SHAP Error: {e}")
        
        st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
# DEFAULT OUTPUT WHEN OTHER MODELS ARE SELECTED
# =======================================================
# =======================================================
#            DIABETES PREDICTION LOGIC
# =======================================================
elif model_choice == "Diabetes":
    st.subheader("🩸 Diabetes Input Values")
    with st.container():
        st.markdown('<div class="medical-card animate-slide-up">', unsafe_allow_html=True)
        st.markdown('<h3 class="medical-header">Clinical Parameters</h3>', unsafe_allow_html=True)
        col1, col2 = st.columns(2)
        with col1:
            pregnancies = st.number_input("Pregnancies", 0, 20, 1)
            bp = st.number_input("Blood Pressure", 0, 200, 70)
            insulin = st.number_input("Insulin Level", 0, 900, 80)
            dpf = st.number_input("Diabetes Pedigree Function", 0.0, 3.0, 0.5)
        with col2:
            glucose = st.number_input("Glucose Level", 0, 300, 100)
            skin_thickness = st.number_input("Skin Thickness", 0, 100, 20)
            bmi = st.number_input("BMI", 0.0, 70.0, 25.0)
            age = st.number_input("Age", 0, 120, 30)

        if st.button("Predict Diabetes"):
            pred, prob = predict_diabetes([pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age])
            st.session_state['diabetes_pred'] = pred
            st.session_state['diabetes_prob'] = prob
            st.session_state['diabetes_data'] = [pregnancies, glucose, bp, skin_thickness, insulin, bmi, dpf, age]
            
            # Save to history
            result_str = "High Risk" if pred == 1 else "Low Risk"
            save_prediction("Diabetes", result_str, prob)

        if 'diabetes_pred' in st.session_state:
            pred = st.session_state['diabetes_pred']
            prob = st.session_state['diabetes_prob']
            
            if pred is None:
                st.error(prob)
            elif pred == 1:
                st.error(f"⚠️ High Risk of Diabetes\nProbability: {prob:.2f}")
            else:
                st.success(f"✔️ Low Risk of Diabetes\nProbability: {prob:.2f}")
            
            # XAI Section
            if st.button("Explain Prediction (SHAP)", key="shap_diabetes"):
                with st.spinner("Generating explanation..."):
                    # Need to load model here for SHAP
                    model = joblib.load("models/diabetes_model.pkl")
                    scaler = joblib.load("models/diabetes_scaler.pkl")
                    input_data = st.session_state['diabetes_data']
                    input_scaled = scaler.transform([input_data])
                    feature_names = ["Pregnancies", "Glucose", "BloodPressure", "SkinThickness", "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]
                    
                    fig = explain_tabular(model, input_scaled, feature_names)
                    st.pyplot(fig)
        st.markdown('</div>', unsafe_allow_html=True)
# =======================================================
#            PNEUMONIA (X-RAY) PREDICTION LOGIC
# =======================================================
elif model_choice == "Pneumonia (X-Ray)":
    st.subheader("🩻 Pneumonia Detection")
    
    if uploaded_img:
        st.image(uploaded_img, caption="Uploaded X-Ray", use_column_width=True)
        
        if st.button("Analyze X-Ray"):
            # Placeholder Logic
            st.warning("⚠️ Pneumonia Detected (Mock Prediction)")
    else:
        st.info("Please upload an X-Ray image.")

# =======================================================
#            BRAIN TUMOR (MRI) PREDICTION LOGIC
# =======================================================
elif model_choice == "Brain Tumor (MRI)":
    st.subheader("🧠 Brain Tumor Detection")
    with st.container():
        st.markdown('<div class="medical-card animate-slide-up">', unsafe_allow_html=True)
        st.markdown('<h3 class="medical-header">MRI Scan Upload</h3>', unsafe_allow_html=True)
        uploaded_img = st.file_uploader("Upload MRI Image", type=["png", "jpg", "jpeg"])
        
        if uploaded_img:
            st.image(uploaded_img, caption="Uploaded MRI", width=300)
            if st.button("Analyze MRI"):
                image = Image.open(uploaded_img).convert('RGB')
                pred_class, conf = predict_image(image, 'brain_tumor')
                st.session_state['brain_pred'] = pred_class
                st.session_state['brain_conf'] = conf
                st.session_state['brain_img'] = image
                
                # Save to history
                save_prediction("Brain Tumor", pred_class, conf)
            
            if 'brain_pred' in st.session_state:
                pred_class = st.session_state['brain_pred']
                conf = st.session_state['brain_conf']
                image = st.session_state['brain_img']
                
                if pred_class:
                    st.markdown(f"### Result: **{pred_class}**")
                    st.progress(conf)
                    st.write(f"Confidence: {conf:.2%}")
                    if pred_class != "notumor":
                        st.error("⚠️ Tumor Detected")
                    else:
                        st.success("✔️ No Tumor Detected")
                    
                    # XAI Section
                    if st.button("Explain Prediction (Grad-CAM)", key="gradcam_brain"):
                        with st.spinner("Generating heatmap..."):
                            model = load_model("models/brain_tumor_model.pth", 4)
                            target_layer = model.layer4[-1] # ResNet18 last conv layer
                            grad_cam = GradCAM(model, target_layer)
                            
                            # Preprocess image again for GradCAM
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                            img_tensor = transform(image).unsqueeze(0)
                            
                            mask = grad_cam(img_tensor)
                            fig = visualize_cam(mask, image)
                            st.pyplot(fig)
                else:
                    st.error(conf)
        st.markdown('</div>', unsafe_allow_html=True)

# =======================================================
#            SKIN CANCER PREDICTION LOGIC
# =======================================================
elif model_choice == "Skin Cancer":
    st.subheader("🔬 Skin Cancer Detection")
    with st.container():
        st.markdown('<div class="medical-card animate-slide-up">', unsafe_allow_html=True)
        st.markdown('<h3 class="medical-header">Dermoscopy Upload</h3>', unsafe_allow_html=True)
        uploaded_img = st.file_uploader("Upload Skin Image", type=["png", "jpg", "jpeg"])
        
        if uploaded_img:
            st.image(uploaded_img, caption="Uploaded Image", width=300)
            if st.button("Analyze Skin Lesion"):
                image = Image.open(uploaded_img).convert('RGB')
                pred_class, conf = predict_image(image, 'skin_cancer')
                st.session_state['skin_pred'] = pred_class
                st.session_state['skin_conf'] = conf
                st.session_state['skin_img'] = image
                
                # Save to history
                save_prediction("Skin Cancer", pred_class, conf)
            
            if 'skin_pred' in st.session_state:
                pred_class = st.session_state['skin_pred']
                conf = st.session_state['skin_conf']
                image = st.session_state['skin_img']
                
                if pred_class:
                    st.markdown(f"### Result: **{pred_class}**")
                    st.progress(conf)
                    st.write(f"Confidence: {conf:.2%}")
                    if "carcinoma" in pred_class or "melanoma" in pred_class:
                        st.error("⚠️ Potential Malignancy Detected")
                    else:
                        st.success("✔️ Benign / Low Risk")
                    
                    # XAI Section
                    if st.button("Explain Prediction (Grad-CAM)", key="gradcam_skin"):
                        with st.spinner("Generating heatmap..."):
                            model = load_model("models/skin_cancer_model.pth", 9)
                            target_layer = model.layer4[-1] # ResNet18 last conv layer
                            grad_cam = GradCAM(model, target_layer)
                            
                            # Preprocess image again for GradCAM
                            transform = transforms.Compose([
                                transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                            ])
                            img_tensor = transform(image).unsqueeze(0)
                            
                            mask = grad_cam(img_tensor)
                            fig = visualize_cam(mask, image)
                            st.pyplot(fig)
                else:
                    st.error(conf)
        st.markdown('</div>', unsafe_allow_html=True)

else:
    st.info("Prediction for this model will be added soon.")


st.markdown('</div>', unsafe_allow_html=True) # Close fade-in container

# --- History Section ---
st.divider()
st.subheader("🕒 Recent Predictions")
history = get_history()

if history:
    df_history = pd.DataFrame(history)
    st.dataframe(df_history, use_container_width=True)
else:
    st.info("No prediction history available yet.")

# Footer
footer()
