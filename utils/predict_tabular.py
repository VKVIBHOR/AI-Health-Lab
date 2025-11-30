import joblib
import pandas as pd
import os

def predict_diabetes(input_data):
    """
    Predicts diabetes risk.
    input_data: list or numpy array of shape (1, 8)
    Returns: prediction (0 or 1), probability
    """
    model_path = "models/diabetes_model.pkl"
    scaler_path = "models/diabetes_scaler.pkl"
    
    if not os.path.exists(model_path):
        return None, "Model not found"

    model = joblib.load(model_path)
    scaler = joblib.load(scaler_path)
    
    # Scale input
    input_scaled = scaler.transform([input_data])
    
    pred = model.predict(input_scaled)[0]
    prob = model.predict_proba(input_scaled)[0][1]
    
    return pred, prob
