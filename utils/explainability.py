import shap
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def explain_tabular(model, input_data, feature_names):
    """
    Generates a SHAP force plot for a single prediction.
    model: Trained sklearn model (RandomForest, XGBoost, etc.)
    input_data: pandas DataFrame or numpy array (1 row)
    feature_names: list of feature names
    Returns: matplotlib figure
    """
    # Create object that can calculate shap values
    explainer = shap.TreeExplainer(model)
    
    # Calculate Shap values
    shap_values = explainer.shap_values(input_data)
    
    # Handle binary classification (shap_values is a list of arrays, we want the positive class)
    if isinstance(shap_values, list):
        shap_values = shap_values[1]
        
    # Check if shap_values is for single instance (1, n_features) or (1, n_features, n_classes)
    if len(shap_values.shape) == 3:
        # Shape is (1, n_features, n_classes) -> take positive class
        shap_values = shap_values[0, :, 1]
    elif len(shap_values.shape) == 2:
        # Shape is (1, n_features) -> just take the first row
        shap_values = shap_values[0]
    
    # Note: shap.plots.waterfall requires an Explanation object, which is complex to construct manually from TreeExplainer output in older versions.
    # We will use a simple bar plot of the top contributing features for robustness.
    
    # Create a DataFrame for easy sorting
    if isinstance(input_data, pd.DataFrame):
        input_vals = input_data.values[0]
    else:
        input_vals = input_data[0]
        
    # Check if shap_values is for single instance (1, n_features) or just (n_features,)
    if len(shap_values.shape) > 1:
        shap_values = shap_values[0]
        
    # Ensure 1D arrays
    shap_values = np.array(shap_values).flatten()
    input_vals = np.array(input_vals).flatten()
    
    # Debug prints
    print(f"DEBUG: feature_names length: {len(feature_names)}")
    print(f"DEBUG: shap_values length: {len(shap_values)}")
    print(f"DEBUG: input_vals length: {len(input_vals)}")

    feature_importance = pd.DataFrame({
        'Feature': feature_names,
        'SHAP Value': shap_values,
        'Value': input_vals
    })
    
    # Sort by absolute SHAP value
    feature_importance['Abs SHAP'] = feature_importance['SHAP Value'].abs()
    feature_importance = feature_importance.sort_values('Abs SHAP', ascending=True).tail(10) # Top 10
    
    # Plot
    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ['red' if x > 0 else 'blue' for x in feature_importance['SHAP Value']]
    ax.barh(feature_importance['Feature'], feature_importance['SHAP Value'], color=colors)
    ax.set_xlabel("SHAP Value (Impact on Model Output)")
    ax.set_title("Feature Contribution to Prediction")
    
    # Add values as text
    for i, v in enumerate(feature_importance['SHAP Value']):
        ax.text(v, i, f" {v:.2f}", va='center', fontsize=8)
        
    plt.tight_layout()
    return fig

