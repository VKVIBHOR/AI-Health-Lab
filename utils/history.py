import json
import os
from datetime import datetime

HISTORY_FILE = "history.json"

def save_prediction(model_name, result, probability=None):
    """Saves a prediction to the history file. Keeps only last 5."""
    
    # Create entry
    entry = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "result": result,
        "probability": f"{probability:.2f}" if probability is not None else "N/A"
    }
    
    # Load existing history
    history = []
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
        except:
            history = []
    
    # Append new entry
    history.append(entry)
    
    # Keep only last 5
    if len(history) > 5:
        history = history[-5:]
        
    # Save back to file
    with open(HISTORY_FILE, "w") as f:
        json.dump(history, f, indent=4)

def get_history():
    """Returns the list of recent predictions (reversed order)."""
    if os.path.exists(HISTORY_FILE):
        try:
            with open(HISTORY_FILE, "r") as f:
                history = json.load(f)
            return history[::-1] # Newest first
        except:
            return []
    return []
