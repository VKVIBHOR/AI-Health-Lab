# AI Health Lab 🧠

AI Health Lab is a unified platform for advanced medical diagnostics and health monitoring, leveraging AI to predict diseases and analyze health data.

## Features
- **Disease Predictors**: AI models for Heart Disease, Diabetes, Pneumonia (X-Ray), Brain Tumor (MRI), and Skin Cancer.
- **Digital Twin**: Personalized health simulation.
- **Health Analytics**: Visualizations of health trends.
- **Hospital Finder**: Locate nearby healthcare facilities.

## Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/VKVIBHOR/AI-Health-Lab.git
cd AI-Health-Lab
```

### 2. Install Dependencies
Ensure you have Python 3.9+ installed.
```bash
pip install -r requirements.txt
```

### 3. Run the Application
```bash
streamlit run app.py
```

## Note on Datasets
The `datasets/` folder is **not included** in this repository due to size limits. 
- The application **will run fully** using the pre-trained models in the `models/` directory.
- You only need the datasets if you plan to **re-train** the models using the scripts in `train_*.py`.
