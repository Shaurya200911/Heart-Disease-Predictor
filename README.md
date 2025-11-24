❤️ Heart Disease Predictor – Machine Learning + Flask Web App

A complete end-to-end Machine Learning project that predicts the risk of heart disease based on medical, lifestyle, and biometric inputs.
The project includes:

A clean ML training pipeline (Colab)

A trained calibrated model

A Flask backend

A beautiful Bootstrap UI

Deployment-ready folder structure

This project demonstrates skills in ML, pipelines, preprocessing, Flask development, model deployment, and UI integration.

![Heart Disease Predictor - Hero Section.png](../../../../../../Downloads/Heart%20Disease%20Predictor%20-%20Hero%20Section.png)
![Heart Disease Predictor - Hero Section1.png](../../../../../../Downloads/Heart%20Disease%20Predictor%20-%20Hero%20Section1.png)
![Heart Disease Predictor - Form.png](../../../../../../Downloads/Heart%20Disease%20Predictor%20-%20Form.png)
![Heart Disease Predictor - Disclaimer.png](../../../../../../Downloads/Heart%20Disease%20Predictor%20-%20Disclaimer.png)

🚀 Features
✔ Machine Learning

Uses scikit-learn pipeline

Preprocessing:

Numeric: imputation + scaling

Binary: ordinal encoding (Yes/No)

Categorical: OneHotEncoder

Handles class imbalance using SMOTE

Uses calibrated classifier for better probability predictions

Saves model and metadata via joblib

✔ Backend (Flask)

Clean API with / (form) and /predict (prediction)

Automatic model + metadata loading

Converts user input → DataFrame → Model prediction

Threshold-based output (healthy / risk detected)

✔ Frontend (HTML, CSS, Bootstrap)

Modern responsive design

User-friendly form

Prediction displayed at top of page

Smooth scrolling

Flash messages for errors

📁 Project Structure
Heart Disease Predictor/
│
├── app.py                   # Flask backend (minimal, efficient)
├── heart_disease_predictor.joblib   # Saved ML model
├── metadata.json            # Threshold, features, version info
│
├── templates/
│   └── index.html           # Frontend UI
│
├── static/
│   └── heart-bg.jpg         # Optional background image
│
├── README.md                # Project documentation
└── requirements.txt         # Dependencies

🧠 Model Training (Google Colab)

Training was performed in Google Colab with:

Updated and pinned libraries

scikit-learn 1.7.2

Clean notebook design

Final model saved as:

heart_pipeline_calibrated.joblib
metadata.json


The metadata includes:

{
  "threshold": 0.42,
  "features": [...],
  "scikit_learn_version": "1.7.2",
  "numpy_version": "1.26.x"
}

🔧 Local Development Setup
1. Clone the repository
git clone <your-repo-url>
cd Heart Disease Predictor

2. Create and activate virtual environment
Windows:
python -m venv venv
venv\Scripts\activate

macOS/Linux:
python3 -m venv venv
source venv/bin/activate

3. Install dependencies
pip install -r requirements.txt

4. Run the Flask app
python app.py


Visit:

http://127.0.0.1:5000/

🌐 Usage

Fill the health-related details:

Age, gender, BP, cholesterol, BMI

Smoking, diabetes, family history

Triglycerides, CRP, fasting sugar

Stress level, sleep hours, etc.

Click Calculate Risk.

The app returns:

Risk classification (low / high)

Risk probability

Threshold used

Form preserved for editing

🛠 Tech Stack
Machine Learning

Python

NumPy

Pandas

Scikit-Learn

SMOTE (imbalanced-learn)

Joblib / cloudpickle

Backend

Flask

Werkzeug

Frontend

HTML

Bootstrap 5

JavaScript (for dynamic updates)

📊 Model Improvement Ideas

Add SHAP explainability

Try Gradient Boosting, XGBoost, or LightGBM

Add domain-specific features

Hyperparameter tuning (GridSearch / Optuna)

Build ROC/PR dashboards

Deploy on Render / Railway / HuggingFace Spaces

🧪 Testing

Send a POST request using cURL:

curl -X POST -F "Age=45" -F "Gender=Male" -F "Blood Pressure=135" \
     -F "Cholesterol Level=220" -F "Exercise Habits=Low" \
     http://127.0.0.1:5000/predict


Or use Postman.

📦 Deployment

You can deploy this app on:

Render

Railway

PythonAnywhere

HuggingFace Spaces

Heroku (via Docker)

Requirements:

gunicorn

requirements.txt

Correct path for model files

Deployment guide available on request.

🎓 About This Project

This project was created as part of a portfolio / college admissions / ML practice journey.

It demonstrates:

Understanding of ML pipelines

Deployment-ready architecture

Real-world engineering skills

Frontend + backend integration

Perfect for interviews, GitHub profile, and college applications.

⭐ Future Features (Planned)

Persistent user history

Graphical result visualization

REST API mode for mobile apps

Multi-model comparison

Auto-retraining pipeline

🙌 Contributing

Pull requests are welcome.
For major changes, please open an issue first.

📬 Contact

For guidance or improvements, reach out anytime.