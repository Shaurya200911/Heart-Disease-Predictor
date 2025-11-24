# /mnt/data/app.py
from flask import Flask, render_template, request, flash, redirect, url_for
import joblib
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime
import os
import json

app = Flask(__name__)
# For real deployment set this from environment variable or a secure store
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "ReplaceThisSecretInProd")

# Model filename expected in project root (adjust if your file has a different name/path)
MODEL_FILENAME = "heart_disease_predictor.joblib"
META_FILENAME = "metadata.json"  # optional metadata file (threshold, features)

# Full list of form fields expected from index.html (must match form input names)
EXPECTED_FIELDS = [
    "Age", "Gender", "Blood Pressure", "Cholesterol Level", "Exercise Habits",
    "Smoking", "Family Heart Disease", "Diabetes", "BMI", "High Blood Pressure",
    "High LDL Cholesterol", "Alcohol Consumption", "Stress Level", "Sleep Hours",
    "Sugar Consumption", "Triglyceride Level", "Fasting Blood Sugar", "CRP Level",
    "Homocysteine Level"
]

# Which of the above fields are numeric (attempt float conversion)
NUMERIC_FIELDS = {
    "Age", "Blood Pressure", "Cholesterol Level", "BMI", "Sleep Hours",
    "Triglyceride Level", "Fasting Blood Sugar", "CRP Level", "Homocysteine Level"
}

# Load model & metadata at startup (with helpful error messages)
model = None
metadata = {}
app_dir = Path(__file__).resolve().parent

def load_artifacts():
    global model, metadata
    model_path = app_dir / MODEL_FILENAME
    meta_path = app_dir / META_FILENAME

    if not model_path.exists():
        app.logger.error(f"Model file not found at {model_path}")
        return

    try:
        # joblib.load is typically fine; if you used cloudpickle when saving you may need to load accordingly
        model = joblib.load(model_path)
        app.logger.info(f"Loaded model from {model_path}")
    except Exception as e:
        app.logger.exception("Failed to load model")
        model = None

    if meta_path.exists():
        try:
            with open(meta_path, "r") as f:
                metadata = json.load(f)
        except Exception:
            metadata = {}
    else:
        metadata = {}

# Call load on startup
load_artifacts()

@app.template_global()
def current_year():
    return datetime.now().year

@app.route("/", methods=["GET", "POST"])
def predict():
    form_data = {}
    prediction = None
    probability = None
    threshold = metadata.get("threshold", 0.15)  # fallback threshold

    if request.method == "POST":
        # Build a dict with expected fields (keep keys exact to what the model expects)
        for key in EXPECTED_FIELDS:
            raw_val = request.form.get(key, "").strip()
            # Normalize empty strings to NaN for numeric fields and keep blank for categorical
            if raw_val == "":
                form_data[key] = np.nan
            else:
                if key in NUMERIC_FIELDS:
                    # try to convert to float; if fails, set NaN and flash warning
                    try:
                        form_data[key] = float(raw_val)
                    except ValueError:
                        form_data[key] = np.nan
                        flash(f"Could not parse numeric field {key!s}. Treating as missing.", "warning")
                else:
                    # For categorical/binary fields, keep the string as-is
                    form_data[key] = raw_val

        # For convenience, also include any raw form values (so template can access both)
        raw_form = dict(request.form)

        # Build single-row DataFrame matching the model's expected columns
        try:
            df = pd.DataFrame([form_data], columns=EXPECTED_FIELDS)
        except Exception as e:
            flash(f"Error creating input dataframe: {e}", "danger")
            return render_template("index.html", form_data=raw_form)

        # Check model loaded
        if model is None:
            flash(f"Model not loaded. Check server logs. Expected file: {MODEL_FILENAME}", "danger")
            return render_template("index.html", form_data=raw_form)

        # Make prediction
        try:
            # Some models/pipelines require columns exactly as training; ensure column order matches EXPECTED_FIELDS
            # model.predict_proba should accept a pandas DataFrame if pipeline includes preprocessing
            proba = model.predict_proba(df)[:, 1][0]
            prediction = int(proba >= threshold)
            probability = float(proba)
            flash("Prediction completed", "success")
            # Render the template with prediction variables and the original form data
            return render_template("index.html", prediction=prediction, probability=probability,
                                   threshold=threshold, form_data=raw_form)
        except Exception as e:
            # Provide trace in logs and friendly message to user
            app.logger.exception("Prediction error")
            flash(f"Error during prediction: {e}", "danger")
            return render_template("index.html", form_data=raw_form)

    # GET request -> render empty form
    return render_template("index.html", form_data={})

if __name__ == "__main__":
    # For development only. For production use gunicorn / uwsgi.
    app.run(debug=True, host="0.0.0.0", port=5000)
