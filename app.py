from flask import Flask, render_template, request, flash
import joblib, json, os
import pandas as pd
from pathlib import Path
import xgboost

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key")

@app.template_global()
def current_year():
    from datetime import datetime
    return datetime.now().year

MODEL_FILE = os.environ.get("MODEL_FILE", "full_heart_disease_prediction_pipeline.joblib")
META_FILE = os.environ.get("META_FILE", "metadata.json")
APP_DIR = Path(__file__).resolve().parent

# load model safely
model = None
try:
    model = joblib.load(APP_DIR / MODEL_FILE)
except Exception as e:
    print("Error loading model:", e)
    model = None

# load metadata safely and ensure expected is a list
meta = {}
if (APP_DIR / META_FILE).exists():
    try:
        with open(APP_DIR / META_FILE, "r") as f:
            meta = json.load(f) or {}
    except Exception:
        meta = {}
else:
    meta = {}

if not isinstance(meta, dict):
    meta = {}

expected = meta.get("features") or []
if not isinstance(expected, (list, tuple)):
    expected = list(expected) if expected is not None else []
expected = list(expected)

threshold = float(meta.get("threshold", 0.15))

def build_df(form):
    data = {}
    if expected:
        for k in expected:
            raw = form.get(k, "")
            if raw is None:
                raw = ""
            v = str(raw).strip()
            data[k] = None if v == "" else v
    else:
        for k, raw in form.items():
            if raw is None:
                raw = ""
            v = str(raw).strip()
            data[k] = None if v == "" else v
    df = pd.DataFrame([data])
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df

@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", form_data={})

@app.route("/predict", methods=["POST"])
def predict():
    df = build_df(request.form)
    if model is None:
        flash("Model not loaded. Check server logs and ensure the model file is present.", "danger")
        return render_template("index.html", form_data=request.form)
    try:
        probs = model.predict_proba(df)
        proba = float(probs[:, 1][0]) if probs.shape[1] > 1 else float(probs[:, 0][0])
        pred = int(proba >= threshold)
        return render_template("index.html", prediction=pred, probability=proba, threshold=threshold, form_data=request.form)
    except Exception as e:
        flash(f"Prediction error: {e}", "danger")
        return render_template("index.html", form_data=request.form)

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
