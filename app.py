from flask import Flask, render_template, request, flash
import joblib, json, os
import pandas as pd
from pathlib import Path

app = Flask(__name__)
app.secret_key = os.environ.get("FLASK_SECRET_KEY", "dev-key")
MODEL_FILE = os.environ.get("MODEL_FILE", "heart_disease_predictor.joblib")
META_FILE = os.environ.get("META_FILE", "metadata.json")
APP_DIR = Path(__file__).resolve().parent
model, meta = None, {}
try:
    model = joblib.load(APP_DIR / MODEL_FILE)
except Exception:
    model = None
if (APP_DIR / META_FILE).exists():
    try:
        with open(APP_DIR / META_FILE) as f:
            meta = json.load(f)
    except Exception:
        meta = {}
threshold = float(meta.get("threshold", 0.5))
expected = meta.get("features", [])

def build_df(form):
    data = {}
    if expected:
        for k in expected:
            v = form.get(k, "").strip()
            data[k] = None if v=="" else v
    else:
        for k, v in form.items():
            data[k] = None if v.strip()=="" else v.strip()
    df = pd.DataFrame([data])
    for c in df.columns:
        try:
            df[c] = pd.to_numeric(df[c])
        except Exception:
            pass
    return df

@app.template_global()
def current_year():
    from datetime import datetime
    return datetime.now().year


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", form_data={})

@app.route("/predict", methods=["POST"])
def predict():
    df = build_df(request.form)
    if model is None:
        flash("Model not loaded", "danger")
        return render_template("index.html", form_data=request.form)
    try:
        proba = float(model.predict_proba(df)[:,1][0])
        pred = int(proba>=threshold)
        return render_template("index.html", prediction=pred, probability=proba, threshold=threshold, form_data=request.form)
    except Exception as e:
        flash(str(e), "danger")
        return render_template("index.html", form_data=request.form)

if __name__=="__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
