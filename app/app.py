"""Application Flask — charge le dernier modèle du MLflow Model Registry
et fournit un formulaire de prédiction.
"""
import os
import sys

import mlflow
import pandas as pd
from flask import Flask, render_template, request

# Permet d'importer src.utils quand on lance depuis n'importe où
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from src.utils import load_config  # noqa: E402

config = load_config()
mlflow.set_tracking_uri(config["mlflow"]["MLFLOW_TRACKING_URI"])
MODEL_NAME = config["mlflow"]["REGISTERED_MODEL_NAME"]

# Les features du dataset Diabetes (à adapter à ton dataset)
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

app = Flask(__name__)
_model = None


def get_model():
    """Charge (et met en cache) le dernier modèle 'champion' du Registry."""
    global _model
    if _model is None:
        # Essaie l'alias 'champion', sinon la dernière version.
        try:
            uri = f"models:/{MODEL_NAME}@champion"
            _model = mlflow.pyfunc.load_model(uri)
        except Exception:
            uri = f"models:/{MODEL_NAME}/latest"
            _model = mlflow.pyfunc.load_model(uri)
        print(f"[app] Modele charge depuis : {uri}")
    return _model


@app.route("/", methods=["GET", "POST"])
def index():
    prediction = None
    error = None
    if request.method == "POST":
        try:
            values = {f: float(request.form[f]) for f in FEATURES}
            X = pd.DataFrame([values])
            model = get_model()
            pred = model.predict(X)[0]
            prediction = "Diabétique (1)" if int(pred) == 1 else "Non diabétique (0)"
        except Exception as e:
            error = f"Erreur : {e}"
    return render_template("index.html", features=FEATURES,
                           prediction=prediction, error=error)


@app.route("/health")
def health():
    return {"status": "ok", "model": MODEL_NAME}


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5001, debug=False)
