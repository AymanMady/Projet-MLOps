"""Application Flask — charge le dernier modèle du MLflow Model Registry
et fournit un formulaire de prédiction.
"""
import os
import sys

import joblib
import mlflow
import pandas as pd
from flask import Flask, render_template, request

# Permet d'importer src.utils quand on lance depuis n'importe où
PROJECT_ROOT = os.path.join(os.path.dirname(__file__), "..")
sys.path.append(PROJECT_ROOT)
from src.utils import load_config  # noqa: E402

config = load_config()
mlflow.set_tracking_uri(config["mlflow"]["MLFLOW_TRACKING_URI"])
MODEL_NAME = config["mlflow"]["REGISTERED_MODEL_NAME"]
# Modèle local de secours (utilisé si le Registry MLflow est indisponible).
LOCAL_MODEL_PATH = os.path.join(PROJECT_ROOT, config["train"]["model_path"])

# Les features du dataset Diabetes (à adapter à ton dataset)
FEATURES = [
    "Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
    "Insulin", "BMI", "DiabetesPedigreeFunction", "Age",
]

app = Flask(__name__)
_model = None


def get_model():
    """Charge (et met en cache) le modèle.

    Ordre de priorité :
      1. Alias 'champion' du MLflow Model Registry
      2. Dernière version enregistrée dans le Registry
      3. Modèle local models/model.pkl (secours si le Registry est vide/indisponible)
    """
    global _model
    if _model is None:
        for uri in (f"models:/{MODEL_NAME}@champion", f"models:/{MODEL_NAME}/latest"):
            try:
                _model = mlflow.pyfunc.load_model(uri)
                print(f"[app] Modele charge depuis le Registry : {uri}")
                return _model
            except Exception as e:
                print(f"[app] Registry indisponible pour {uri} : {e}")

        # Secours : modèle local versionné (DVC).
        if os.path.exists(LOCAL_MODEL_PATH):
            _model = joblib.load(LOCAL_MODEL_PATH)
            print(f"[app] Modele charge depuis le fichier local : {LOCAL_MODEL_PATH}")
        else:
            raise RuntimeError(
                f"Aucun modele disponible : ni dans le Registry MLflow "
                f"('{MODEL_NAME}'), ni en local ('{LOCAL_MODEL_PATH}'). "
                f"Lancez d'abord l'entrainement (src/train.py) et l'evaluation "
                f"(src/evaluate.py) pour enregistrer le modele."
            )
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
