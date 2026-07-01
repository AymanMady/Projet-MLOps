"""Étape 2 — Entraînement du modèle.

Lit les données prétraitées depuis S3, entraîne un RandomForest,
logue hyperparamètres + métriques + modèle dans MLflow, et sauvegarde
le modèle localement (models/model.pkl) pour DVC.
"""
import os
import joblib
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score

from src.utils import load_config, read_csv_anywhere, setup_mlflow


def main():
    config = load_config()
    p_train = config["train"]
    p_pre = config["preprocess"]
    target = p_train["target_column"]

    mlflow = setup_mlflow(config)

    df = read_csv_anywhere(p_train["data"])
    X = df.drop(columns=[target])
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=p_pre["test_size"], random_state=p_pre["random_state"]
    )

    with mlflow.start_run(run_name="train") as run:
        # --- Hyperparamètres ---
        hyperparams = {
            "n_estimators": p_train["n_estimators"],
            "max_depth": p_train["max_depth"],
            "min_samples_split": p_train["min_samples_split"],
            "random_state": p_pre["random_state"],
        }
        mlflow.log_params(hyperparams)

        # --- Entraînement ---
        model = RandomForestClassifier(**hyperparams)
        model.fit(X_train, y_train)

        # --- Métriques sur le jeu de test ---
        preds = model.predict(X_test)
        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds, average="weighted")
        mlflow.log_metric("accuracy", acc)
        mlflow.log_metric("f1_score", f1)
        print(f"[train] accuracy={acc:.4f}  f1={f1:.4f}")

        # --- Logger le modèle dans MLflow (artefact sur S3) ---
        mlflow.sklearn.log_model(model, artifact_path="model")

        # --- Sauvegarde locale pour DVC ---
        os.makedirs(os.path.dirname(p_train["model_path"]), exist_ok=True)
        joblib.dump(model, p_train["model_path"])
        print(f"[train] Modele sauvegarde : {p_train['model_path']}  (run_id={run.info.run_id})")


if __name__ == "__main__":
    main()
