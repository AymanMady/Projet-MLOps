"""Étape 3 — Évaluation du modèle + enregistrement du MEILLEUR modèle.

Charge le modèle entraîné, l'évalue, logue les métriques dans MLflow,
et n'enregistre le modèle dans le Model Registry QUE s'il dépasse
la meilleure accuracy déjà enregistrée (ou le seuil minimal).
"""
import joblib
import mlflow
from mlflow.tracking import MlflowClient
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

from src.utils import load_config, read_csv_anywhere, setup_mlflow


def get_best_registered_accuracy(client: MlflowClient, model_name: str) -> float:
    """Retourne la meilleure accuracy parmi les versions déjà enregistrées (0 si aucune)."""
    best = 0.0
    try:
        for mv in client.search_model_versions(f"name='{model_name}'"):
            run = client.get_run(mv.run_id)
            acc = run.data.metrics.get("accuracy", 0.0)
            best = max(best, acc)
    except Exception:
        pass
    return best


def main():
    config = load_config()
    p_eval = config["evaluate"]
    p_pre = config["preprocess"]
    target = p_eval["target_column"]
    model_name = config["mlflow"]["REGISTERED_MODEL_NAME"]

    mlflow = setup_mlflow(config)
    client = MlflowClient()

    df = read_csv_anywhere(p_eval["data_path"])
    X = df.drop(columns=[target])
    y = df[target]
    _, X_test, _, y_test = train_test_split(
        X, y, test_size=p_pre["test_size"], random_state=p_pre["random_state"]
    )

    model = joblib.load(p_eval["model_path"])

    with mlflow.start_run(run_name="evaluate") as run:
        preds = model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, preds),
            "f1_score": f1_score(y_test, preds, average="weighted"),
            "precision": precision_score(y_test, preds, average="weighted", zero_division=0),
            "recall": recall_score(y_test, preds, average="weighted", zero_division=0),
        }
        mlflow.log_metrics(metrics)
        print("[evaluate] Metriques :", {k: round(v, 4) for k, v in metrics.items()})

        # Logger le modèle de ce run (nécessaire pour l'enregistrer ensuite)
        mlflow.sklearn.log_model(model, artifact_path="model")

        acc = metrics["accuracy"]
        best_prev = get_best_registered_accuracy(client, model_name)

        # --- Enregistrer UNIQUEMENT le meilleur modèle (plus de seuil minimal) ---
        if acc >= best_prev:
            model_uri = f"runs:/{run.info.run_id}/model"
            result = mlflow.register_model(model_uri, model_name)
            print(f"[evaluate] ✅ Nouveau meilleur modele enregistre : "
                  f"{model_name} v{result.version} (acc={acc:.4f} >= best={best_prev:.4f})")
            # Promouvoir ce modele comme 'champion'
            client.set_registered_model_alias(model_name, "champion", result.version)
        else:
            print(f"[evaluate] ⏭️  Modele NON enregistre "
                  f"(acc={acc:.4f} < best_actuel={best_prev:.4f})")


if __name__ == "__main__":
    main()
