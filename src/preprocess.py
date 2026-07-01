"""Étape 1 — Prétraitement des données.

Lit les données brutes (S3 ou local), nettoie / met à l'échelle,
et écrit la version prétraitée sur S3. Logue l'étape dans MLflow.
"""
import mlflow
from sklearn.preprocessing import StandardScaler

from src.utils import load_config, read_csv_anywhere, write_csv_anywhere, setup_mlflow


def main():
    config = load_config()
    params = config["preprocess"]
    target = config["train"]["target_column"]

    mlflow = setup_mlflow(config)

    with mlflow.start_run(run_name="preprocess"):
        print(f"[preprocess] Lecture des donnees brutes : {params['input']}")
        df = read_csv_anywhere(params["input"])
        mlflow.log_param("raw_rows", len(df))
        mlflow.log_param("raw_columns", df.shape[1])

        # 1. Supprimer les doublons
        df = df.drop_duplicates()

        # 2. Gérer les valeurs manquantes (imputation par la médiane des colonnes numériques)
        num_cols = df.select_dtypes(include="number").columns
        df[num_cols] = df[num_cols].fillna(df[num_cols].median())

        # 3. Mise à l'échelle des features (on ne touche pas à la cible)
        feature_cols = [c for c in num_cols if c != target]
        scaler = StandardScaler()
        df[feature_cols] = scaler.fit_transform(df[feature_cols])

        print(f"[preprocess] Ecriture des donnees traitees : {params['output']}")
        write_csv_anywhere(df, params["output"])

        mlflow.log_param("processed_rows", len(df))
        mlflow.log_param("n_features", len(feature_cols))
        print("[preprocess] Termine.")


if __name__ == "__main__":
    main()
