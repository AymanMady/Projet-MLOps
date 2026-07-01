"""Fonctions utilitaires partagées : chargement de la config, S3, MLflow."""
import os
import yaml
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "params.yaml")


def load_config(path: str = CONFIG_PATH) -> dict:
    """Charge params.yaml et exporte les credentials AWS/MLflow dans l'environnement."""
    with open(path, "r") as f:
        config = yaml.safe_load(f)

    aws = config.get("aws", {})
    # On ne force les variables d'env que si elles sont fournies dans le yaml.
    if aws.get("aws_access_key_id"):
        os.environ["AWS_ACCESS_KEY_ID"] = aws["aws_access_key_id"]
    if aws.get("aws_secret_access_key"):
        os.environ["AWS_SECRET_ACCESS_KEY"] = aws["aws_secret_access_key"]
    if aws.get("region_name"):
        os.environ["AWS_DEFAULT_REGION"] = aws["region_name"]

    return config


def read_csv_anywhere(path: str) -> pd.DataFrame:
    """Lit un CSV depuis S3 (s3://...) ou un chemin local, de façon transparente.

    pandas gère les URI s3:// automatiquement grâce à s3fs.
    """
    return pd.read_csv(path)


def write_csv_anywhere(df: pd.DataFrame, path: str) -> None:
    """Écrit un CSV vers S3 (s3://...) ou local. Crée les dossiers locaux au besoin."""
    if not path.startswith("s3://"):
        os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def setup_mlflow(config: dict):
    """Configure MLflow (tracking URI + experiment) et retourne le module mlflow."""
    import mlflow

    mlflow.set_tracking_uri(config["mlflow"]["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["mlflow"]["EXPERIMENT_NAME"])
    return mlflow
