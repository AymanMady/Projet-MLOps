"""Fonctions utilitaires partagées : chargement de la config, S3, MLflow."""
import io
import os
from urllib.parse import urlparse

import yaml
import pandas as pd

CONFIG_PATH = os.path.join(os.path.dirname(__file__), "..", "params.yaml")


def _s3_client():
    """Crée un client S3 boto3 (synchrone, compatible botocore récent).

    On passe par boto3 plutôt que par s3fs/aiobotocore, car aiobotocore
    ne supporte pas toujours la version de botocore installée.
    """
    import boto3

    region = os.environ.get("AWS_DEFAULT_REGION")
    return boto3.client("s3", region_name=region) if region else boto3.client("s3")


def _split_s3(uri: str):
    """Découpe s3://bucket/key en (bucket, key)."""
    parsed = urlparse(uri)
    return parsed.netloc, parsed.path.lstrip("/")


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
    """Lit un CSV depuis S3 (s3://...) ou un chemin local, de façon transparente."""
    if path.startswith("s3://"):
        bucket, key = _split_s3(path)
        obj = _s3_client().get_object(Bucket=bucket, Key=key)
        return pd.read_csv(io.BytesIO(obj["Body"].read()))
    return pd.read_csv(path)


def write_csv_anywhere(df: pd.DataFrame, path: str) -> None:
    """Écrit un CSV vers S3 (s3://...) ou local. Crée les dossiers locaux au besoin."""
    if path.startswith("s3://"):
        bucket, key = _split_s3(path)
        buf = io.StringIO()
        df.to_csv(buf, index=False)
        _s3_client().put_object(Bucket=bucket, Key=key, Body=buf.getvalue().encode("utf-8"))
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False)


def setup_mlflow(config: dict):
    """Configure MLflow (tracking URI + experiment) et retourne le module mlflow."""
    import mlflow

    mlflow.set_tracking_uri(config["mlflow"]["MLFLOW_TRACKING_URI"])
    mlflow.set_experiment(config["mlflow"]["EXPERIMENT_NAME"])
    return mlflow
