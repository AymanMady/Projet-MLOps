# Projet MLOps — Prédiction du Diabète 🩺

Projet du module **MLOps et Déploiement de modèles** (SupNum, Master DEML M1 2026).
Pipeline ML complet, reproductible et déployé : de la donnée brute à l'application web.


## 1. Problème ML & dataset
On traite un problème de **classification binaire** : prédire si un patient est diabétique
(`Outcome = 1`) ou non (`Outcome = 0`) à partir de mesures cliniques (glucose, IMC, âge, etc.).

- **Dataset** : Pima Indians Diabetes (~768 lignes, 8 features numériques).
- **Modèle** : `RandomForestClassifier` (scikit-learn).
- **Métriques suivies** : accuracy, F1, précision, rappel.

## 2. Architecture technique

```
PC local (code, DVC, pipeline)
   │  logs                       ┌── EC2 #1 : MLflow Tracking Server (:5000)
   ├──────────────────────────► │      └─ artefacts & modèles sur S3
   │                             │
   ├── S3 bucket ────────────────┤   data/raw · data/processed · dvcstore · modèles
   │                             │
   └── push code ──► EC2 #2 : App Flask (:5001) ─► charge le modèle "champion"
                                     du MLflow Model Registry
```

- **S3** : stockage partagé (données brutes, prétraitées, cache DVC, artefacts).
- **MLflow (EC2 #1)** : tracking des expériences + Model Registry.
- **DVC** : versioning des données/modèles + pipeline reproductible (`dvc repro`).
- **Flask (EC2 #2)** : sert le dernier modèle enregistré.

## 3. Structure du repository
```
├── params.yaml          # configuration centrale
├── dvc.yaml             # pipeline DVC (preprocess → train → evaluate)
├── src/
│   ├── preprocess.py    # nettoyage + scaling → S3
│   ├── train.py         # entraînement + logging MLflow
│   └── evaluate.py      # évaluation + enregistrement du meilleur modèle
└── app/                 # application Flask
```

## 4. Reproduire le projet

### Prérequis
```bash
pip install -r requirements.txt
aws configure            # credentials AWS
```

### Configurer
Éditer `params.yaml` : remplacer `YOUR_BUCKET` et `YOUR_IP_ADRESS`.

### Lancer le pipeline
```bash
dvc pull            # récupérer les données versionnées
dvc repro           # preprocess → train → evaluate
dvc push            # versionner les sorties
```

### Lancer l'application
```bash
python app/app.py   # http://localhost:5001
```

