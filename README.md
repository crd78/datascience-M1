# Maintenance Predictive Industrielle

Projet Data Science pour predire les pannes industrielles a partir de donnees capteurs.

## Objectif

Construire un MVP complet de maintenance predictive :

- preparation et analyse des donnees ;
- creation de variables temporelles et de lags quand `timestamp` est disponible ;
- entrainement de plusieurs modeles Machine Learning ;
- comparaison quantitative des performances ;
- selection du meilleur modele ;
- interpretation par importance des variables ;
- dashboard Streamlit pour un utilisateur metier ;
- API FastAPI optionnelle pour exposer le modele.

La cible choisie par defaut est `failure_within_24h`, donc le probleme est une classification binaire :

- `0` : pas de panne prevue dans les 24h ;
- `1` : panne probable dans les 24h.

## Structure

```text
.
|-- data/
|   |-- README.md
|   `-- predictive_maintenance_v3.csv
|-- dashboard/
|   `-- app.py
|-- models/
|   `.gitkeep
|-- reports/
|   `.gitkeep
|-- scripts/
|   `-- analyze_dataset.py
|-- src/
|   |-- __init__.py
|   |-- api.py
|   |-- config.py
|   |-- data.py
|   |-- modeling.py
|   |-- predict.py
|   `-- train.py
|-- requirements.txt
`-- README.md
```

## Installation

```bash
pip install -r requirements.txt
```

## Donnees

Le seul dataset utilise pour ce TP est celui du sujet :

`predictive_maintenance_v3.csv`

Place le fichier dans le dossier `data/`.

## Entrainement

Avant l'entrainement, tu peux generer le rapport qualite donnees :

```bash
python scripts/analyze_dataset.py
```

Ce script produit :

- `reports/data_quality_report.csv`
- `reports/data_quality_report.json`

Il resume les valeurs manquantes, les outliers IQR, les correlations avec la cible et les
variables a exclure pour eviter le data leakage.

Avec le dataset du sujet :

```bash
python -m src.train
```

Le script entraine 5 modeles :

- Logistic Regression ;
- Random Forest ;
- Gradient Boosting ;
- Hist Gradient Boosting ;
- XGBoost ;
- MLP Classifier, modele de type reseau de neurones.

Si le dataset contient une colonne `timestamp` et une colonne `machine_id`, le pipeline cree
automatiquement des features temporelles :

- `timestamp_hour`
- `timestamp_dayofweek`
- `timestamp_hours_since_machine_start`
- `vibration_rms_lag_1`, `vibration_rms_lag_3`, `vibration_rms_lag_6`, `vibration_rms_lag_12`
- `temperature_motor_lag_1`, `current_phase_avg_lag_6`, `pressure_level_lag_12`, etc.
- deltas comme `vibration_rms_delta_1`
- moyennes glissantes comme `vibration_rms_rolling_3_mean` et `vibration_rms_rolling_6_mean`

Le split train/test devient chronologique quand `timestamp` existe, afin d'eviter de tester
sur le passe avec un modele entraine sur le futur.

Les variables numeriques passent aussi par un clipping IQR dans le pipeline sklearn. Ce
traitement est ajuste uniquement sur le train set pour eviter la fuite de donnees.

Les resultats sont crees dans :

- `reports/model_metrics.csv`
- `reports/model_metrics.json`
- `reports/feature_importance.csv`
- `models/best_model.joblib`

Pour la cible `failure_within_24h`, le meilleur modele est choisi sur le **F2-score**.
Cette metrique donne plus de poids au recall, ce qui est plus coherent en maintenance
predictive : rater une panne future est souvent plus grave que declencher une alerte en trop.

Les metriques principales du rapport sont :

- `recall` : proportion de pannes detectees ;
- `f2` : compromis qui favorise le recall ;
- `pr_auc` : plus informative que ROC-AUC quand la classe panne est rare ;
- `balanced_accuracy` : accuracy corrigee pour le desequilibre de classes ;
- `precision` : proportion d'alertes reellement justifiees ;
- `false_negative_rate` : taux de pannes ratees.

## Dashboard

```bash
streamlit run dashboard/app.py
```

Le dashboard permet de :

- voir les indicateurs du dataset ;
- visualiser les distributions utiles a la maintenance ;
- suivre les risques lisses par machine ;
- projeter le risque futur apres le dernier historique ;
- consulter la comparaison des modeles ;
- saisir un scenario machine ;
- obtenir une prediction en temps reel ;
- afficher les variables les plus importantes.

## API Optionnelle

```bash
uvicorn src.api:app --reload
```

Endpoints :

- `GET /health`
- `GET /model-info`
- `POST /predict`

Exemple de requete :

```json
{
  "vibration_rms": 0.8,
  "temperature_motor": 82.0,
  "rpm": 1650,
  "pressure_level": 6.2,
  "operating_mode": "high_load"
}
```

## Livrables

Pour le rendu, tu peux fournir :

- le code source ;
- les fichiers dans `reports/` ;
- le modele sauvegarde dans `models/` ;
- le rapport de projet ;
- le support de presentation ;
- une demonstration du dashboard.
