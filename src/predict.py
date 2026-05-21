from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import pandas as pd
import sklearn

from src.config import BEST_MODEL_PATH, DEFAULT_DATA_PATH


def load_artifact(path: str | Path = BEST_MODEL_PATH) -> dict[str, Any]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(
            f"Modele introuvable: {path}. Lance d'abord: python -m src.train --data {DEFAULT_DATA_PATH}"
        )
    artifact = joblib.load(path)
    trained_version = artifact.get("sklearn_version")
    if trained_version and trained_version != sklearn.__version__:
        raise RuntimeError(
            "Modele incompatible avec l'environnement actuel: "
            f"entraine avec scikit-learn {trained_version}, "
            f"charge avec scikit-learn {sklearn.__version__}. "
            "Relance l'entrainement avec la meme venv que Streamlit: "
            f"python -m src.train --data {DEFAULT_DATA_PATH}"
        )
    return artifact


def predict_one(payload: dict[str, Any], artifact: dict[str, Any] | None = None) -> dict[str, Any]:
    artifact = artifact or load_artifact()
    feature_columns = artifact["feature_columns"]
    missing = [col for col in feature_columns if col not in payload]

    if missing:
        raise ValueError(f"Features manquantes: {missing}")

    row = pd.DataFrame([{col: payload[col] for col in feature_columns}])
    pipeline = artifact["pipeline"]
    prediction = int(pipeline.predict(row)[0])

    response: dict[str, Any] = {
        "model_name": artifact["model_name"],
        "target": artifact["target"],
        "prediction": prediction,
    }

    if hasattr(pipeline, "predict_proba"):
        probabilities = pipeline.predict_proba(row)[0]
        response["probability_no_failure"] = float(probabilities[0])
        response["probability_failure"] = float(probabilities[1])
        response["risk_percent"] = round(float(probabilities[1]) * 100, 2)

    return response
