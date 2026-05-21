from __future__ import annotations

from typing import Any

import numpy as np
from fastapi import FastAPI, HTTPException

from src.predict import load_artifact, predict_one


app = FastAPI(
    title="Maintenance Predictive API",
    description="API d'inference pour la prediction de panne industrielle.",
    version="1.0.0",
)


def to_native(value: Any) -> Any:
    if isinstance(value, dict):
        return {key: to_native(item) for key, item in value.items()}
    if isinstance(value, list):
        return [to_native(item) for item in value]
    if isinstance(value, tuple):
        return tuple(to_native(item) for item in value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    return value


@app.get("/health")
def health() -> dict[str, Any]:
    try:
        artifact = load_artifact()
        return {
            "status": "ok",
            "model_loaded": True,
            "model_name": artifact["model_name"],
            "target": artifact["target"],
        }
    except FileNotFoundError:
        return {"status": "warning", "model_loaded": False}


@app.get("/model-info")
def model_info() -> dict[str, Any]:
    try:
        artifact = load_artifact()
        return {
            "model_name": artifact["model_name"],
            "target": artifact["target"],
            "feature_columns": artifact["feature_columns"],
            "temporal_features": artifact.get("temporal_features", []),
            "metrics": to_native(artifact["metrics"]),
        }
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.post("/predict")
def predict(payload: dict[str, Any]) -> dict[str, Any]:
    try:
        artifact = load_artifact()
        return predict_one(payload, artifact)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
