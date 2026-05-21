from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import pandas as pd
import sklearn

from src.config import (
    BEST_MODEL_PATH,
    DEFAULT_DATA_PATH,
    DEFAULT_TARGET,
    FEATURE_IMPORTANCE_PATH,
    METRICS_CSV_PATH,
    METRICS_JSON_PATH,
    MODELS_DIR,
    REPORTS_DIR,
)
from src.data import load_dataset, prepare_supervised_dataset
from src.modeling import (
    compute_feature_importance,
    select_best_model,
    train_and_evaluate_models,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Entrainement maintenance predictive.")
    parser.add_argument(
        "--data",
        default=str(DEFAULT_DATA_PATH),
        help="Chemin vers predictive_maintenance_v3.csv.",
    )
    parser.add_argument("--target", default=DEFAULT_TARGET, help="Variable cible.")
    parser.add_argument(
        "--no-cv",
        action="store_true",
        help="Desactive la validation croisee pour accelerer l'entrainement.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    split = prepare_supervised_dataset(df, target=args.target)

    results = train_and_evaluate_models(
        X_train=split.X_train,
        X_test=split.X_test,
        y_train=split.y_train,
        y_test=split.y_test,
        numeric_features=split.numeric_features,
        categorical_features=split.categorical_features,
        use_cross_validation=not args.no_cv,
    )

    metrics_df = pd.DataFrame(
        [{"model": result.name, **result.metrics} for result in results]
    ).sort_values("f2", ascending=False)

    metrics_df.to_csv(METRICS_CSV_PATH, index=False)
    METRICS_JSON_PATH.write_text(
        json.dumps(metrics_df.to_dict(orient="records"), indent=2),
        encoding="utf-8",
    )

    best = select_best_model(results, metric="f2")
    feature_importance = compute_feature_importance(best.pipeline, split.X_test, split.y_test)
    feature_importance.to_csv(FEATURE_IMPORTANCE_PATH, index=False)

    artifact = {
        "model_name": best.name,
        "target": args.target,
        "pipeline": best.pipeline,
        "feature_columns": split.feature_columns,
        "numeric_features": split.numeric_features,
        "categorical_features": split.categorical_features,
        "temporal_features": split.temporal_features,
        "metrics": best.metrics,
        "sklearn_version": sklearn.__version__,
    }
    joblib.dump(artifact, BEST_MODEL_PATH)

    print("Entrainement termine.")
    print(f"Meilleur modele: {best.name}")
    print(f"F1-score: {best.metrics['f1']:.4f}")
    print(f"F2-score: {best.metrics['f2']:.4f}")
    print(f"Recall: {best.metrics['recall']:.4f}")
    print(f"PR-AUC: {best.metrics['pr_auc']:.4f}")
    print(f"Modele sauvegarde: {Path(BEST_MODEL_PATH).resolve()}")
    print(f"Rapport metriques: {Path(METRICS_CSV_PATH).resolve()}")
    print(f"Importance variables: {Path(FEATURE_IMPORTANCE_PATH).resolve()}")


if __name__ == "__main__":
    main()
