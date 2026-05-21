from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import (
    GradientBoostingClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    fbeta_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    make_scorer,
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier

from src.config import RANDOM_STATE
from src.data import build_preprocessor


@dataclass
class ModelResult:
    name: str
    pipeline: Pipeline
    metrics: dict[str, float]


def build_candidate_models(
    numeric_features: list[str],
    categorical_features: list[str],
    y_train: pd.Series,
) -> dict[str, Pipeline]:
    class_counts = y_train.value_counts()
    negative_count = int(class_counts.get(0, 0))
    positive_count = int(class_counts.get(1, 1))
    scale_pos_weight = negative_count / max(positive_count, 1)

    model_specs = {
        "Logistic Regression": LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=RANDOM_STATE,
        ),
        "Random Forest": RandomForestClassifier(
            n_estimators=250,
            max_depth=None,
            min_samples_leaf=2,
            class_weight="balanced",
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
        "Hist Gradient Boosting": HistGradientBoostingClassifier(random_state=RANDOM_STATE),
        "XGBoost": XGBClassifier(
            n_estimators=350,
            max_depth=4,
            learning_rate=0.05,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            reg_lambda=1.0,
            objective="binary:logistic",
            eval_metric="aucpr",
            scale_pos_weight=scale_pos_weight,
            n_jobs=-1,
            random_state=RANDOM_STATE,
        ),
        "MLP Neural Network": MLPClassifier(
            hidden_layer_sizes=(64, 32),
            activation="relu",
            alpha=0.001,
            learning_rate_init=0.001,
            max_iter=300,
            early_stopping=True,
            random_state=RANDOM_STATE,
        ),
    }

    pipelines: dict[str, Pipeline] = {}
    for name, estimator in model_specs.items():
        preprocessor = build_preprocessor(
            numeric_features=numeric_features,
            categorical_features=categorical_features,
            scale_numeric=True,
        )
        pipelines[name] = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", estimator),
            ]
        )

    return pipelines


def train_and_evaluate_models(
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    numeric_features: list[str],
    categorical_features: list[str],
    use_cross_validation: bool = True,
) -> list[ModelResult]:
    results: list[ModelResult] = []
    candidates = build_candidate_models(numeric_features, categorical_features, y_train)

    for name, pipeline in candidates.items():
        pipeline.fit(X_train, y_train)
        metrics = evaluate_classifier(pipeline, X_test, y_test)

        if use_cross_validation:
            metrics.update(cross_validation_scores(pipeline, X_train, y_train))

        results.append(ModelResult(name=name, pipeline=pipeline, metrics=metrics))

    return results


def evaluate_classifier(pipeline: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    y_pred = pipeline.predict(X_test)
    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1": f1_score(y_test, y_pred, zero_division=0),
        "f2": fbeta_score(y_test, y_pred, beta=2, zero_division=0),
    }

    if hasattr(pipeline, "predict_proba"):
        try:
            y_score = pipeline.predict_proba(X_test)[:, 1]
            metrics["roc_auc"] = roc_auc_score(y_test, y_score)
            metrics["pr_auc"] = average_precision_score(y_test, y_score)
        except Exception:
            metrics["roc_auc"] = np.nan
            metrics["pr_auc"] = np.nan
    else:
        metrics["roc_auc"] = np.nan
        metrics["pr_auc"] = np.nan

    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
    specificity = tn / (tn + fp) if (tn + fp) else 0
    false_negative_rate = fn / (fn + tp) if (fn + tp) else 0
    false_positive_rate = fp / (fp + tn) if (fp + tn) else 0
    metrics.update(
        {
            "tn": tn,
            "fp": fp,
            "fn": fn,
            "tp": tp,
            "specificity": specificity,
            "false_negative_rate": false_negative_rate,
            "false_positive_rate": false_positive_rate,
        }
    )
    return metrics


def cross_validation_scores(pipeline: Pipeline, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
    class_counts = y_train.value_counts()
    min_class_count = int(class_counts.min())
    n_splits = max(2, min(5, min_class_count))

    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=RANDOM_STATE)
    scoring = {
        "f1": "f1",
        "f2": make_scorer(fbeta_score, beta=2, zero_division=0),
        "recall": "recall",
        "precision": "precision",
        "roc_auc": "roc_auc",
        "pr_auc": "average_precision",
        "balanced_accuracy": "balanced_accuracy",
    }
    scores = cross_validate(
        pipeline,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
        error_score=np.nan,
    )

    return {
        "cv_f1_mean": float(np.nanmean(scores["test_f1"])),
        "cv_f2_mean": float(np.nanmean(scores["test_f2"])),
        "cv_recall_mean": float(np.nanmean(scores["test_recall"])),
        "cv_precision_mean": float(np.nanmean(scores["test_precision"])),
        "cv_roc_auc_mean": float(np.nanmean(scores["test_roc_auc"])),
        "cv_pr_auc_mean": float(np.nanmean(scores["test_pr_auc"])),
        "cv_balanced_accuracy_mean": float(np.nanmean(scores["test_balanced_accuracy"])),
    }


def select_best_model(results: list[ModelResult], metric: str = "f1") -> ModelResult:
    return max(results, key=lambda result: result.metrics.get(metric, float("-inf")))


def compute_feature_importance(
    pipeline: Pipeline,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    max_rows: int = 2000,
) -> pd.DataFrame:
    if len(X_test) > max_rows:
        X_sample = X_test.sample(max_rows, random_state=RANDOM_STATE)
        y_sample = y_test.loc[X_sample.index]
    else:
        X_sample = X_test
        y_sample = y_test

    scoring = make_scorer(fbeta_score, beta=2, zero_division=0) if y_test.nunique() == 2 else "accuracy"
    result = permutation_importance(
        pipeline,
        X_sample,
        y_sample,
        n_repeats=8,
        random_state=RANDOM_STATE,
        scoring=scoring,
        n_jobs=-1,
    )

    importance = pd.DataFrame(
        {
            "feature": X_sample.columns,
            "importance_mean": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    return importance.sort_values("importance_mean", ascending=False)
