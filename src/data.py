from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.config import (
    DROP_IF_PRESENT,
    GROUP_COLUMN_CANDIDATES,
    LAG_SOURCE_FEATURES,
    LAG_STEPS,
    RANDOM_STATE,
    ROLLING_WINDOWS,
    TEST_SIZE,
    TIME_COLUMN_CANDIDATES,
)


@dataclass
class DatasetSplit:
    X_train: pd.DataFrame
    X_test: pd.DataFrame
    y_train: pd.Series
    y_test: pd.Series
    feature_columns: list[str]
    numeric_features: list[str]
    categorical_features: list[str]
    temporal_features: list[str]


class IQRClipper(BaseEstimator, TransformerMixin):
    def __init__(self, factor: float = 1.5):
        self.factor = factor

    def fit(self, X, y=None):
        X_array = np.asarray(X, dtype=float)
        q1 = np.nanquantile(X_array, 0.25, axis=0)
        q3 = np.nanquantile(X_array, 0.75, axis=0)
        iqr = q3 - q1
        self.lower_bounds_ = q1 - self.factor * iqr
        self.upper_bounds_ = q3 + self.factor * iqr
        return self

    def transform(self, X):
        X_array = np.asarray(X, dtype=float)
        return np.clip(X_array, self.lower_bounds_, self.upper_bounds_)


def load_dataset(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset introuvable: {path}")

    df = pd.read_csv(path)
    df.columns = [clean_column_name(col) for col in df.columns]
    return df


def clean_column_name(column: str) -> str:
    return (
        str(column)
        .strip()
        .lower()
        .replace(" ", "_")
        .replace("-", "_")
        .replace("/", "_")
    )


def find_first_existing_column(columns: list[str], candidates: list[str]) -> str | None:
    column_set = set(columns)
    for candidate in candidates:
        if candidate in column_set:
            return candidate
    return None


def add_temporal_features(df: pd.DataFrame, target: str) -> tuple[pd.DataFrame, list[str], list[str]]:
    data = df.copy()
    columns = data.columns.tolist()
    time_col = find_first_existing_column(columns, TIME_COLUMN_CANDIDATES)
    group_col = find_first_existing_column(columns, GROUP_COLUMN_CANDIDATES)

    if not time_col:
        return data, [], []

    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    sort_columns = [col for col in [group_col, time_col] if col]
    data = data.sort_values(sort_columns).reset_index(drop=True)

    created_features: list[str] = []
    raw_temporal_columns = [time_col]

    data[f"{time_col}_hour"] = data[time_col].dt.hour
    data[f"{time_col}_dayofweek"] = data[time_col].dt.dayofweek
    created_features.extend([f"{time_col}_hour", f"{time_col}_dayofweek"])

    if group_col:
        first_timestamp = data.groupby(group_col)[time_col].transform("min")
        data[f"{time_col}_hours_since_machine_start"] = (
            data[time_col] - first_timestamp
        ).dt.total_seconds() / 3600
    else:
        first_timestamp = data[time_col].min()
        data[f"{time_col}_hours_since_machine_start"] = (
            data[time_col] - first_timestamp
        ).dt.total_seconds() / 3600
    created_features.append(f"{time_col}_hours_since_machine_start")

    available_lag_sources = [
        col
        for col in LAG_SOURCE_FEATURES
        if col in data.columns and col != target and pd.api.types.is_numeric_dtype(data[col])
    ]

    if group_col:
        grouped = data.groupby(group_col, sort=False)
        for col in available_lag_sources:
            for lag in LAG_STEPS:
                feature_name = f"{col}_lag_{lag}"
                data[feature_name] = grouped[col].shift(lag)
                created_features.append(feature_name)

            delta_name = f"{col}_delta_1"
            data[delta_name] = data[col] - data[f"{col}_lag_1"]
            created_features.append(delta_name)

            for window in ROLLING_WINDOWS:
                rolling_name = f"{col}_rolling_{window}_mean"
                data[rolling_name] = grouped[col].transform(
                    lambda series: series.shift(1).rolling(window=window, min_periods=1).mean()
                )
                created_features.append(rolling_name)
    else:
        for col in available_lag_sources:
            for lag in LAG_STEPS:
                feature_name = f"{col}_lag_{lag}"
                data[feature_name] = data[col].shift(lag)
                created_features.append(feature_name)

            delta_name = f"{col}_delta_1"
            data[delta_name] = data[col] - data[f"{col}_lag_1"]
            created_features.append(delta_name)

            for window in ROLLING_WINDOWS:
                rolling_name = f"{col}_rolling_{window}_mean"
                data[rolling_name] = data[col].shift(1).rolling(window=window, min_periods=1).mean()
                created_features.append(rolling_name)

    return data, created_features, raw_temporal_columns


def prepare_supervised_dataset(
    df: pd.DataFrame,
    target: str,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> DatasetSplit:
    target = clean_column_name(target)
    if target not in df.columns:
        raise ValueError(
            f"La cible '{target}' est absente du dataset. Colonnes disponibles: {list(df.columns)}"
        )

    data = df.copy()
    data = data.drop_duplicates()
    data = data.dropna(subset=[target])
    data, temporal_features, raw_temporal_columns = add_temporal_features(data, target)

    y = data[target]
    X = data.drop(columns=[target])

    leakage_columns = [col for col in DROP_IF_PRESENT if col in X.columns]
    X = X.drop(columns=leakage_columns)
    X = X.drop(columns=[col for col in raw_temporal_columns if col in X.columns])

    identifier_columns = [
        col
        for col in X.columns
        if col == "id" or col.endswith("_id") or X[col].nunique(dropna=True) / max(len(X), 1) > 0.95
    ]
    X = X.drop(columns=identifier_columns)

    numeric_features = X.select_dtypes(include=["number", "bool"]).columns.tolist()
    categorical_features = [col for col in X.columns if col not in numeric_features]

    if raw_temporal_columns:
        split_index = int(len(X) * (1 - test_size))
        X_train = X.iloc[:split_index]
        X_test = X.iloc[split_index:]
        y_train = y.iloc[:split_index]
        y_test = y.iloc[split_index:]
    else:
        stratify = y if y.nunique() <= 20 else None
        X_train, X_test, y_train, y_test = train_test_split(
            X,
            y,
            test_size=test_size,
            random_state=random_state,
            stratify=stratify,
        )

    return DatasetSplit(
        X_train=X_train,
        X_test=X_test,
        y_train=y_train,
        y_test=y_test,
        feature_columns=X.columns.tolist(),
        numeric_features=numeric_features,
        categorical_features=categorical_features,
        temporal_features=temporal_features,
    )


def build_preprocessor(
    numeric_features: list[str],
    categorical_features: list[str],
    scale_numeric: bool = True,
) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [
        ("imputer", SimpleImputer(strategy="median")),
        ("outlier_clipper", IQRClipper(factor=1.5)),
    ]
    if scale_numeric:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_features),
            ("cat", categorical_pipeline, categorical_features),
        ],
        remainder="drop",
    )
