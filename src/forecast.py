from __future__ import annotations

import math
from typing import Any

import numpy as np
import pandas as pd

from src.config import DROP_IF_PRESENT
from src.data import add_temporal_features, clean_column_name
from src.postprocessing import estimate_sampling_interval_minutes


def forecast_future_failures(
    history_df: pd.DataFrame,
    artifact: dict[str, Any],
    horizon_hours: int = 24,
    step_minutes: int | None = None,
    trend_window: int = 12,
    scenario: str = "stable_recent",
    trend_strength: float = 0.15,
) -> pd.DataFrame:
    """Forecast future machine failure risk after the last observed timestamp.

    The model is not a pure time-series forecaster. We build short-term future
    sensor scenarios, recompute temporal features/lags, then apply the trained
    classifier.
    """
    if "timestamp" not in history_df.columns:
        raise ValueError("La prevision future exige une colonne timestamp.")
    if "machine_id" not in history_df.columns:
        raise ValueError("La prevision future exige une colonne machine_id.")

    history = history_df.copy()
    history.columns = [clean_column_name(col) for col in history.columns]
    history["timestamp"] = pd.to_datetime(history["timestamp"], errors="coerce")
    history = history.dropna(subset=["timestamp", "machine_id"]).sort_values(["machine_id", "timestamp"])

    if step_minutes is None:
        estimated_interval = estimate_sampling_interval_minutes(history)
        step_minutes = int(round(estimated_interval or 15))
    step_minutes = max(int(step_minutes), 1)

    steps = max(int(math.ceil((horizon_hours * 60) / step_minutes)), 1)
    future_raw = build_future_raw_rows(
        history_df=history,
        steps=steps,
        step_minutes=step_minutes,
        trend_window=trend_window,
        scenario=scenario,
        trend_strength=trend_strength,
    )

    if future_raw.empty:
        return future_raw

    combined = pd.concat([history, future_raw], ignore_index=True, sort=False)
    engineered = add_temporal_features(combined, artifact["target"])[0]
    future_engineered = engineered[engineered["is_future"] == True].copy()  # noqa: E712

    feature_columns = artifact["feature_columns"]
    for col in feature_columns:
        if col not in future_engineered.columns:
            future_engineered[col] = np.nan

    pipeline = artifact["pipeline"]
    future_engineered["predicted_failure"] = pipeline.predict(future_engineered[feature_columns])
    if hasattr(pipeline, "predict_proba"):
        future_engineered["risk_probability"] = pipeline.predict_proba(future_engineered[feature_columns])[:, 1]
    else:
        future_engineered["risk_probability"] = future_engineered["predicted_failure"].astype(float)

    return future_engineered.sort_values(["machine_id", "timestamp"]).reset_index(drop=True)


def build_future_raw_rows(
    history_df: pd.DataFrame,
    steps: int,
    step_minutes: int,
    trend_window: int,
    scenario: str = "stable_recent",
    trend_strength: float = 0.15,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    numeric_cols = history_df.select_dtypes(include="number").columns.tolist()
    numeric_cols = [
        col
        for col in numeric_cols
        if col not in DROP_IF_PRESENT
        and col not in {"failure_within_24h", "machine_id"}
        and not col.endswith("_id")
    ]
    categorical_cols = [
        col
        for col in history_df.columns
        if col not in numeric_cols
        and col not in DROP_IF_PRESENT
        and col not in {"timestamp", "failure_within_24h"}
    ]

    clip_bounds = build_clip_bounds(history_df, numeric_cols)
    step_hours = step_minutes / 60
    scenario = scenario.lower().strip()
    if scenario not in {"stable_recent", "damped_trend"}:
        raise ValueError("scenario doit etre 'stable_recent' ou 'damped_trend'.")
    trend_strength = float(np.clip(trend_strength, 0.0, 1.0))

    for machine_id, machine_df in history_df.groupby("machine_id", sort=False):
        machine_df = machine_df.sort_values("timestamp")
        last_row = machine_df.iloc[-1]
        last_timestamp = last_row["timestamp"]

        numeric_trends = {}
        numeric_baselines = {}
        numeric_shift_limits = {}
        for col in numeric_cols:
            if col not in machine_df.columns:
                continue
            recent = pd.to_numeric(machine_df[col], errors="coerce").dropna().tail(max(trend_window, 2))
            if recent.empty:
                numeric_baselines[col] = np.nan
                numeric_trends[col] = 0.0
                numeric_shift_limits[col] = 0.0
                continue

            numeric_baselines[col] = float(recent.median())
            numeric_trends[col] = estimate_recent_trend(machine_df[col], trend_window)
            std = float(recent.std()) if len(recent) > 1 else 0.0
            q25 = float(recent.quantile(0.25))
            q75 = float(recent.quantile(0.75))
            iqr = q75 - q25
            numeric_shift_limits[col] = max(std, iqr, 1e-9) * 1.5

        for step in range(1, steps + 1):
            future_timestamp = last_timestamp + pd.Timedelta(minutes=step * step_minutes)
            row: dict[str, Any] = {
                "timestamp": future_timestamp,
                "machine_id": machine_id,
                "is_future": True,
                "forecast_step": step,
                "forecast_horizon_hours": step * step_hours,
            }

            for col in categorical_cols:
                if col in machine_df.columns:
                    row[col] = last_row[col]

            for col in numeric_cols:
                if col not in machine_df.columns:
                    continue

                if col == "hours_since_maintenance":
                    value = float(last_row[col]) + step * step_hours
                else:
                    baseline = numeric_baselines.get(col, np.nan)
                    if pd.isna(baseline):
                        baseline = float(last_row[col]) if pd.notna(last_row[col]) else float(machine_df[col].median())

                    value = baseline
                    if scenario == "damped_trend":
                        raw_shift = numeric_trends[col] * step * trend_strength
                        shift_limit = numeric_shift_limits.get(col, 0.0)
                        value = baseline + float(np.clip(raw_shift, -shift_limit, shift_limit))

                lower, upper = clip_bounds.get(col, (-np.inf, np.inf))
                row[col] = float(np.clip(value, lower, upper))

            rows.append(row)

    return pd.DataFrame(rows)


def estimate_recent_trend(series: pd.Series, trend_window: int) -> float:
    recent = pd.to_numeric(series, errors="coerce").dropna().tail(max(trend_window, 2) + 1)
    if len(recent) < 2:
        return 0.0

    diffs = recent.diff().dropna()
    if diffs.empty:
        return 0.0

    return float(diffs.median())


def build_clip_bounds(df: pd.DataFrame, numeric_cols: list[str]) -> dict[str, tuple[float, float]]:
    bounds: dict[str, tuple[float, float]] = {}
    for col in numeric_cols:
        series = pd.to_numeric(df[col], errors="coerce").dropna()
        if series.empty:
            continue

        q01 = float(series.quantile(0.01))
        q99 = float(series.quantile(0.99))
        margin = (q99 - q01) * 0.25
        bounds[col] = (q01 - margin, q99 + margin)

    return bounds
