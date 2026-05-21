from __future__ import annotations

import pandas as pd


def add_persistent_risk_alerts(
    df: pd.DataFrame,
    risk_col: str = "risk_probability",
    group_col: str = "machine_id",
    time_col: str = "timestamp",
    threshold: float = 0.5,
    smoothing_window: int = 5,
    persistence_points: int = 3,
) -> pd.DataFrame:
    """Smooth model probabilities and keep only sustained alerts.

    A single high probability can be a noisy spike. The dashboard should alert when
    risk is high and remains high for several consecutive observations.
    """
    if risk_col not in df.columns:
        raise ValueError(f"Colonne de risque absente: {risk_col}")

    smoothing_window = max(int(smoothing_window), 1)
    persistence_points = max(int(persistence_points), 1)

    sort_cols = [col for col in [group_col, time_col] if col in df.columns]
    result = df.sort_values(sort_cols).copy() if sort_cols else df.copy()

    if group_col in result.columns:
        grouped_risk = result.groupby(group_col, sort=False)[risk_col]
        result["risk_smoothed"] = grouped_risk.transform(
            lambda series: series.rolling(
                window=smoothing_window,
                min_periods=1,
            ).mean()
        )
        result["risk_raw_above_threshold"] = result[risk_col] >= threshold
        result["risk_smoothed_above_threshold"] = result["risk_smoothed"] >= threshold
        result["risk_persistence_count"] = result.groupby(group_col, sort=False)[
            "risk_smoothed_above_threshold"
        ].transform(
            lambda series: series.rolling(
                window=persistence_points,
                min_periods=persistence_points,
            ).sum()
        )
    else:
        result["risk_smoothed"] = result[risk_col].rolling(
            window=smoothing_window,
            min_periods=1,
        ).mean()
        result["risk_raw_above_threshold"] = result[risk_col] >= threshold
        result["risk_smoothed_above_threshold"] = result["risk_smoothed"] >= threshold
        result["risk_persistence_count"] = result["risk_smoothed_above_threshold"].rolling(
            window=persistence_points,
            min_periods=persistence_points,
        ).sum()

    result["risk_persistence_count"] = result["risk_persistence_count"].fillna(0).astype(int)
    result["persistent_alert"] = result["risk_persistence_count"] >= persistence_points
    result["ignored_spike"] = result["risk_raw_above_threshold"] & ~result["persistent_alert"]
    return result


def estimate_sampling_interval_minutes(
    df: pd.DataFrame,
    group_col: str = "machine_id",
    time_col: str = "timestamp",
) -> float | None:
    if time_col not in df.columns:
        return None

    data = df.copy()
    data[time_col] = pd.to_datetime(data[time_col], errors="coerce")
    if group_col in data.columns:
        diffs = data.sort_values([group_col, time_col]).groupby(group_col)[time_col].diff()
    else:
        diffs = data.sort_values(time_col)[time_col].diff()

    minutes = diffs.dt.total_seconds().dropna() / 60
    if minutes.empty:
        return None
    return float(minutes.median())
