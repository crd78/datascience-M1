from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from src.config import DEFAULT_DATA_PATH, REPORTS_DIR
from src.data import clean_column_name, load_dataset


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Analyse qualite du dataset industriel.")
    parser.add_argument("--data", default=str(DEFAULT_DATA_PATH))
    parser.add_argument("--target", default="failure_within_24h")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    df = load_dataset(args.data)
    target = clean_column_name(args.target)
    numeric = df.select_dtypes(include="number")

    missing = df.isna().mean().sort_values(ascending=False)
    unique_counts = df.nunique(dropna=True).sort_values()

    q1 = numeric.quantile(0.25)
    q3 = numeric.quantile(0.75)
    iqr = q3 - q1
    outlier_rate = ((numeric.lt(q1 - 1.5 * iqr)) | (numeric.gt(q3 + 1.5 * iqr))).mean()
    outlier_rate = outlier_rate.sort_values(ascending=False)

    correlation_to_target = pd.Series(dtype=float)
    if target in numeric.columns:
        correlation_to_target = numeric.corr(numeric_only=True)[target].abs().sort_values(ascending=False)

    summary = pd.DataFrame(
        {
            "dtype": df.dtypes.astype(str),
            "missing_rate": missing,
            "unique_values": unique_counts,
            "outlier_rate_iqr": outlier_rate,
            "abs_corr_with_target": correlation_to_target,
        }
    ).fillna(0)
    summary.to_csv(REPORTS_DIR / "data_quality_report.csv")

    recommendations = {
        "target": target,
        "rows": int(len(df)),
        "columns": int(len(df.columns)),
        "class_distribution": df[target].value_counts(normalize=True).to_dict() if target in df else {},
        "removed_as_leakage_for_failure_prediction": [
            "failure_type",
            "rul_hours",
            "estimated_repair_cost",
        ],
        "identifier_columns_to_drop": ["machine_id"],
        "raw_low_signal_columns_to_monitor": [
            col
            for col, value in correlation_to_target.items()
            if col != target and value < 0.03
        ],
        "outlier_strategy": "IQR clipping inside the numeric preprocessing pipeline, fitted only on train data.",
        "lag_strategy": "Limited lags: lag_1, lag_3, lag_6 and lag_12 plus delta_1 and rolling_3/rolling_6 means for selected sensor/process variables.",
    }
    (REPORTS_DIR / "data_quality_report.json").write_text(
        json.dumps(recommendations, indent=2),
        encoding="utf-8",
    )

    print("Analyse terminee.")
    print(f"Rapport colonnes: {REPORTS_DIR / 'data_quality_report.csv'}")
    print(f"Recommandations: {REPORTS_DIR / 'data_quality_report.json'}")
    print(summary.sort_values("abs_corr_with_target", ascending=False).head(12))


if __name__ == "__main__":
    main()
