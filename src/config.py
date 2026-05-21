from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DEFAULT_DATA_PATH = DATA_DIR / "predictive_maintenance_v3.csv"

DEFAULT_TARGET = "failure_within_24h"
RANDOM_STATE = 42
TEST_SIZE = 0.2

TIME_COLUMN_CANDIDATES = [
    "timestamp",
    "datetime",
    "date_time",
    "measurement_time",
    "time",
    "date",
]

GROUP_COLUMN_CANDIDATES = [
    "machine_id",
    "equipment_id",
    "asset_id",
    "device_id",
]

LAG_SOURCE_FEATURES = [
    "vibration_rms",
    "temperature_motor",
    "current_phase_avg",
    "hours_since_maintenance",
    "pressure_level",
]

LAG_STEPS = [1, 3]
ROLLING_WINDOWS = [3]

BEST_MODEL_PATH = MODELS_DIR / "best_model.joblib"
METRICS_CSV_PATH = REPORTS_DIR / "model_metrics.csv"
METRICS_JSON_PATH = REPORTS_DIR / "model_metrics.json"
FEATURE_IMPORTANCE_PATH = REPORTS_DIR / "feature_importance.csv"


DROP_IF_PRESENT = {
    "failure_type",
    "rul_hours",
    "maintenance_required",
    "repair_cost",
    "cost_estimate",
    "estimated_repair_cost",
}
