import yaml
from pathlib import Path

# Common Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "hour.csv"
PROCESSED_DIR = DATA_DIR / "processed"
TRAIN_PATH = PROCESSED_DIR / "train.csv"
TEST_PATH = PROCESSED_DIR / "test.csv"
MODEL_PATH = PROJECT_ROOT / "models" / "model.joblib"
METRICS_PATH = PROJECT_ROOT / "metrics.json"

# Features selected for modeling
FEATURE_COLS = [
    "season", "yr", "mnth", "hr", "holiday", "weekday", "workingday",
    "weathersit", "temp", "atemp", "hum", "windspeed",
    "hr_sin", "hr_cos", "mnth_sin", "mnth_cos", "weekday_sin", "weekday_cos",
    "is_rush_hour"
]

TARGET_COL = "cnt"

def load_params():
    """Load configuration parameters from params.yaml."""
    params_path = PROJECT_ROOT / "params.yaml"
    with open(params_path, "r") as f:
        return yaml.safe_load(f)