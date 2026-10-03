import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from common import FEATURE_COLS, MODEL_PATH, TARGET_COL, TRAIN_PATH, load_params


def train_model():
    params = load_params()["train"]

    print(f"Loading training data from {TRAIN_PATH}...")
    train_df = pd.read_csv(TRAIN_PATH)

    X_train = train_df[FEATURE_COLS]
    y_train = train_df[TARGET_COL]

    # Target transformation: log1p stabilizes variance on count data
    if params.get("log_target", True):
        y_train_fit = np.log1p(y_train)
    else:
        y_train_fit = y_train

    print("Training HistGradientBoostingRegressor model...")
    model = HistGradientBoostingRegressor(
        max_iter=params["max_iter"],
        learning_rate=params["learning_rate"],
        max_depth=params["max_depth"],
        min_samples_leaf=params["min_samples_leaf"],
        l2_regularization=params["l2_regularization"],
        random_state=params["seed"],
    )

    model.fit(X_train, y_train_fit)

    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_PATH)
    print(f"Successfully saved trained model artifact to {MODEL_PATH}")


if __name__ == "__main__":
    train_model()