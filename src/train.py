import joblib
import numpy as np
import pandas as pd
import yaml
from sklearn.ensemble import HistGradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import Ridge

def train():
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)["train"]

    # Load preprocessed training data
    train_df = pd.read_csv("data/processed/train.csv")
    
    # Identify non-feature columns to drop
    drop_cols = ["cnt", "dteday", "instant"]
    existing_drop_cols = [col for col in drop_cols if col in train_df.columns]
    
    X_train = train_df.drop(columns=existing_drop_cols)
    y_train = train_df["cnt"]

    # Optional log transformation on target variable
    if params.get("log_target", False):
        y_train = np.log1p(y_train)

    model_type = params.get("model_type", "hist_gb")

    if model_type == "random_forest":
        print("Training RandomForestRegressor model...")
        model = RandomForestRegressor(
            n_estimators=params.get("max_iter", 200),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            random_state=params.get("seed", 42),
            n_jobs=-1
        )
    elif model_type == "ridge":
        print("Training Ridge Regression model...")
        model = Ridge(
            alpha=params.get("l2_regularization", 1.0),
            random_state=params.get("seed", 42)
        )
    else:
        print("Training HistGradientBoostingRegressor model...")
        model = HistGradientBoostingRegressor(
            learning_rate=params.get("learning_rate", 0.05),
            max_iter=params.get("max_iter", 200),
            max_depth=params.get("max_depth", 8),
            min_samples_leaf=params.get("min_samples_leaf", 20),
            l2_regularization=params.get("l2_regularization", 1.0),
            random_state=params.get("seed", 42)
        )

    model.fit(X_train, y_train)
    joblib.dump(model, "models/model.joblib")
    print(f"Saved {model_type} model artifact to models/model.joblib")

if __name__ == "__main__":
    train()