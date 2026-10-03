import json
import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import yaml
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score


def evaluate():
    # Load configuration parameters
    with open("params.yaml", "r") as f:
        params = yaml.safe_load(f)

    train_params = params.get("train", {})
    eval_params = params.get("evaluate", {})

    # Load preprocessed test dataset
    test_df = pd.read_csv("data/processed/test.csv")

    # Drop non-feature string/metadata columns
    drop_cols = ["cnt", "dteday", "instant"]
    existing_drop_cols = [col for col in drop_cols if col in test_df.columns]

    X_test = test_df.drop(columns=existing_drop_cols)
    y_test = test_df["cnt"]

    # Load trained model artifact
    model_path = "models/model.joblib"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found at {model_path}. Run training first.")

    model = joblib.load(model_path)
    print(f"Loaded trained model from {model_path}")

    # Generate predictions
    y_pred = model.predict(X_test)

    # Invert log-transformation if log_target was used during training
    if train_params.get("log_target", False):
        y_pred = np.expm1(y_pred)
        y_pred = np.clip(y_pred, 0, None)

    # Calculate evaluation metrics
    mae = float(mean_absolute_error(y_test, y_pred))
    rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
    r2 = float(r2_score(y_test, y_pred))

    metrics = {
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2
    }

    # Save metrics.json in root directory
    metrics_path = "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Metrics saved to {metrics_path}: {metrics}")

    os.makedirs("reports/figures", exist_ok=True)

    # Output 1: Actual vs. Predicted Plot
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, y_pred, alpha=0.3, color="blue")
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], "r--", lw=2)
    plt.xlabel("Actual Bike Counts")
    plt.ylabel("Predicted Bike Counts")
    plt.title("Actual vs. Predicted Bike Sharing Demands")
    plt.tight_layout()
    plot_path = "reports/figures/actual_vs_predicted.png"
    plt.savefig(plot_path)
    plt.close()
    print(f"Plot saved to {plot_path}")

    # Output 2: Feature Importance Plot
    print("Calculating feature importances...")
    plt.figure(figsize=(10, 6))

    if hasattr(model, "feature_importances_"):
        importances = model.feature_importances_
        indices = np.argsort(importances)
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [X_test.columns[i] for i in indices])
    else:
        # Fallback for models without direct feature_importances_ (e.g., Ridge)
        sample_size = min(len(X_test), eval_params.get("importance_sample", 2000))
        X_sample = X_test.iloc[:sample_size]
        y_sample = y_test.iloc[:sample_size]
        result = permutation_importance(
            model, X_sample, y_sample, n_repeats=5, random_state=eval_params.get("seed", 42)
        )
        indices = np.argsort(result.importances_mean)
        plt.barh(range(len(indices)), result.importances_mean[indices], align="center")
        plt.yticks(range(len(indices)), [X_test.columns[i] for i in indices])

    plt.xlabel("Importance Score")
    plt.title("Feature Importances")
    plt.tight_layout()
    fi_plot_path = "reports/figures/feature_importance.png"
    plt.savefig(fi_plot_path)
    plt.close()
    print(f"Plot saved to {fi_plot_path}")


if __name__ == "__main__":
    evaluate()