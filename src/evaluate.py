import json
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from common import (
    FEATURE_COLS,
    MODEL_PATH,
    PROJECT_ROOT,
    TARGET_COL,
    TEST_PATH,
    load_params,
)


def evaluate_model():
    params = load_params()
    eval_params = params.get("evaluate", {})
    train_params = params.get("train", {})

    print(f"Loading test data from {TEST_PATH}...")
    test_df = pd.read_csv(TEST_PATH)

    X_test = test_df[FEATURE_COLS]
    y_test = test_df[TARGET_COL]

    print(f"Loading trained model from {MODEL_PATH}...")
    model = joblib.load(MODEL_PATH)

    # Predict and reverse target log-transform if applied during training
    raw_preds = model.predict(X_test)
    if train_params.get("log_target", True):
        preds = np.expm1(raw_preds)
    else:
        preds = raw_preds

    # Ensure non-negative count predictions
    preds = np.clip(preds, 0, None)

    # Calculate Regression Metrics
    mae = float(mean_absolute_error(y_test, preds))
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    r2 = float(r2_score(y_test, preds))

    metrics = {
        "test_mae": mae,
        "test_rmse": rmse,
        "test_r2": r2,
    }

    print("\n--- Evaluation Metrics ---")
    print(f"MAE:  {mae:.2f}")
    print(f"RMSE: {rmse:.2f}")
    print(f"R²:   {r2:.4f}\n")

    # Save metrics JSON
    metrics_path = PROJECT_ROOT / "metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)
    print(f"Saved metrics to {metrics_path}")

    # Generate Reports/Figures
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    # 1. Actual vs Predicted Plot
    plt.figure(figsize=(12, 5))
    plt.plot(y_test.values[:300], label="Actual Demand", alpha=0.8)
    plt.plot(preds[:300], label="Predicted Demand", alpha=0.8)
    plt.title("Bike Sharing Demand: Actual vs Predicted (First 300 Test Hours)")
    plt.xlabel("Hours")
    plt.ylabel("Bike Rentals Count")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "actual_vs_predicted.png")
    plt.close()

    # 2. Permutation Feature Importance
    print("Computing feature importances...")
    n_sample = min(eval_params.get("importance_sample", 2000), len(X_test))
    perm_importance = permutation_importance(
        model,
        X_test.iloc[:n_sample],
        y_test.iloc[:n_sample],
        random_state=eval_params.get("seed", 42),
    )

    plt.figure(figsize=(10, 6))
    sorted_idx = perm_importance.importances_mean.argsort()
    plt.barh(np.array(FEATURE_COLS)[sorted_idx], perm_importance.importances_mean[sorted_idx])
    plt.title("Permutation Feature Importance")
    plt.xlabel("Mean Importance Decrease")
    plt.tight_layout()
    plt.savefig(figures_dir / "feature_importance.png")
    plt.close()

    print(f"Saved evaluation figures to {figures_dir}")


if __name__ == "__main__":
    evaluate_model()