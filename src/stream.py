import json
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from river import drift, metrics, forest, tree
from common import FEATURE_COLS, PROJECT_ROOT, TARGET_COL, TEST_PATH, load_params


def run_streaming():
    params = load_params().get("stream", {})
    adwin_delta = params.get("adwin_delta", 0.002)

    print(f"Loading test stream data from {TEST_PATH}...")
    test_df = pd.read_csv(TEST_PATH)

    # Instantiate River online regressor (handles updated River package structure)
    if hasattr(forest, "ARFRegressor"):
        model = forest.ARFRegressor(
            n_models=params.get("n_models", 10),
            seed=params.get("seed", 42),
        )
    elif hasattr(forest, "AMAFRegressor"):
        model = forest.AMAFRegressor(
            n_models=params.get("n_models", 10),
            seed=params.get("seed", 42),
        )
    else:
        model = tree.HoeffdingTreeRegressor()

    drift_detector = drift.ADWIN(delta=adwin_delta)

    online_mae = metrics.MAE()
    online_rmse = metrics.RMSE()

    errors = []
    drift_points = []

    print("Executing streaming simulation (row-by-row replay)...")
    for i, row in test_df.iterrows():
        x = row[FEATURE_COLS].to_dict()
        y = float(row[TARGET_COL])

        # 1. Predict on current hour instance
        y_pred = model.predict_one(x)
        if y_pred is None:
            y_pred = 0.0
        y_pred = max(0.0, float(y_pred))

        # 2. Update metrics
        online_mae.update(y, y_pred)
        online_rmse.update(y, y_pred)

        # 3. Calculate absolute error and pass to ADWIN drift detector
        abs_error = abs(y - y_pred)
        errors.append(abs_error)
        drift_detector.update(abs_error)

        # 4. Check for detected concept drift
        if drift_detector.drift_detected:
            print(f"-> Concept drift detected at hour step {i}! Absolute error spike: {abs_error:.2f}")
            drift_points.append(i)

        # 5. Incremental online model learning
        model.learn_one(x, y)

    stream_metrics = {
        "stream_mae": float(online_mae.get()),
        "stream_rmse": float(online_rmse.get()),
        "drift_events_count": len(drift_points),
        "drift_step_indices": drift_points,
    }

    print("\n--- Online Streaming Performance ---")
    print(f"Streaming MAE:  {stream_metrics['stream_mae']:.2f}")
    print(f"Streaming RMSE: {stream_metrics['stream_rmse']:.2f}")
    print(f"Drift Events:   {stream_metrics['drift_events_count']}\n")

    # Save metrics JSON
    metrics_path = PROJECT_ROOT / "stream_metrics.json"
    with open(metrics_path, "w") as f:
        json.dump(stream_metrics, f, indent=4)
    print(f"Saved stream metrics to {metrics_path}")

    # Plot Streaming Error and Drift Events
    figures_dir = PROJECT_ROOT / "reports" / "figures"
    figures_dir.mkdir(parents=True, exist_ok=True)

    plt.figure(figsize=(12, 5))
    window_size = params.get("rolling_window", 168)  # 1-week rolling average
    rolling_errors = pd.Series(errors).rolling(window=window_size, min_periods=1).mean()

    plt.plot(rolling_errors, label=f"Absolute Error ({window_size}h Rolling Mean)", color="tab:blue")
    
    if drift_points:
        plt.vlines(
            x=drift_points,
            ymin=0,
            ymax=max(errors),
            colors="red",
            linestyles="dashed",
            label="ADWIN Drift Detected",
        )

    plt.title("Online Streaming Model: Error Trajectory & Concept Drift Events")
    plt.xlabel("Streaming Step (Hours)")
    plt.ylabel("Absolute Error")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "stream_drift.png")
    plt.close()

    print(f"Saved streaming drift plot to {figures_dir / 'stream_drift.png'}")


if __name__ == "__main__":
    run_streaming()