import numpy as np
import pandas as pd
from common import RAW_DATA_PATH, PROCESSED_DIR, TRAIN_PATH, TEST_PATH, load_params


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 1. Target Leakage Guard: Drop casual and registered
    drop_leakage = ["casual", "registered"]
    df = df.drop(columns=[col for col in drop_leakage if col in df.columns])

    # 2. Cyclical Transformations
    df["hr_sin"] = np.sin(2 * np.pi * df["hr"] / 24.0)
    df["hr_cos"] = np.cos(2 * np.pi * df["hr"] / 24.0)

    df["mnth_sin"] = np.sin(2 * np.pi * df["mnth"] / 12.0)
    df["mnth_cos"] = np.cos(2 * np.pi * df["mnth"] / 12.0)

    df["weekday_sin"] = np.sin(2 * np.pi * df["weekday"] / 7.0)
    df["weekday_cos"] = np.cos(2 * np.pi * df["weekday"] / 7.0)

    # 3. Rush Hour Indicator (7-9 AM & 5-7 PM on working days)
    df["is_rush_hour"] = (
        (df["workingday"] == 1) & 
        (df["hr"].isin([7, 8, 9, 17, 18, 19]))
    ).astype(int)

    return df


def prepare_data():
    params = load_params()
    test_ratio = params["prepare"]["test_ratio"]

    print(f"Reading raw data from {RAW_DATA_PATH}...")
    df = pd.read_csv(RAW_DATA_PATH)

    # Sort chronologically by date and hour
    df = df.sort_values(by=["dteday", "hr"]).reset_index(drop=True)

    # Apply Feature Engineering
    df_transformed = engineer_features(df)

    # Chronological Split
    split_idx = int(len(df_transformed) * (1 - test_ratio))
    train_df = df_transformed.iloc[:split_idx]
    test_df = df_transformed.iloc[split_idx:]

    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(TRAIN_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    print(f"Data successfully prepared:")
    print(f" - Train set: {len(train_df)} rows -> {TRAIN_PATH}")
    print(f" - Test set:  {len(test_df)} rows -> {TEST_PATH}")


if __name__ == "__main__":
    prepare_data()