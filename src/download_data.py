import io
import urllib.request
import zipfile
from pathlib import Path

UCI_URL = "https://archive.ics.uci.edu/ml/machine-learning-databases/00275/Bike-Sharing-Dataset.zip"
RAW_DATA_DIR = Path("data/raw")
TARGET_FILE = RAW_DATA_DIR / "hour.csv"


def download_dataset():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)

    if TARGET_FILE.exists():
        print(f"Dataset already exists at: {TARGET_FILE}")
        return

    print(f"Downloading dataset from {UCI_URL}...")
    req = urllib.request.Request(
        UCI_URL,
        headers={"User-Agent": "Mozilla/5.0"},
    )

    with urllib.request.urlopen(req) as response:
        zip_buffer = io.BytesIO(response.read())

    with zipfile.ZipFile(zip_buffer) as zf:
        zf.extract("hour.csv", path=RAW_DATA_DIR)

    print(f"Successfully saved raw dataset to {TARGET_FILE}")


if __name__ == "__main__":
    download_dataset()