"""Download the Kaggle competition data into data/raw/.

Usage:  python data/download_data.py

Authentication is handled by the Kaggle CLI, never by this file:
  * run `kaggle auth login` once (browser login), or
  * set the KAGGLE_API_TOKEN environment variable (used by GitHub Actions secrets).
You must also click "Join Competition" / accept the rules on the competition page.
"""
from __future__ import annotations

import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pandas as pd

COMPETITION = "loan-approval-prediction-ulima"
RAW_DIR = Path(__file__).resolve().parent / "raw"
EXPECTED = {"train.csv": (58645, 13), "test.csv": (39098, 12), "sample_submission.csv": (39098, 2)}


def download() -> None:
    if shutil.which("kaggle") is None:
        sys.exit("Kaggle CLI not found. Activate the venv and run: pip install -r requirements.txt")
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    cmd = ["kaggle", "competitions", "download", "-c", COMPETITION, "-p", str(RAW_DIR)]
    print("Running:", " ".join(cmd))
    result = subprocess.run(cmd)
    if result.returncode != 0:
        sys.exit("Download failed. Check that you are logged in (kaggle auth login) "
                 "and have accepted the competition rules on Kaggle.")
    for zip_path in RAW_DIR.glob("*.zip"):
        with zipfile.ZipFile(zip_path) as z:
            z.extractall(RAW_DIR)
        zip_path.unlink()


def verify() -> None:
    for name, shape in EXPECTED.items():
        path = RAW_DIR / name
        if not path.exists():
            sys.exit(f"Missing {path}")
        actual = pd.read_csv(path).shape
        status = "OK" if actual == shape else f"UNEXPECTED (expected {shape})"
        print(f"{name:24s} {actual}  {status}")


if __name__ == "__main__":
    if not (RAW_DIR / "train.csv").exists() or "--force" in sys.argv:
        download()
    else:
        print("Files already present in", RAW_DIR, "(use --force to re-download)")
    verify()
