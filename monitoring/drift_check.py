"""Compare logged prediction inputs with the training data (data drift check).

Usage: python monitoring/drift_check.py --train data/raw/train.csv --log logs/predictions.jsonl

Prints mean and standard deviation of every numeric input (training vs logged), then
the Population Stability Index (PSI) per feature:
  PSI < 0.10 -> stable   0.10-0.25 -> moderate shift   > 0.25 -> significant drift
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from loan_features import CATEGORICAL_FEATURES, NUMERIC_FEATURES  # noqa: E402


def psi(expected: np.ndarray, actual: np.ndarray, eps: float = 1e-4) -> float:
    expected, actual = np.clip(expected, eps, None), np.clip(actual, eps, None)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--train", default="data/raw/train.csv")
    p.add_argument("--log", default="logs/predictions.jsonl")
    p.add_argument("--min_rows", type=int, default=500)
    a = p.parse_args()

    train = pd.read_csv(a.train)
    if not Path(a.log).exists():
        sys.exit(f"No prediction log at {a.log}. Make some predictions first.")
    live = pd.read_json(a.log, lines=True)
    print(f"Training rows: {len(train)} | logged predictions: {len(live)}")
    if len(live) < a.min_rows:
        print(f"Warning: fewer than {a.min_rows} logged rows, so PSI values are unreliable.\n")

    print(f"\n{'feature':28s} {'train mean':>12s} {'live mean':>12s} {'train std':>12s} {'live std':>12s}")
    for c in NUMERIC_FEATURES:
        print(f"{c:28s} {train[c].mean():12.2f} {live[c].mean():12.2f} {train[c].std():12.2f} {live[c].std():12.2f}")
    print()

    rows = []
    for c in NUMERIC_FEATURES:
        edges = np.unique(np.quantile(train[c], np.linspace(0, 1, 11)))
        edges[0], edges[-1] = -np.inf, np.inf
        e = np.histogram(train[c], edges)[0] / len(train)
        o = np.histogram(live[c], edges)[0] / len(live)
        rows.append((c, psi(e, o), f"train mean {train[c].mean():.2f} | live mean {live[c].mean():.2f}"))
    for c in CATEGORICAL_FEATURES:
        cats = sorted(set(train[c]) | set(live[c]))
        e = train[c].value_counts(normalize=True).reindex(cats, fill_value=0).values
        o = live[c].value_counts(normalize=True).reindex(cats, fill_value=0).values
        rows.append((c, psi(e, o), f"live top value: {live[c].mode().iat[0]}"))

    print(f"{'feature':28s} {'PSI':>7s}  status       detail")
    for name, value, detail in sorted(rows, key=lambda r: -r[1]):
        status = "DRIFT" if value > 0.25 else "moderate" if value > 0.10 else "stable"
        print(f"{name:28s} {value:7.3f}  {status:11s}  {detail}")
    print(f"\nMean predicted probability (live): {live['probability'].mean():.3f} | "
          f"training positive rate: {train['loan_status'].mean():.3f}")


if __name__ == "__main__":
    main()
