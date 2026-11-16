"""Create a small SYNTHETIC dataset with the same schema as train.csv.

Used only by GitHub Actions to check that train.py runs end to end, because
the Kaggle data must not be committed to GitHub. These rows are random and are
never used for the real model or for any reported score.

Usage: python data/make_ci_fixture.py data/ci_fixture.csv
"""
import sys

import numpy as np
import pandas as pd

rng = np.random.default_rng(0)
n = 2000
grades = np.array(list("ABCDEFG"))
g = rng.choice(7, n, p=[.35, .35, .18, .08, .02, .01, .01])
income = rng.lognormal(11, 0.5, n).round()
amount = rng.integers(500, 35001, n)
df = pd.DataFrame({
    "id": np.arange(n),
    "person_age": rng.integers(20, 70, n),
    "person_income": income,
    "person_home_ownership": rng.choice(["RENT", "OWN", "MORTGAGE", "OTHER"], n, p=[.5, .05, .44, .01]),
    "person_emp_length": rng.integers(0, 15, n).astype(float),
    "loan_intent": rng.choice(["EDUCATION", "MEDICAL", "PERSONAL", "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT"], n),
    "loan_grade": grades[g],
    "loan_amnt": amount,
    "loan_int_rate": (7 + 2 * g + rng.normal(0, 1, n)).round(2),
    "loan_percent_income": np.clip(amount / income, 0, 1).round(2),
    "cb_person_default_on_file": rng.choice(["N", "Y"], n, p=[.85, .15]),
    "cb_person_cred_hist_length": rng.integers(2, 20, n),
})
logit = -3 + 0.8 * g + 6 * df["loan_percent_income"] + (df["cb_person_default_on_file"] == "Y")
df["loan_status"] = (rng.random(n) < 1 / (1 + np.exp(-logit))).astype(int)
out = sys.argv[1] if len(sys.argv) > 1 else "data/ci_fixture.csv"
df.to_csv(out, index=False)
print(f"Wrote {len(df)} synthetic rows to {out} (positive rate {df.loan_status.mean():.2%})")
