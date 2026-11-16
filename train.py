"""Train, evaluate and log a loan_status model.

Examples
--------
python train.py --model logistic_regression --C 1.0
python train.py --model random_forest --n_estimators 300 --max_depth 12
python train.py --model xgboost --n_estimators 600 --max_depth 5 --learning_rate 0.05

What one run does
1. Loads data/raw/train.csv.
2. Holds out a stratified 20% validation split.
3. Runs stratified 5-fold cross-validation (ROC-AUC) on the other 80%.
4. Fits on the 80%, scores the 20% validation split.
5. Refits on 100% of train.csv (the final model).
6. Logs params + metrics to MLflow (sqlite:///mlflow.db).
7. Writes a Kaggle submission CSV to submissions/.
8. Saves model/model.joblib if this run beats the saved model's CV ROC-AUC.
"""
from __future__ import annotations

import argparse
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, log_loss, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from loan_features import (CATEGORICAL_FEATURES, ENGINEERED_NUMERIC, ID_COL,
                           NUMERIC_FEATURES, RAW_FEATURES, TARGET, LoanFeatureEngineer)

MODELS = ["logistic_regression", "random_forest", "xgboost"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--model", choices=MODELS, default="logistic_regression")
    # Logistic regression
    p.add_argument("--C", type=float, default=1.0, help="Inverse regularisation strength (logistic regression)")
    # Trees
    p.add_argument("--n_estimators", type=int, default=300)
    p.add_argument("--max_depth", type=int, default=None)
    p.add_argument("--min_samples_leaf", type=int, default=1, help="Random forest only")
    p.add_argument("--learning_rate", type=float, default=0.05, help="XGBoost only")
    p.add_argument("--subsample", type=float, default=0.8, help="XGBoost only")
    p.add_argument("--colsample_bytree", type=float, default=0.8, help="XGBoost only")
    # Experiment setup
    p.add_argument("--cv_folds", type=int, default=5)
    p.add_argument("--val_size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--data", default="data/raw/train.csv")
    p.add_argument("--test", default="data/raw/test.csv", help="Kaggle test.csv for the submission file")
    p.add_argument("--output", default="model/model.joblib")
    p.add_argument("--save", choices=["auto", "always", "never"], default="auto",
                   help="auto = save only if CV ROC-AUC beats the currently saved model")
    p.add_argument("--no_mlflow", action="store_true", help="Skip MLflow logging (used in CI)")
    p.add_argument("--tracking_uri", default="sqlite:///mlflow.db")
    p.add_argument("--experiment", default="loan-approval-prediction")
    return p.parse_args()


def build_estimator(args: argparse.Namespace):
    if args.model == "logistic_regression":
        return LogisticRegression(C=args.C, max_iter=2000, random_state=args.seed)
    if args.model == "random_forest":
        return RandomForestClassifier(
            n_estimators=args.n_estimators, max_depth=args.max_depth,
            min_samples_leaf=args.min_samples_leaf, n_jobs=-1, random_state=args.seed)
    from xgboost import XGBClassifier  # imported lazily so LR/RF runs don't need it loaded
    return XGBClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth or 6,
        learning_rate=args.learning_rate, subsample=args.subsample,
        colsample_bytree=args.colsample_bytree, tree_method="hist",
        eval_metric="auc", n_jobs=-1, random_state=args.seed)


def build_pipeline(args: argparse.Namespace) -> Pipeline:
    numeric = NUMERIC_FEATURES + ENGINEERED_NUMERIC
    # Scaling matters for logistic regression; tree models don't need it.
    num_step = StandardScaler() if args.model == "logistic_regression" else "passthrough"
    preprocess = ColumnTransformer([
        ("num", num_step, numeric),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), CATEGORICAL_FEATURES),
    ])
    return Pipeline([
        ("features", LoanFeatureEngineer()),
        ("preprocess", preprocess),
        ("model", build_estimator(args)),
    ])


def model_params(args: argparse.Namespace) -> dict:
    common = {"model": args.model, "cv_folds": args.cv_folds, "val_size": args.val_size, "seed": args.seed}
    if args.model == "logistic_regression":
        return {**common, "C": args.C}
    if args.model == "random_forest":
        return {**common, "n_estimators": args.n_estimators, "max_depth": args.max_depth,
                "min_samples_leaf": args.min_samples_leaf}
    return {**common, "n_estimators": args.n_estimators, "max_depth": args.max_depth or 6,
            "learning_rate": args.learning_rate, "subsample": args.subsample,
            "colsample_bytree": args.colsample_bytree}


def main() -> None:
    args = parse_args()
    t0 = time.time()

    df = pd.read_csv(args.data)
    missing = [c for c in RAW_FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise SystemExit(f"Training data is missing columns: {missing}")
    X, y = df[RAW_FEATURES], df[TARGET].astype(int)

    X_tr, X_val, y_tr, y_val = train_test_split(
        X, y, test_size=args.val_size, stratify=y, random_state=args.seed)

    # 1) Cross-validation on the training split. Same folds for every model.
    cv = StratifiedKFold(n_splits=args.cv_folds, shuffle=True, random_state=args.seed)
    cv_scores = cross_val_score(build_pipeline(args), X_tr, y_tr, cv=cv, scoring="roc_auc", n_jobs=1)

    # 2) Fit on the training split, evaluate on the untouched validation split.
    pipe = build_pipeline(args).fit(X_tr, y_tr)
    p_val = pipe.predict_proba(X_val)[:, 1]
    y_hat = (p_val >= 0.5).astype(int)
    metrics = {
        "cv_roc_auc_mean": float(cv_scores.mean()),
        "cv_roc_auc_std": float(cv_scores.std()),
        "val_roc_auc": float(roc_auc_score(y_val, p_val)),
        "val_accuracy": float(accuracy_score(y_val, y_hat)),
        "val_precision": float(precision_score(y_val, y_hat, zero_division=0)),
        "val_recall": float(recall_score(y_val, y_hat)),
        "val_f1": float(f1_score(y_val, y_hat)),
        "val_log_loss": float(log_loss(y_val, p_val)),
    }

    # 3) Refit on all labelled rows for serving and the Kaggle submission.
    final = build_pipeline(args).fit(X, y)
    metrics["train_seconds"] = round(time.time() - t0, 1)
    params = model_params(args)

    print(f"\n== {args.model} ==")
    print(f"CV ROC-AUC   : {metrics['cv_roc_auc_mean']:.5f} +/- {metrics['cv_roc_auc_std']:.5f}")
    print(f"Val ROC-AUC  : {metrics['val_roc_auc']:.5f}")
    print(f"Val accuracy : {metrics['val_accuracy']:.4f} | precision {metrics['val_precision']:.4f} "
          f"| recall {metrics['val_recall']:.4f} | F1 {metrics['val_f1']:.4f}")

    run_name = f"{args.model}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

    # Kaggle submission: probability of loan_status = 1 for every test id.
    submission_path = None
    if args.test and Path(args.test).exists():
        test = pd.read_csv(args.test)
        sub = pd.DataFrame({ID_COL: test[ID_COL], TARGET: final.predict_proba(test[RAW_FEATURES])[:, 1]})
        Path("submissions").mkdir(exist_ok=True)
        submission_path = Path("submissions") / f"submission_{run_name}.csv"
        sub.to_csv(submission_path, index=False)
        print(f"Submission   : {submission_path}")

    # Save the model if it is the best so far.
    out = Path(args.output)
    meta_path = out.with_name("metadata.json")
    previous = json.loads(meta_path.read_text())["metrics"]["cv_roc_auc_mean"] if meta_path.exists() else -1
    should_save = args.save == "always" or (args.save == "auto" and metrics["cv_roc_auc_mean"] > previous)
    if should_save:
        out.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(final, out)
        meta = {
            "run_name": run_name, "params": params, "metrics": metrics,
            "features": RAW_FEATURES, "target": TARGET, "positive_class": 1, "threshold": 0.5,
            "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "python": platform.python_version(), "sklearn": sklearn.__version__,
            "training_rows": int(len(df)),
        }
        meta_path.write_text(json.dumps(meta, indent=2))
        print(f"Saved model  : {out} (previous best CV ROC-AUC {previous:.5f})")
    else:
        print(f"Not saved    : CV ROC-AUC did not beat saved model ({previous:.5f})")

    if not args.no_mlflow:
        import mlflow
        mlflow.set_tracking_uri(args.tracking_uri)
        mlflow.set_experiment(args.experiment)
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params(params)
            mlflow.log_metrics(metrics)
            mlflow.set_tags({"algorithm": args.model, "saved_as_best": str(should_save),
                             "training_rows": len(df), "features": len(RAW_FEATURES) + len(ENGINEERED_NUMERIC)})
            if submission_path:
                mlflow.set_tag("submission_file", str(submission_path))
        print(f"MLflow       : logged run '{run_name}' to {args.tracking_uri}")


if __name__ == "__main__":
    main()
