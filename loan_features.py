"""Shared feature definitions and the feature-engineering transformer.

This module is imported by both train.py and app.py. The saved model
(model/model.joblib) references the LoanFeatureEngineer class, so this file
must be present wherever the model is loaded (including the Docker image).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin

TARGET = "loan_status"
ID_COL = "id"

NUMERIC_FEATURES = [
    "person_age",
    "person_income",
    "person_emp_length",
    "loan_amnt",
    "loan_int_rate",
    "loan_percent_income",
    "cb_person_cred_hist_length",
]
CATEGORICAL_FEATURES = [
    "person_home_ownership",
    "loan_intent",
    "loan_grade",
    "cb_person_default_on_file",
]
RAW_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES

# Allowed category values, taken from train.csv (test.csv has the same set).
HOME_OWNERSHIP = ["RENT", "OWN", "MORTGAGE", "OTHER"]
LOAN_INTENT = [
    "EDUCATION", "MEDICAL", "PERSONAL",
    "VENTURE", "DEBTCONSOLIDATION", "HOMEIMPROVEMENT",
]
LOAN_GRADE = ["A", "B", "C", "D", "E", "F", "G"]
DEFAULT_ON_FILE = ["N", "Y"]

GRADE_TO_NUM = {g: i + 1 for i, g in enumerate(LOAN_GRADE)}

ENGINEERED_NUMERIC = [
    "loan_to_income",
    "grade_num",
    "rate_vs_grade",
    "emp_to_age",
    "cred_to_age",
    "high_lti",
    "log_income",
]


class LoanFeatureEngineer(BaseEstimator, TransformerMixin):
    """Caps outliers and adds engineered features.

    Everything that is *learned* (cap values, mean interest rate per grade)
    is learned in fit() from the training rows only. Inside a Pipeline that
    means each CV fold learns from its own training fold, so nothing leaks
    from validation or test data.
    """

    def __init__(self, income_quantile: float = 0.995):
        self.income_quantile = income_quantile

    def fit(self, X: pd.DataFrame, y=None):
        X = pd.DataFrame(X)
        self.income_cap_ = float(X["person_income"].quantile(self.income_quantile))
        self.grade_rate_mean_ = X.groupby("loan_grade")["loan_int_rate"].mean().to_dict()
        self.global_rate_mean_ = float(X["loan_int_rate"].mean())
        return self

    def transform(self, X: pd.DataFrame) -> pd.DataFrame:
        X = pd.DataFrame(X).copy()
        # Cap impossible / extreme values (age 123, employment 123 years, income 1.9M).
        X["person_age"] = X["person_age"].clip(upper=80)
        X["person_emp_length"] = X["person_emp_length"].clip(upper=60)
        X["person_income"] = X["person_income"].clip(upper=self.income_cap_)

        income = X["person_income"].astype(float)
        age = X["person_age"].astype(float)
        X["loan_to_income"] = X["loan_amnt"] / income
        X["grade_num"] = X["loan_grade"].map(GRADE_TO_NUM).astype(float)
        expected_rate = X["loan_grade"].map(self.grade_rate_mean_).astype(float)
        X["rate_vs_grade"] = X["loan_int_rate"] - expected_rate.fillna(self.global_rate_mean_)
        X["emp_to_age"] = X["person_emp_length"] / age
        X["cred_to_age"] = X["cb_person_cred_hist_length"] / age
        X["high_lti"] = (X["loan_percent_income"] > 0.30).astype(float)
        X["log_income"] = np.log1p(income)
        return X

    def get_feature_names_out(self, input_features=None):
        return np.array(RAW_FEATURES + ENGINEERED_NUMERIC, dtype=object)
