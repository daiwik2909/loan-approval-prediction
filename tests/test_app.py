"""API tests. Run with:  pytest -v --tb=short"""
import sys
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from app import app  # noqa: E402

VALID = {
    "person_age": 28, "person_income": 60000, "person_home_ownership": "RENT",
    "person_emp_length": 4.0, "loan_intent": "EDUCATION", "loan_grade": "B",
    "loan_amnt": 9000, "loan_int_rate": 11.0, "loan_percent_income": 0.15,
    "cb_person_default_on_file": "N", "cb_person_cred_hist_length": 5,
}

# Profiles built from the EDA: grade A, owner, no default, low loan-to-income
# had ~1-5% positive rates; grade G, renter, prior default, loan > 40% of
# income had ~60-80%. We only assert the ORDER, never an exact value.
LOW_RISK = {**VALID, "person_income": 120000, "person_home_ownership": "OWN", "person_emp_length": 8.0,
            "loan_grade": "A", "loan_amnt": 6000, "loan_int_rate": 7.0, "loan_percent_income": 0.05}
HIGH_RISK = {**VALID, "person_income": 20000, "person_home_ownership": "RENT", "person_emp_length": 0.0,
             "loan_grade": "G", "loan_amnt": 10000, "loan_int_rate": 20.5, "loan_percent_income": 0.5,
             "cb_person_default_on_file": "Y"}


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    log_file = tmp_path_factory.mktemp("logs") / "predictions.jsonl"
    mp = pytest.MonkeyPatch()
    mp.setenv("PREDICTION_LOG_PATH", str(log_file))
    with TestClient(app) as c:  # "with" runs the startup that loads the model
        c.log_file = log_file
        yield c
    mp.undo()


def test_health(client):
    r = client.get("/health")
    assert r.status_code == 200
    body = r.json()
    assert body["status"] == "ok" and body["model_loaded"] is True


def test_homepage_loads(client):
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "Loan" in r.text


def test_valid_prediction_format(client):
    r = client.post("/predict", json=VALID)
    assert r.status_code == 200
    body = r.json()
    assert set(body) == {"loan_status", "probability", "label", "threshold", "disclaimer"}
    assert body["loan_status"] in (0, 1)
    assert 0.0 <= body["probability"] <= 1.0
    assert body["loan_status"] == int(body["probability"] >= body["threshold"])


def test_domain_logic_high_risk_scores_above_low_risk(client):
    low = client.post("/predict", json=LOW_RISK).json()["probability"]
    high = client.post("/predict", json=HIGH_RISK).json()["probability"]
    assert high > low + 0.3, f"expected a clear gap, got low={low}, high={high}"


def test_prediction_is_deterministic(client):
    a = client.post("/predict", json=VALID).json()["probability"]
    b = client.post("/predict", json=VALID).json()["probability"]
    assert a == b


@pytest.mark.parametrize("field,value", [
    ("person_age", 17), ("person_age", 101),
    ("loan_amnt", 100), ("loan_amnt", 50000),
    ("loan_int_rate", 30.0), ("loan_percent_income", 1.5),
    ("person_income", -5),
])
def test_out_of_range_returns_422(client, field, value):
    r = client.post("/predict", json={**VALID, field: value})
    assert r.status_code == 422


def test_invalid_category_returns_422(client):
    r = client.post("/predict", json={**VALID, "loan_grade": "Z"})
    assert r.status_code == 422


def test_missing_field_returns_422(client):
    payload = {k: v for k, v in VALID.items() if k != "loan_int_rate"}
    assert client.post("/predict", json=payload).status_code == 422


def test_null_and_wrong_type_return_422(client):
    assert client.post("/predict", json={**VALID, "person_income": None}).status_code == 422
    assert client.post("/predict", json={**VALID, "person_age": "twenty"}).status_code == 422


def test_inconsistent_employment_returns_422(client):
    r = client.post("/predict", json={**VALID, "person_age": 22, "person_emp_length": 15})
    assert r.status_code == 422


def test_unknown_field_returns_422(client):
    assert client.post("/predict", json={**VALID, "customer_name": "x"}).status_code == 422


def test_prediction_is_logged(client):
    before = client.log_file.read_text().count("\n") if client.log_file.exists() else 0
    client.post("/predict", json=VALID)
    lines = client.log_file.read_text().splitlines()
    assert len(lines) == before + 1
    assert '"loan_grade": "B"' in lines[-1]
