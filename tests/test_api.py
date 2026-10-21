import pytest
from fastapi.testclient import TestClient
from api.main import app, load_assets

@pytest.fixture(scope="session", autouse=True)
def init_api_assets():
    load_assets()

@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root_endpoint(client):
    response = client.get("/")
    assert response.status_code == 200
    data = response.json()
    assert "service" in data
    assert data["version"] == "2.0.0"


def test_health_endpoint(client):
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "model_loaded" in data


def test_predict_single_endpoint(client):
    # Default EmployeeInput values
    payload = {}
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert "probability" in data
    assert "prediction" in data
    assert "risk_tier" in data
    assert "financial_exposure" in data
    assert "expected_loss" in data["financial_exposure"]
    assert "replacement_cost" in data["financial_exposure"]
    assert data["probability"] >= 0.0 and data["probability"] <= 1.0


def test_predict_batch_endpoint(client):
    payload = {
        "records": [
            {"Age": 25, "OverTime": "Yes", "MonthlyIncome": 3000},
            {"Age": 45, "OverTime": "No", "MonthlyIncome": 12000}
        ]
    }
    response = client.post("/predict/batch", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["total_records"] == 2
    assert "cohort_financial_summary" in data
    assert "predictions" in data


def test_validate_endpoint(client):
    payload = {
        "Age": 150,  # invalid out-of-range
        "OverTime": "Maybe",  # invalid category
        "MonthlyIncome": -500  # invalid negative
    }
    response = client.post("/validate", json=payload)
    assert response.status_code == 200
    data = response.json()
    assert data["is_valid"] is False
    assert len(data["errors"]) >= 1
    assert "sanitized_record" in data


def test_analytics_kpis_endpoint(client):
    response = client.get("/analytics/kpis")
    assert response.status_code == 200
    data = response.json()
    assert "total_headcount" in data
    assert "attrition_rate_pct" in data
    assert data["total_headcount"] == 1470
