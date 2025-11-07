import pytest
from fastapi.testclient import TestClient
from src.api.main import app

client = TestClient(app)

def test_health():
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json() == {"status": "ok"}

def test_predict_valid():
    payload = {
        "company": "Maruti",
        "model": "Swift",
        "fuel": "Petrol",
        "seller_type": "Individual",
        "transmission": "Manual",
        "km_driven": 50000,
        "mileage_mpg": 45.0,
        "engine_cc": 1200.0,
        "max_power_bhp": 80.0,
        "seats": 5
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200
    assert "predicted_price_mad" in response.json()
