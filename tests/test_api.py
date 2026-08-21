"""Tests de la API FastAPI (requiere artefactos entrenados en data/processed)."""

import os

import pytest

os.environ["MINING_API_KEY"] = "test-key"
os.environ["MINING_RATE_LIMIT"] = "1000/minute"  # evita interferencia con el limite de negocio en tests

from src.api.main import app  # noqa: E402
from src.models.scoring import PROCESSED_DIR  # noqa: E402

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "models" / "rul_lightgbm.joblib").exists(),
    reason="Artefactos entrenados no disponibles: corre el pipeline completo primero.",
)


@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient

    with TestClient(app) as c:
        yield c


def test_health_does_not_require_api_key(client):
    response = client.get("/health")
    assert response.status_code == 200
    assert response.json()["status"] == "ok"


def test_protected_endpoint_without_key_returns_401(client):
    response = client.get("/equipment")
    assert response.status_code == 401


def test_protected_endpoint_with_wrong_key_returns_401(client):
    response = client.get("/equipment", headers={"X-API-Key": "wrong-key"})
    assert response.status_code == 401


def test_protected_endpoint_with_correct_key_returns_200(client):
    response = client.get("/equipment", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert isinstance(response.json(), list)
    assert len(response.json()) > 0


def test_equipment_risk_requires_key(client):
    known_id = client.get("/equipment", headers={"X-API-Key": "test-key"}).json()[0]["equipment_id"]

    unauthorized = client.get(f"/equipment/{known_id}/risk")
    assert unauthorized.status_code == 401

    authorized = client.get(f"/equipment/{known_id}/risk", headers={"X-API-Key": "test-key"})
    assert authorized.status_code == 200
    assert authorized.json()["equipment_id"] == known_id


def test_fleet_risk_summary_requires_key(client):
    response = client.get("/fleet/risk-summary", headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    assert response.json()["n_equipment_scored"] > 0
