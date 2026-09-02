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

_MULTITASK_ARTIFACTS_READY = (PROCESSED_DIR / "models" / "multi_task_shap_background.pt").exists()

_RAW_TELEMETRY_PAYLOAD = {
    "equipment_type": "CAEX",
    "faena": "Escondida",
    "operating_hours": 12000.0,
    "age_years": 4.0,
    "engine_temp_roll_mean_short": 92.0,
    "engine_temp_roll_mean_med": 90.5,
    "engine_temp_roll_std_med": 2.5,
    "engine_temp_delta_long": 1.5,
    "vibration_roll_mean_short": 3.8,
    "vibration_roll_mean_med": 3.6,
    "vibration_roll_std_med": 0.4,
    "vibration_cum_var": 8.0,
    "vibration_fft_dominant_amp": 1.2,
    "vibration_fft_spectral_energy": 15.0,
    "hydraulic_pressure_roll_mean_med": 190.0,
    "hydraulic_pressure_delta_long": -1.0,
    "rpm_roll_std_med": 30.0,
    "fuel_consumption_roll_mean_med": 280.0,
}


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


@pytest.mark.skipif(
    not _MULTITASK_ARTIFACTS_READY,
    reason="Artefactos DeepSHAP no disponibles: corre `python multitask_pdm.py` primero.",
)
def test_multitask_score_requires_key_and_returns_prediction(client):
    unauthorized = client.post("/multitask/score", json=_RAW_TELEMETRY_PAYLOAD)
    assert unauthorized.status_code == 401

    response = client.post("/multitask/score", json=_RAW_TELEMETRY_PAYLOAD, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    body = response.json()
    assert body["predicted_rul_hours"] >= 0
    assert body["predicted_failure_type"] in body["failure_type_probabilities"]
    assert abs(sum(body["failure_type_probabilities"].values()) - 1.0) < 1e-3


@pytest.mark.skipif(
    not _MULTITASK_ARTIFACTS_READY,
    reason="Artefactos DeepSHAP no disponibles: corre `python multitask_pdm.py` primero.",
)
def test_multitask_explicar_returns_shap_for_both_heads(client):
    response = client.post("/multitask/explicar", json=_RAW_TELEMETRY_PAYLOAD, headers={"X-API-Key": "test-key"})
    assert response.status_code == 200
    body = response.json()
    expected_features = {
        "engine_temp_roll_mean_short", "engine_temp_roll_mean_med", "engine_temp_roll_std_med",
        "engine_temp_delta_long", "vibration_roll_mean_short", "vibration_roll_mean_med",
        "vibration_roll_std_med", "vibration_cum_var", "vibration_fft_dominant_amp",
        "vibration_fft_spectral_energy", "hydraulic_pressure_roll_mean_med", "hydraulic_pressure_delta_long",
        "rpm_roll_std_med", "fuel_consumption_roll_mean_med", "operating_hours", "age_years",
        "equipment_type", "faena",
    }
    assert set(body["shap_rul"]) == expected_features
    assert set(body["shap_failure_type"]) == expected_features
