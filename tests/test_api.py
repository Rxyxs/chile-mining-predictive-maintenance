import pytest
from fastapi.testclient import TestClient

from src.api import main
from tests.test_scorer import SPECS, make_readouts, scorer  # noqa: F401  (fixture reutilizado)

HEADERS = {"X-API-Key": main.API_KEY}


@pytest.fixture()
def client(scorer, monkeypatch):  # noqa: F811
    monkeypatch.setattr(main, "_scorer", scorer)
    main.limiter.reset()
    return TestClient(main.app)


def payload(n=10):
    return {"readouts": make_readouts(n), "specs": SPECS}


def test_health_needs_no_key(client):
    assert client.get("/health").json() == {"status": "ok"}


def test_score_requires_an_api_key(client):
    assert client.post("/score", json=payload()).status_code == 401


def test_score_rejects_a_wrong_key(client):
    assert client.post("/score", json=payload(), headers={"X-API-Key": "nope"}).status_code == 401


def test_score_returns_the_recommendation(client):
    r = client.post("/score", json=payload(), headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert set(body["probabilidad_por_clase"]) == {"0", "1", "2", "3", "4"}
    assert 0 <= body["clase_recomendada"] <= 4
    assert isinstance(body["accion_recomendada"], str)


def test_invalid_history_is_a_422_with_a_reason(client):
    bad = payload(5)
    bad["readouts"][2]["time_step"] = bad["readouts"][0]["time_step"]
    r = client.post("/score", json=bad, headers=HEADERS)
    assert r.status_code == 422
    assert "creciente" in r.json()["detail"]


def test_missing_columns_are_a_422(client):
    bad = payload(5)
    for row in bad["readouts"]:
        row.pop("100_0")
    assert client.post("/score", json=bad, headers=HEADERS).status_code == 422


def test_empty_readouts_fail_validation(client):
    assert client.post("/score", json={"readouts": [], "specs": SPECS}, headers=HEADERS).status_code == 422


def test_untrained_model_is_a_503(monkeypatch):
    monkeypatch.setattr(main, "_scorer", None)

    def boom():
        raise FileNotFoundError

    monkeypatch.setattr(main.RiskScorer, "load", staticmethod(boom))
    main.limiter.reset()
    r = TestClient(main.app).post("/score", json=payload(), headers=HEADERS)
    assert r.status_code == 503
    assert "train_pipeline" in r.json()["detail"]


def test_rate_limit_kicks_in(client, monkeypatch):
    main.limiter.reset()
    codes = [client.get("/health").status_code for _ in range(70)]
    assert 429 in codes
