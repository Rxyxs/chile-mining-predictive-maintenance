"""Tests sobre los artefactos ya entrenados en data/processed (requiere haber
corrido generate -> features -> train_survival_pipeline al menos una vez)."""

import pytest

from src.models.scoring import PROCESSED_DIR, FleetScorer

pytestmark = pytest.mark.skipif(
    not (PROCESSED_DIR / "models" / "rul_lightgbm.joblib").exists(),
    reason="Artefactos entrenados no disponibles: corre el pipeline completo primero.",
)


@pytest.fixture(scope="module")
def scorer():
    return FleetScorer()


def test_risk_level_thresholds_are_ordered(scorer):
    assert scorer.risk_level(0) == "CRITICO"
    assert scorer.risk_level(10_000) == "BAJO"
    levels_seen = {scorer.risk_level(h) for h in [10, 60, 130, 250]}
    assert levels_seen <= {"CRITICO", "ALTO", "MEDIO", "BAJO"}


def test_score_fleet_covers_all_scoreable_equipment(scorer):
    fleet_scores = scorer.score_fleet()
    assert fleet_scores.height == scorer.latest_features.height
    assert set(fleet_scores.columns) >= {"equipment_id", "predicted_rul_hours", "risk_level"}


def test_score_equipment_known_id_returns_valid_score(scorer):
    known_id = scorer.latest_features["equipment_id"][0]
    score = scorer.score_equipment(known_id)
    assert score is not None
    assert score["risk_level"] in {"CRITICO", "ALTO", "MEDIO", "BAJO"}
    assert sum(score["failure_type_probabilities"].values()) == pytest.approx(1.0, abs=1e-2)


def test_score_equipment_unknown_id_returns_none(scorer):
    assert scorer.score_equipment("EQ-NOPE-9999") is None


def test_conditional_survival_probability_unknown_id_is_none(scorer):
    assert scorer.conditional_survival_probability("EQ-NOPE-9999", 1000.0) is None
