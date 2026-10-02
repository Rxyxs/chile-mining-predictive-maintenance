import numpy as np
import pytest

from src.data.scania_loader import COST_MATRIX
from src.models import cost_decision as C


def test_certain_healthy_truck_gets_no_alarm():
    assert C.bayes_decision(np.array([[1.0, 0, 0, 0, 0]]))[0] == 0


def test_certain_failure_gets_the_matching_alarm():
    for k in (1, 2, 3, 4):
        proba = np.zeros((1, 5))
        proba[0, k] = 1.0
        assert C.bayes_decision(proba)[0] == k


def test_small_failure_probability_is_enough_to_alarm_because_misses_are_expensive():
    # P(falla)=10%: no alarmar cuesta 0.1*~350 = 35; alarmar cuesta ~0.9*7 = 6.3
    proba = np.array([[0.90, 0.0, 0.0, 0.0, 0.10]])
    assert C.bayes_decision(proba)[0] > 0
    assert proba.argmax() == 0  # argmax nunca habria alarmado


def test_decision_is_minimum_of_the_expected_cost():
    rng = np.random.default_rng(0)
    proba = rng.dirichlet(np.ones(5), size=50)
    expected = C.expected_cost_matrix(proba)
    decision = C.bayes_decision(proba)
    assert np.allclose(expected[np.arange(50), decision], expected.min(axis=1))


def test_expected_cost_matches_a_hand_computation():
    proba = np.array([[0.5, 0.5, 0, 0, 0]])
    expected = C.expected_cost_matrix(proba)
    # accion 0: 0.5*0 + 0.5*200 = 100 ; accion 1: 0.5*7 + 0.5*0 = 3.5
    assert expected[0, 0] == pytest.approx(100.0)
    assert expected[0, 1] == pytest.approx(3.5)


def test_confusion_and_cost_agree_with_the_direct_sum():
    y_true = [0, 0, 1, 2, 4, 3]
    y_pred = [0, 1, 0, 2, 3, 3]
    matrix = C.confusion(y_true, y_pred)
    assert matrix.sum() == len(y_true)
    direct = sum(COST_MATRIX[t][p] for t, p in zip(y_true, y_pred))
    assert C.cost_from_confusion(matrix) == direct


def test_alarm_metrics():
    out = C.alarm_metrics([0, 0, 0, 1, 2, 4], [0, 1, 0, 0, 3, 4])
    assert out["fallas_reales"] == 3
    assert out["fallas_alarmadas"] == 2
    assert out["recall_alarma"] == pytest.approx(2 / 3)
    assert out["falsas_alarmas"] == 1
    assert out["tasa_falsa_alarma"] == pytest.approx(1 / 3)


def test_alarm_metrics_without_failures_does_not_divide_by_zero():
    out = C.alarm_metrics([0, 0], [0, 1])
    assert np.isnan(out["recall_alarma"])
    assert out["falsas_alarmas"] == 1
