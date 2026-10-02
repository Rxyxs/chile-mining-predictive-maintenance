import polars as pl
import pytest

from src.data import scania_loader as S


def test_class_windows_follow_the_official_definition():
    # clase 4: 0-6 | clase 3: 6-12 | clase 2: 12-24 | clase 1: 24-48 | clase 0: mas de 48
    assert [S.class_from_time_to_failure(t) for t in (0, 6, 6.1, 12, 12.1, 24, 24.1, 48, 48.1, 500)] == [
        4, 4, 3, 3, 2, 2, 1, 1, 0, 0,
    ]


def test_label_train_readouts_marks_censored_vehicles_as_class_zero():
    readouts = pl.DataFrame({"vehicle_id": [1, 1, 1, 2, 2], "time_step": [10.0, 90.0, 99.0, 10.0, 50.0]})
    tte = pl.DataFrame({"vehicle_id": [1, 2], "duration": [100.0, 400.0], "event": [1, 0]})
    out = S.label_train_readouts(readouts, tte).sort(["vehicle_id", "time_step"])
    assert out["class_label"].to_list() == [0, 3, 4, 0, 0]  # tiempo restante 90, 10, 1
    assert out["time_to_failure"].to_list() == [90.0, 10.0, 1.0, None, None]


def test_label_boundaries_inside_a_vehicle():
    readouts = pl.DataFrame({"vehicle_id": [1] * 5, "time_step": [50.0, 70.0, 82.0, 90.0, 96.0]})
    tte = pl.DataFrame({"vehicle_id": [1], "duration": [100.0], "event": [1]})
    out = S.label_train_readouts(readouts, tte)
    # tiempo restante: 50, 30, 18, 10, 4  ->  clases 0, 1, 2, 3, 4
    assert out["class_label"].to_list() == [0, 1, 2, 3, 4]


def test_total_cost_uses_the_official_matrix():
    assert S.total_cost([0, 0], [0, 0]) == 0
    assert S.total_cost([1], [0]) == 200  # falla no detectada
    assert S.total_cost([4], [0]) == 500
    assert S.total_cost([0], [4]) == 10  # revision innecesaria
    assert S.total_cost([0, 1, 4], [1, 0, 3]) == 7 + 200 + 200


def test_undetected_failures_cost_far_more_than_false_alarms():
    cheapest_miss = min(S.COST_MATRIX[t][0] for t in (1, 2, 3, 4))
    dearest_false_alarm = max(S.COST_MATRIX[0][p] for p in (1, 2, 3, 4))
    assert cheapest_miss > 15 * dearest_false_alarm


def test_cost_matrix_is_zero_on_the_diagonal():
    assert all(S.COST_MATRIX[i][i] == 0 for i in range(5))


@pytest.mark.parametrize("name", S.FILES)
def test_expected_files_are_listed(name):
    assert name.endswith(".csv")
