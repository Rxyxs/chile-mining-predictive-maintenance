"""La capa de features se prueba con camiones sinteticos de contadores acumulados."""
import numpy as np
import polars as pl
import pytest

from src.features import engineering as E


def _readouts(n_vehicles: int = 3, n_reads: int = 30, seed: int = 0) -> pl.DataFrame:
    """Contadores acumulados (no decrecientes) y un histograma de 3 bins, uno por vehiculo."""
    rng = np.random.default_rng(seed)
    rows = []
    for v in range(n_vehicles):
        t = np.cumsum(rng.uniform(1, 5, n_reads))
        counter = np.cumsum(rng.uniform(0, 10, n_reads))
        bins = np.cumsum(rng.uniform(0, 3, (n_reads, 3)), axis=0)
        for i in range(n_reads):
            rows.append(
                {"vehicle_id": v, "time_step": float(t[i]), "100_0": counter[i],
                 "200_0": bins[i, 0], "200_1": bins[i, 1], "200_2": bins[i, 2]}
            )
    return pl.DataFrame(rows)


def test_variable_groups_orders_bins_numerically():
    groups = E.variable_groups(["vehicle_id", "time_step", "5_10", "5_2", "5_0", "7_0"])
    assert groups == {"5": ["5_0", "5_2", "5_10"], "7": ["7_0"]}


def test_feature_names_match_the_output_columns():
    r = _readouts()
    out = E.engineer_features(r)
    assert set(E.feature_names(r.columns)) == set(out.columns) - {"vehicle_id"}


def test_one_row_per_readout_and_sorted():
    r = _readouts()
    out = E.engineer_features(r.reverse())
    assert out.height == r.height
    assert out.select(["vehicle_id", "time_step"]).equals(r.select(["vehicle_id", "time_step"]))


def test_time_step_keeps_full_precision_for_joins():
    out = E.engineer_features(_readouts())
    assert out.schema["time_step"] == pl.Float64


def test_life_rate_is_the_counter_over_elapsed_time():
    r = _readouts(1, 10)
    out = E.engineer_features(r)
    expected = r["100_0"] / r["time_step"]
    assert out["life_rate_100_0"].to_numpy() == pytest.approx(expected.to_numpy(), rel=1e-5)


def test_window_rate_hand_computed():
    r = pl.DataFrame(
        {"vehicle_id": [1] * 4, "time_step": [10.0, 20.0, 30.0, 40.0], "9_0": [0.0, 100.0, 300.0, 600.0]}
    )
    out = E.engineer_features(r, window=2)
    # lectura 4: (600 - 100) / (40 - 20) = 25 por unidad de tiempo
    assert out["win_rate_9_0"][3] == pytest.approx(25.0)


def test_histogram_fractions_sum_to_one():
    out = E.engineer_features(_readouts())
    life = out.select([f"life_frac_200_{i}" for i in range(3)]).to_numpy()
    assert np.nansum(life, axis=1) == pytest.approx(np.ones(len(life)), abs=1e-4)


def test_features_use_only_the_past():
    """Calcular sobre la historia completa y sobre la historia truncada da lo mismo
    en cada lectura: ninguna feature mira hacia adelante."""
    r = _readouts(2, 30)
    full = E.engineer_features(r)
    cut = 17
    truncated = r.filter(pl.col("vehicle_id").is_in([0, 1])).group_by("vehicle_id", maintain_order=True).head(cut)
    part = E.engineer_features(truncated)
    joined = part.join(full, on=["vehicle_id", "time_step"], suffix="_full")
    for name in E.feature_names(r.columns):
        if name == "time_step":
            continue
        a = joined[name].fill_null(-999).to_numpy()
        b = joined[f"{name}_full"].fill_null(-999).to_numpy()
        assert a == pytest.approx(b, rel=1e-5, abs=1e-6), name


def test_a_vehicle_does_not_leak_into_another():
    r = _readouts(3, 20)
    alone = E.engineer_features(r.filter(pl.col("vehicle_id") == 1))
    together = E.engineer_features(r).filter(pl.col("vehicle_id") == 1)
    for name in ("life_rate_100_0", "win_rate_100_0", "recent_frac_200_1"):
        assert alone[name].fill_null(-1).to_numpy() == pytest.approx(together[name].fill_null(-1).to_numpy())


def test_null_counters_are_filled_forward_within_the_vehicle():
    r = pl.DataFrame(
        {"vehicle_id": [1, 1, 1, 2, 2], "time_step": [1.0, 2.0, 3.0, 1.0, 2.0],
         "9_0": [10.0, None, 30.0, None, 5.0]}
    )
    out = E.engineer_features(r)
    # el nulo del vehiculo 1 toma 10; el primer valor del vehiculo 2 sigue nulo (no hereda del 1)
    assert out["life_rate_9_0"][1] == pytest.approx(10.0 / 2.0)
    assert out["life_rate_9_0"][3] is None


def test_last_readout_per_vehicle():
    out = E.engineer_features(_readouts(3, 12))
    last = E.last_readout_per_vehicle(out)
    assert last.height == 3
    assert last["n_readouts"].to_list() == [12, 12, 12]
