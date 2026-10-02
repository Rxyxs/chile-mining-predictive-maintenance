"""El scorer y la API se prueban con un modelo pequeño entrenado sobre camiones sinteticos,
asi no dependen de los 1,6 GB de datos reales ni de un entrenamiento previo."""
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import pytest

from src.features import engineering as E
from src.models.scorer import ACTIONS, SPEC_COLUMNS, InvalidReadouts, RiskScorer

READOUT_COLUMNS = ["time_step", "100_0", "200_0", "200_1", "200_2"]
CATEGORIES = {c: ["Cat0", "Cat1"] for c in SPEC_COLUMNS}
SPECS = {c: "Cat0" for c in SPEC_COLUMNS}


def make_readouts(n_reads: int = 25, rate: float = 5.0, seed: int = 0) -> list[dict]:
    rng = np.random.default_rng(seed)
    t = np.cumsum(rng.uniform(1, 4, n_reads))
    counter = np.cumsum(rng.uniform(0, rate, n_reads))
    bins = np.cumsum(rng.uniform(0, 3, (n_reads, 3)), axis=0)
    return [
        {"time_step": float(t[i]), "100_0": float(counter[i]), "200_0": float(bins[i, 0]),
         "200_1": float(bins[i, 1]), "200_2": float(bins[i, 2])}
        for i in range(n_reads)
    ]


@pytest.fixture(scope="module")
def scorer() -> RiskScorer:
    frames = []
    for v in range(60):
        r = pl.DataFrame(make_readouts(25, rate=2.0 + 0.2 * v, seed=v)).with_columns(pl.lit(v).alias("vehicle_id"))
        frames.append(r.select(["vehicle_id", *READOUT_COLUMNS]))
    feats = E.engineer_features(pl.concat(frames)).to_pandas()
    # etiqueta de juguete: los camiones de mayor uso estan en riesgo; clase 0 abunda
    labels = np.where(feats["vehicle_id"] >= 50, 4, 0)
    labels[feats["vehicle_id"].between(45, 49)] = 2
    cols = [c for c in feats.columns if c != "vehicle_id"]
    model = lgb.LGBMClassifier(objective="multiclass", num_class=5, n_estimators=30, verbosity=-1, min_child_samples=5)
    # el modelo debe conocer las 5 clases: se agrega una fila por clase ausente
    X = feats[cols]
    for k in (1, 3):
        X = pd.concat([X, X.iloc[[0]]], ignore_index=True)
        labels = np.append(labels, k)
    model.fit(X, labels)
    full_cols = cols + SPEC_COLUMNS
    return RiskScorer(_WithSpecs(model, cols), full_cols, CATEGORIES, READOUT_COLUMNS)


class _WithSpecs:
    """Envuelve el modelo de juguete (entrenado sin especificaciones) para ignorar las columnas Spec."""

    def __init__(self, model, cols):
        self.model, self.cols = model, cols

    def predict_proba(self, X):
        return self.model.predict_proba(X[self.cols])


def test_score_returns_a_valid_probability_vector(scorer):
    out = scorer.score(make_readouts(), SPECS)
    probs = np.array(list(out["probabilidad_por_clase"].values()))
    assert len(probs) == 5
    assert probs.sum() == pytest.approx(1.0, abs=1e-4)
    assert 0 <= out["probabilidad_algun_riesgo"] <= 1
    assert out["probabilidad_algun_riesgo"] == pytest.approx(1 - probs[0], abs=1e-4)


def test_recommended_action_is_the_cheapest_in_expectation(scorer):
    out = scorer.score(make_readouts(), SPECS)
    costs = out["costo_esperado_por_accion"]
    best = min(costs, key=costs.get)
    assert out["clase_recomendada"] == int(best)
    assert out["accion_recomendada"] == ACTIONS[out["clase_recomendada"]]
    assert out["costo_esperado_accion_recomendada"] <= out["costo_esperado_no_hacer_nada"] + 1e-9


def test_score_uses_only_the_history_it_receives(scorer):
    history = make_readouts(25)
    full = scorer.score(history, SPECS)
    truncated = scorer.score(history[:20], SPECS)
    assert full["n_lecturas"] == 25 and truncated["n_lecturas"] == 20


def test_single_readout_is_scored_without_error(scorer):
    out = scorer.score(make_readouts(1), SPECS)
    assert out["n_lecturas"] == 1


def test_missing_readout_columns_are_rejected(scorer):
    bad = [{k: v for k, v in r.items() if k != "100_0"} for r in make_readouts(5)]
    with pytest.raises(InvalidReadouts, match="Faltan"):
        scorer.score(bad, SPECS)


def test_time_step_must_increase(scorer):
    readouts = make_readouts(5)
    readouts[3]["time_step"] = readouts[1]["time_step"]
    with pytest.raises(InvalidReadouts, match="creciente"):
        scorer.score(readouts, SPECS)


def test_empty_history_is_rejected(scorer):
    with pytest.raises(InvalidReadouts):
        scorer.score([], SPECS)


def test_missing_specs_are_rejected(scorer):
    with pytest.raises(InvalidReadouts, match="especificaciones"):
        scorer.score(make_readouts(5), {"Spec_0": "Cat0"})


def test_unknown_spec_category_does_not_crash(scorer):
    out = scorer.score(make_readouts(5), {**SPECS, "Spec_3": "CategoriaNueva"})
    assert "clase_recomendada" in out
