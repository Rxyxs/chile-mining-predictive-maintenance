"""Scoring de un camion a partir de su historial de lecturas.

Recibe las lecturas operacionales acumuladas (contadores e histogramas) y las
especificaciones del vehiculo, recalcula las features con SOLO ese historial y entrega:
probabilidad por clase de riesgo, costo esperado de cada accion y la accion recomendada.
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import pandas as pd
import polars as pl

from src.features import engineering as E
from src.models import cost_decision as C

MODELS_DIR = Path(__file__).resolve().parents[2] / "data" / "processed" / "models"
SPEC_COLUMNS = [f"Spec_{i}" for i in range(8)]

ACTIONS = {
    0: "Sin accion: operacion normal.",
    1: "Programar revision en el proximo mantenimiento (falla probable en 24-48 unidades).",
    2: "Programar revision pronto (falla probable en 12-24 unidades).",
    3: "Revision urgente (falla probable en 6-12 unidades).",
    4: "Retirar de operacion y revisar ahora (falla probable en menos de 6 unidades).",
}


class InvalidReadouts(ValueError):
    """El historial enviado no sirve para calcular features."""


class RiskScorer:
    def __init__(self, model, feature_cols: list[str], spec_categories: dict[str, list[str]], readout_columns: list[str]):
        self.model = model
        self.feature_cols = feature_cols
        self.spec_categories = spec_categories
        self.readout_columns = readout_columns

    @classmethod
    def load(cls, models_dir: Path = MODELS_DIR) -> "RiskScorer":
        cols = json.loads((models_dir / "feature_columns.json").read_text(encoding="utf-8"))
        cats = json.loads((models_dir / "spec_categories.json").read_text(encoding="utf-8"))
        readout_cols = json.loads((models_dir / "readout_columns.json").read_text(encoding="utf-8"))
        return cls(joblib.load(models_dir / "risk_classifier.joblib"), cols, cats, readout_cols)

    def _validate(self, readouts: list[dict], specs: dict[str, str]) -> None:
        if not readouts:
            raise InvalidReadouts("Se necesita al menos una lectura.")
        missing = [c for c in self.readout_columns if c not in readouts[0]]
        if missing:
            raise InvalidReadouts(f"Faltan {len(missing)} columnas en las lecturas, por ejemplo: {missing[:5]}")
        steps = [r["time_step"] for r in readouts]
        if any(b <= a for a, b in zip(steps, steps[1:])):
            raise InvalidReadouts("time_step debe ser estrictamente creciente.")
        missing_specs = [c for c in SPEC_COLUMNS if c not in specs]
        if missing_specs:
            raise InvalidReadouts(f"Faltan especificaciones: {missing_specs}")

    def score(self, readouts: list[dict], specs: dict[str, str]) -> dict:
        self._validate(readouts, specs)
        df = pl.DataFrame([{**{k: r.get(k) for k in self.readout_columns}, "vehicle_id": 0} for r in readouts])
        df = df.select(["vehicle_id", *self.readout_columns]).cast({c: pl.Float64 for c in self.readout_columns})
        last = E.last_readout_per_vehicle(E.engineer_features(df)).to_pandas()
        for c in SPEC_COLUMNS:
            last[c] = pd.Categorical([specs[c]], categories=self.spec_categories[c])
        proba = self.model.predict_proba(last[self.feature_cols])
        expected = C.expected_cost_matrix(proba)[0]
        action = int(expected.argmin())
        return {
            "n_lecturas": len(readouts),
            "probabilidad_por_clase": {str(k): round(float(p), 6) for k, p in enumerate(proba[0])},
            "probabilidad_algun_riesgo": round(float(1.0 - proba[0, 0]), 6),
            "costo_esperado_por_accion": {str(k): round(float(v), 3) for k, v in enumerate(expected)},
            "clase_recomendada": action,
            "accion_recomendada": ACTIONS[action],
            "costo_esperado_no_hacer_nada": round(float(expected[0]), 3),
            "costo_esperado_accion_recomendada": round(float(expected[action]), 3),
        }

