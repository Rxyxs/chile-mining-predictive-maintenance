"""Carga de artefactos entrenados y scoring de riesgo de flota en memoria.

Logica compartida entre `src/api/main.py` (FastAPI) y `src/app/dashboard.py`
(Streamlit) para evitar duplicar la carga de modelos y el calculo de RUL /
probabilidad de falla / supervivencia condicional.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import polars as pl

from src.features.engineering import FEATURE_COLUMNS
from src.models.train_survival_pipeline import PROCESSED_DIR, build_survival_table, prepare_model_frame

MODELS_DIR = PROCESSED_DIR / "models"
SURVIVAL_HORIZON_HOURS = 720.0  # 30 dias de operacion continua

# El RUL ahora se entrena sobre el ciclo de vida completo (ver
# mining_data_generator.build_sensor_telemetry), por lo que estos umbrales
# son horizontes de negocio reales -- no fracciones de una ventana acotada.
RISK_THRESHOLDS_HOURS = {
    "CRITICO": 168.0,  # < 1 semana
    "ALTO": 720.0,  # < 1 mes
    "MEDIO": 2160.0,  # < 3 meses
}


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Corre primero: "
            "python -m src.data.mining_data_generator && "
            "python -m src.features.engineering && "
            "python -m src.models.train_survival_pipeline"
        )
    return path


class FleetScorer:
    """Carga los artefactos entrenados una vez y sirve scoring de riesgo en memoria."""

    def __init__(self) -> None:
        self.equipment_metadata = pl.read_parquet(_require(PROCESSED_DIR / "equipment_metadata.parquet"))
        telemetry_features = pl.read_parquet(_require(PROCESSED_DIR / "telemetry_features.parquet"))

        self.rul_model = joblib.load(_require(MODELS_DIR / "rul_lightgbm.joblib"))
        self.failure_classifier = joblib.load(_require(MODELS_DIR / "failure_classifier_lightgbm.joblib"))
        self.coxph_model = joblib.load(_require(MODELS_DIR / "coxph_model.joblib"))

        meta_for_join = self.equipment_metadata.with_columns(
            ((pl.col("reference_date") - pl.col("install_date")).dt.total_days() / 365.25).alias("age_years")
        ).select(["equipment_id", "equipment_type", "faena", "age_years"])

        self.latest_features = (
            telemetry_features.sort(["equipment_id", "timestamp"])
            .group_by("equipment_id", maintain_order=True)
            .last()
            .join(meta_for_join, on="equipment_id", how="inner")
            .drop_nulls(subset=FEATURE_COLUMNS)
        )
        self.survival_covariates = build_survival_table(self.equipment_metadata).drop(["duration", "event"])

    def risk_level(self, rul_hours: float) -> str:
        if rul_hours < RISK_THRESHOLDS_HOURS["CRITICO"]:
            return "CRITICO"
        if rul_hours < RISK_THRESHOLDS_HOURS["ALTO"]:
            return "ALTO"
        if rul_hours < RISK_THRESHOLDS_HOURS["MEDIO"]:
            return "MEDIO"
        return "BAJO"

    def conditional_survival_probability(self, equipment_id: str, current_hours: float) -> float | None:
        row = self.survival_covariates.filter(pl.col("equipment_id") == equipment_id).drop("equipment_id")
        if row.height == 0:
            return None

        covariates = row.to_pandas()
        horizon = current_hours + SURVIVAL_HORIZON_HOURS
        survival_fn = self.coxph_model.predict_survival_function(covariates, times=[current_hours, horizon])
        s_now, s_horizon = survival_fn.iloc[0, 0], survival_fn.iloc[1, 0]
        if s_now <= 0:
            return 0.0
        return float(max(0.0, min(1.0, s_horizon / s_now)))

    def score_row(self, row: dict, equipment_id: str, timestamp: str) -> dict:
        X = prepare_model_frame(pl.DataFrame([row]))

        rul_pred = float(self.rul_model.predict(X)[0])
        failure_pred = str(self.failure_classifier.predict(X)[0])
        proba = self.failure_classifier.predict_proba(X)[0]
        proba_map = {cls: round(float(p), 4) for cls, p in zip(self.failure_classifier.classes_, proba)}
        survival_prob = self.conditional_survival_probability(equipment_id, row["operating_hours"])

        return {
            "equipment_id": equipment_id,
            "equipment_type": row["equipment_type"],
            "faena": row["faena"],
            "last_reading_timestamp": timestamp,
            "operating_hours": round(row["operating_hours"], 2),
            "predicted_rul_hours": round(rul_pred, 1),
            "risk_level": self.risk_level(rul_pred),
            "predicted_failure_type": failure_pred,
            "failure_type_probabilities": proba_map,
            "survival_probability_30d": round(survival_prob, 4) if survival_prob is not None else None,
        }

    def score_equipment(self, equipment_id: str) -> dict | None:
        match = self.latest_features.filter(pl.col("equipment_id") == equipment_id)
        if match.height == 0:
            return None
        row = match.row(0, named=True)
        return self.score_row(row, equipment_id, str(row["timestamp"]))

    def score_fleet(self) -> pl.DataFrame:
        """Score de riesgo para la ultima lectura conocida de cada equipo de la flota."""
        X = prepare_model_frame(self.latest_features)
        rul_preds = self.rul_model.predict(X)
        failure_preds = self.failure_classifier.predict(X)
        risk_levels = [self.risk_level(r) for r in rul_preds]

        return self.latest_features.select(
            ["equipment_id", "equipment_type", "faena", "timestamp", "operating_hours"]
        ).with_columns(
            [
                pl.Series("predicted_rul_hours", rul_preds).round(1),
                pl.Series("predicted_failure_type", failure_preds),
                pl.Series("risk_level", risk_levels),
            ]
        )
