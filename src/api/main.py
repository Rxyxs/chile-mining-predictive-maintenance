"""Servicio FastAPI de evaluacion de riesgo operacional de flota.

Expone el score de salud (RUL predicho, tipo de falla mas probable y
probabilidad de supervivencia a 30 dias) para camiones CAEX y chancadores
primarios, a partir de los modelos entrenados por
`src.models.train_survival_pipeline`.

Ejecutar desde la raiz del repositorio con:
    uvicorn src.api.main:app --reload
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Literal

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import polars as pl
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.data.mining_data_generator import EQUIPMENT_TYPES, FAENAS
from src.models.scoring import FleetScorer

EquipmentType = Literal[tuple(EQUIPMENT_TYPES)]  # type: ignore[misc]
Faena = Literal[tuple(FAENAS)]  # type: ignore[misc]

scorer = FleetScorer()


class RiskScoreResponse(BaseModel):
    equipment_id: str
    equipment_type: str
    faena: str
    last_reading_timestamp: str
    operating_hours: float
    predicted_rul_hours: float
    risk_level: str
    predicted_failure_type: str
    failure_type_probabilities: dict[str, float]
    survival_probability_30d: float | None


class RawTelemetryFeatures(BaseModel):
    """Payload para scoring en linea directa, sin necesidad de un equipment_id existente."""

    equipment_type: EquipmentType
    faena: Faena
    operating_hours: float = Field(ge=0)
    age_years: float = Field(ge=0)
    engine_temp_roll_mean_short: float
    engine_temp_roll_mean_med: float
    engine_temp_roll_std_med: float
    engine_temp_delta_long: float
    vibration_roll_mean_short: float
    vibration_roll_mean_med: float
    vibration_roll_std_med: float
    vibration_cum_var: float
    vibration_fft_dominant_amp: float
    vibration_fft_spectral_energy: float
    hydraulic_pressure_roll_mean_med: float
    hydraulic_pressure_delta_long: float
    rpm_roll_std_med: float
    fuel_consumption_roll_mean_med: float


app = FastAPI(
    title="Chile Mining Predictive Maintenance API",
    description=(
        "Score de salud de flota (RUL, probabilidad de falla y supervivencia) "
        "para CAEX y chancadores primarios."
    ),
    version="1.0.0",
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "n_equipment": scorer.equipment_metadata.height,
        "n_equipment_scoreable": scorer.latest_features.height,
    }


@app.get("/equipment")
def list_equipment(faena: Faena | None = None, equipment_type: EquipmentType | None = None) -> list[dict]:
    df = scorer.equipment_metadata
    if faena is not None:
        df = df.filter(pl.col("faena") == faena)
    if equipment_type is not None:
        df = df.filter(pl.col("equipment_type") == equipment_type)
    return df.select(
        ["equipment_id", "equipment_type", "model", "faena", "manufacture_year", "event_observed"]
    ).to_dicts()


@app.get("/equipment/{equipment_id}/risk", response_model=RiskScoreResponse)
def get_equipment_risk(equipment_id: str) -> dict:
    score = scorer.score_equipment(equipment_id)
    if score is None:
        raise HTTPException(
            status_code=404,
            detail=f"Equipo '{equipment_id}' no encontrado o sin telemetria suficiente para scoring.",
        )
    return score


@app.get("/fleet/risk-summary")
def fleet_risk_summary() -> dict:
    fleet_scores = scorer.score_fleet()
    return {
        "n_equipment_scored": fleet_scores.height,
        "by_risk_level": fleet_scores.group_by("risk_level").len().sort("risk_level").to_dicts(),
        "by_faena_and_risk": fleet_scores.group_by(["faena", "risk_level"])
        .len()
        .sort(["faena", "risk_level"])
        .to_dicts(),
    }


@app.post("/score/raw", response_model=RiskScoreResponse)
def score_raw_telemetry(payload: RawTelemetryFeatures) -> dict:
    row = payload.model_dump()
    return scorer.score_row(row, equipment_id="manual-input", timestamp="n/a")
