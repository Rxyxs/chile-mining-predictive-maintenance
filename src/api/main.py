"""Servicio FastAPI de riesgo de falla de Component X (camiones pesados SCANIA).

``POST /score`` recibe el historial de lecturas de un camion y sus especificaciones, y
devuelve la probabilidad de cada clase de riesgo, el costo esperado de cada accion y la
accion de mantenimiento recomendada (la de menor costo esperado).

Requiere API key (header ``X-API-Key``) en todo salvo ``/health``, y limita las
solicitudes por IP.

Ejecutar desde la raiz del repositorio, despues de entrenar:
    uvicorn src.api.main:app --reload

Variables de entorno:
    RISK_API_KEY     API key esperada (default: "dev-key-change-me").
    RISK_RATE_LIMIT  Limite por IP, formato de ``slowapi`` (default: "60/minute").
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from fastapi import Depends, FastAPI, HTTPException, Request, Security
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.middleware import SlowAPIMiddleware
from slowapi.util import get_remote_address

from src.models.scorer import InvalidReadouts, RiskScorer

API_KEY = os.environ.get("RISK_API_KEY", "dev-key-change-me")
RATE_LIMIT = os.environ.get("RISK_RATE_LIMIT", "60/minute")

_scorer: RiskScorer | None = None


def get_scorer() -> RiskScorer:
    """Carga el modelo la primera vez que se necesita (los tests inyectan uno propio)."""
    global _scorer
    if _scorer is None:
        try:
            _scorer = RiskScorer.load()
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="Modelo no entrenado. Corre `python -m src.models.train_pipeline` primero.",
            ) from exc
    return _scorer


_api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)


def require_api_key(key: str | None = Security(_api_key_header)) -> None:
    if key != API_KEY:
        raise HTTPException(status_code=401, detail="API key invalida o ausente. Envia el header 'X-API-Key'.")


class ScoreRequest(BaseModel):
    readouts: list[dict[str, float | None]] = Field(
        ..., min_length=1, max_length=2000,
        description="Lecturas operacionales en orden de time_step: contadores e histogramas acumulados.",
    )
    specs: dict[str, str] = Field(..., description="Spec_0 ... Spec_7 del vehiculo (categoricas).")


limiter = Limiter(key_func=get_remote_address, default_limits=[RATE_LIMIT])
app = FastAPI(title="Component X risk scoring", version="2.0.0")
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(SlowAPIMiddleware)


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/score", dependencies=[Depends(require_api_key)])
def score(request: Request, body: ScoreRequest) -> dict:
    scorer = get_scorer()
    try:
        return scorer.score(body.readouts, body.specs)
    except InvalidReadouts as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
