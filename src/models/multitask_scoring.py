"""Carga en memoria de la red multi-task (PyTorch) y sus explicadores
DeepSHAP, para servirlos desde `src/api/main.py` sin volver a entrenar ni
reconstruir los explicadores en cada request.

Artefactos requeridos (generados por `python multitask_pdm.py`):
    - data/processed/models/multi_task_net.pt
    - data/processed/models/multi_task_scaler.joblib
    - data/processed/models/multi_task_shap_background.pt
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[2]
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

import joblib
import numpy as np
import torch

from src.data.mining_data_generator import FAILURE_TYPES
from src.models.multi_task_net import (
    EQUIPMENT_TYPE_TO_IDX,
    FAENA_TO_IDX,
    NUMERIC_COLUMNS,
    RUL_SCALE_HOURS,
    MultiTaskDegradationNet,
)
from src.models.train_survival_pipeline import MODELS_DIR
from multitask_pdm import build_deep_explainers, explain_instance


def _require(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(
            f"No se encontro {path}. Corre primero: python multitask_pdm.py"
        )
    return path


class MultiTaskScorer:
    """Carga el modelo multi-task, su scaler y sus explicadores DeepSHAP una
    sola vez, y sirve scoring + explicabilidad en memoria."""

    def __init__(self) -> None:
        self.model = MultiTaskDegradationNet(n_numeric_features=len(NUMERIC_COLUMNS))
        self.model.load_state_dict(torch.load(_require(MODELS_DIR / "multi_task_net.pt"), weights_only=True))
        self.model.eval()

        self.scaler = joblib.load(_require(MODELS_DIR / "multi_task_scaler.joblib"))
        background = torch.load(_require(MODELS_DIR / "multi_task_shap_background.pt"), weights_only=True)
        self.rul_explainer, self.failure_explainer = build_deep_explainers(self.model, background)

    def _row_tensors(self, row: dict) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        numeric_values = np.array([[row[col] for col in NUMERIC_COLUMNS]], dtype=np.float32)
        numeric_scaled = self.scaler.transform(numeric_values)
        numeric_t = torch.tensor(numeric_scaled, dtype=torch.float32)
        equipment_type_idx = torch.tensor([EQUIPMENT_TYPE_TO_IDX[row["equipment_type"]]], dtype=torch.long)
        faena_idx = torch.tensor([FAENA_TO_IDX[row["faena"]]], dtype=torch.long)
        return numeric_t, equipment_type_idx, faena_idx

    def score(self, row: dict) -> dict:
        numeric_t, equipment_type_idx, faena_idx = self._row_tensors(row)
        with torch.no_grad():
            rul_pred_scaled, failure_logits = self.model(numeric_t, equipment_type_idx, faena_idx)
            failure_proba = torch.softmax(failure_logits, dim=1)[0]
            predicted_class_idx = int(failure_proba.argmax())

        return {
            "predicted_rul_hours": round(float(rul_pred_scaled.item() * RUL_SCALE_HOURS), 1),
            "predicted_failure_type": FAILURE_TYPES[predicted_class_idx],
            "failure_type_probabilities": {
                name: round(float(p), 4) for name, p in zip(FAILURE_TYPES, failure_proba.tolist())
            },
        }

    def explain(self, row: dict) -> dict:
        numeric_t, equipment_type_idx, faena_idx = self._row_tensors(row)
        return explain_instance(
            self.model, self.rul_explainer, self.failure_explainer, numeric_t, equipment_type_idx, faena_idx
        )
