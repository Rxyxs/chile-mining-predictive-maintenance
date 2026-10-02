"""Decision de mantenimiento bajo la matriz de costo oficial del reto.

Un modelo de clasificacion entrega probabilidades por clase; la accion a tomar no es
``argmax`` sino la que minimiza el costo esperado. Revisar un camion sano cuesta 7-10;
no detectar una falla cuesta 200-500. Con costos tan asimetricos, ``argmax`` casi
nunca alarma (la clase 0 es el 97% de los camiones) y sale caro.
"""

from __future__ import annotations

import numpy as np

from src.data.scania_loader import COST_MATRIX


def expected_cost_matrix(proba: np.ndarray, cost: np.ndarray | None = None) -> np.ndarray:
    """Costo esperado de cada accion para cada fila: ``E[c, a] = sum_k p[c, k] * COST[k][a]``."""
    cost = np.asarray(COST_MATRIX if cost is None else cost, dtype=float)
    return np.asarray(proba, dtype=float) @ cost


def bayes_decision(proba: np.ndarray, cost: np.ndarray | None = None) -> np.ndarray:
    """Accion (clase 0-4 a predecir) que minimiza el costo esperado de cada vehiculo."""
    return expected_cost_matrix(proba, cost).argmin(axis=1)


def confusion(y_true, y_pred, n_classes: int = 5) -> np.ndarray:
    """Matriz de confusion ``M[real, predicho]``."""
    m = np.zeros((n_classes, n_classes), dtype=int)
    for t, p in zip(y_true, y_pred):
        m[int(t), int(p)] += 1
    return m


def cost_from_confusion(matrix: np.ndarray, cost: np.ndarray | None = None) -> int:
    cost = np.asarray(COST_MATRIX if cost is None else cost)
    return int((matrix * cost).sum())


def alarm_metrics(y_true, y_pred) -> dict:
    """Metricas operativas: de los camiones que van a fallar, cuantos se alarman, y a que costo."""
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    failing = y_true > 0
    alarmed = y_pred > 0
    n_fail = int(failing.sum())
    return {
        "fallas_reales": n_fail,
        "fallas_alarmadas": int((failing & alarmed).sum()),
        "recall_alarma": float((failing & alarmed).sum() / n_fail) if n_fail else float("nan"),
        "falsas_alarmas": int((~failing & alarmed).sum()),
        "camiones_sanos": int((~failing).sum()),
        "tasa_falsa_alarma": float((~failing & alarmed).sum() / max(int((~failing).sum()), 1)),
    }
