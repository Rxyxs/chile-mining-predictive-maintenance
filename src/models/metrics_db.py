"""Persistencia de metricas comparativas de los 3 enfoques de modelado
(CoxPH, LightGBM, PyTorch multi-task) y de la ablacion de activaciones
en una base DuckDB local, para poder consultarlas con SQL en vez de solo
leer `metrics.json` / `activation_comparison.json`.

Ejecutar desde la raiz del repositorio con:
    python -m src.models.metrics_db
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import duckdb

from src.models.train_survival_pipeline import PROCESSED_DIR

DB_PATH = PROCESSED_DIR / "metrics.duckdb"

_CREATE_COMPARISON_TABLE = """
CREATE TABLE IF NOT EXISTS model_comparison (
    run_ts      TIMESTAMP,
    model       VARCHAR,
    task        VARCHAR,
    metric      VARCHAR,
    value       DOUBLE
)
"""

_CREATE_ACTIVATION_TABLE = """
CREATE TABLE IF NOT EXISTS activation_comparison (
    run_ts      TIMESTAMP,
    activation  VARCHAR,
    metric      VARCHAR,
    value       DOUBLE
)
"""


def _connect(db_path: Path = DB_PATH) -> duckdb.DuckDBPyConnection:
    db_path.parent.mkdir(parents=True, exist_ok=True)
    con = duckdb.connect(str(db_path))
    con.execute(_CREATE_COMPARISON_TABLE)
    con.execute(_CREATE_ACTIVATION_TABLE)
    return con


# Claves de nivel superior de metrics.json producidas directamente por
# train_survival_pipeline.main() (CoxPH + LightGBM), no anidadas bajo un
# sub-modelo: se mapean explicitamente a (model, task).
_FLAT_METRIC_TO_MODEL_TASK = {
    "c_index_survival": ("coxph", "survival"),
    "mae_rul_hours": ("lightgbm", "rul_regression"),
    "accuracy_failure_type": ("lightgbm", "failure_classification"),
    "f1_macro_failure_type": ("lightgbm", "failure_classification"),
}


def persist_model_comparison(
    metrics: dict, db_path: Path = DB_PATH, run_ts: datetime | None = None
) -> None:
    """Aplana `metrics.json` (CoxPH + LightGBM + PyTorch multi-task) a filas
    `(model, task, metric, value)` y las inserta en `model_comparison`.

    `metrics.json` mezcla dos formas: claves planas de nivel superior para
    CoxPH/LightGBM (como las escribe `train_survival_pipeline.main()`) y una
    clave anidada `multi_task_pytorch` con sus propias metricas (como la
    agrega `multi_task_net.main()`). Ambas se soportan; cualquier otra clave
    anidada con metricas numericas tambien se persiste, inferiendo la tarea
    por el nombre de la clave. Ignora valores no numericos (p.ej.
    `loss_history`, `n_train_rows`, `activation`).
    """
    run_ts = run_ts or datetime.now(timezone.utc)
    rows: list[tuple] = []
    for key, value in metrics.items():
        if isinstance(value, dict):
            task = _infer_task(key)
            for metric_name, sub_value in value.items():
                if isinstance(sub_value, bool) or not isinstance(sub_value, (int, float)):
                    continue
                rows.append((run_ts, key, task, metric_name, float(sub_value)))
        elif key in _FLAT_METRIC_TO_MODEL_TASK and isinstance(value, (int, float)) and not isinstance(value, bool):
            model_key, task = _FLAT_METRIC_TO_MODEL_TASK[key]
            rows.append((run_ts, model_key, task, key, float(value)))

    con = _connect(db_path)
    try:
        con.executemany(
            "INSERT INTO model_comparison VALUES (?, ?, ?, ?, ?)", rows
        )
    finally:
        con.close()


def persist_activation_comparison(
    results: dict[str, dict], db_path: Path = DB_PATH, run_ts: datetime | None = None
) -> None:
    """Inserta los resultados de `activation_comparison.run_activation_comparison`
    (uno por activacion: relu/gelu/swish) en la tabla `activation_comparison`.
    """
    run_ts = run_ts or datetime.now(timezone.utc)
    rows: list[tuple] = []
    for activation_name, act_metrics in results.items():
        for metric_name, value in act_metrics.items():
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                continue
            rows.append((run_ts, activation_name, metric_name, float(value)))

    con = _connect(db_path)
    try:
        con.executemany(
            "INSERT INTO activation_comparison VALUES (?, ?, ?, ?)", rows
        )
    finally:
        con.close()


def _infer_task(model_key: str) -> str:
    if "coxph" in model_key or "survival" in model_key:
        return "survival"
    if "classifier" in model_key or "failure" in model_key:
        return "failure_classification"
    if "rul" in model_key:
        return "rul_regression"
    if "multi_task" in model_key:
        return "rul_regression + failure_classification"
    return "unknown"


def load_model_comparison(db_path: Path = DB_PATH):
    """Devuelve la tabla `model_comparison` completa como `pyarrow`/dataframe-like
    (via `duckdb`'s `.df()`), util para inspeccion rapida o notebooks."""
    con = _connect(db_path)
    try:
        return con.execute("SELECT * FROM model_comparison ORDER BY run_ts DESC").df()
    finally:
        con.close()


def main() -> None:
    metrics_path = PROCESSED_DIR / "metrics.json"
    if not metrics_path.exists():
        raise FileNotFoundError(
            f"No se encontro {metrics_path}. Corre primero "
            "`python -m src.models.train_survival_pipeline` y "
            "`python -m src.models.multi_task_net`."
        )
    with open(metrics_path, encoding="utf-8") as f:
        metrics = json.load(f)
    persist_model_comparison(metrics)
    print(f"Metricas comparativas persistidas en: {DB_PATH}")

    activation_path = PROCESSED_DIR / "activation_comparison.json"
    if activation_path.exists():
        with open(activation_path, encoding="utf-8") as f:
            activation_results = json.load(f)
        persist_activation_comparison(activation_results)
        print("Comparacion de activaciones tambien persistida.")


if __name__ == "__main__":
    main()
