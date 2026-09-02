"""Comparacion de funciones de activacion (ReLU vs GELU vs Swish/SiLU) para
la red multi-task de `multi_task_net.py`.

Entrena la misma arquitectura (`MultiTaskDegradationNet`), sobre el mismo
split de holdout, variando unicamente la activacion del tronco compartido.
El objetivo no es reemplazar la comparacion CoxPH / LightGBM / PyTorch ya
documentada en el README, sino profundizar el analisis del tercer enfoque
(deep learning) con una ablacion estandar de arquitectura.

Ejecutar desde la raiz del repositorio con:
    python -m src.models.activation_comparison
"""

from __future__ import annotations

import json

import polars as pl

from src.models.metrics_db import persist_activation_comparison
from src.models.multi_task_net import (
    ACTIVATIONS,
    build_multi_task_table,
    train_multi_task_model,
)
from src.models.train_survival_pipeline import (
    PROCESSED_DIR,
    equipment_train_test_split,
)


def run_activation_comparison(
    mt_df: pl.DataFrame,
    train_ids: set[str],
    test_ids: set[str],
    epochs: int = 60,
    seed: int = 42,
) -> dict[str, dict]:
    """Entrena una variante de la red por cada activacion en `ACTIVATIONS`.

    Devuelve un dict `{activation_name: metrics}` con las mismas metricas
    que produce `train_multi_task_model` (MAE de RUL, accuracy/F1 de tipo
    de falla, historial de loss), para que sea directamente comparable con
    la entrada `multi_task_pytorch` de `metrics.json`.
    """
    results: dict[str, dict] = {}
    for activation_name in ACTIVATIONS:
        print(f"Entrenando red multi-task con activacion '{activation_name}'...")
        _, _, metrics = train_multi_task_model(
            mt_df, train_ids, test_ids, epochs=epochs, seed=seed, activation=activation_name
        )
        # El historial de loss por epoca infla mucho el JSON de comparacion;
        # se conserva solo en el metrics.json principal via multi_task_net.py.
        metrics_summary = {k: v for k, v in metrics.items() if k != "loss_history"}
        results[activation_name] = metrics_summary
        print(
            f"  [{activation_name}] MAE RUL: {metrics_summary['mae_rul_hours']:.1f} h | "
            f"Accuracy falla: {metrics_summary['accuracy_failure_type']:.3f}"
        )
    return results


def main() -> None:
    equipment_metadata = pl.read_parquet(PROCESSED_DIR / "equipment_metadata.parquet")
    maintenance_logs = pl.read_parquet(PROCESSED_DIR / "maintenance_logs.parquet")
    features = pl.read_parquet(PROCESSED_DIR / "telemetry_features.parquet")

    mt_df = build_multi_task_table(features, equipment_metadata, maintenance_logs)
    failed_ids = mt_df["equipment_id"].unique().to_list()
    train_ids, test_ids = equipment_train_test_split(failed_ids, test_size=0.25, seed=42)

    results = run_activation_comparison(mt_df, train_ids, test_ids)

    out_path = PROCESSED_DIR / "activation_comparison.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    print(f"\nComparacion de activaciones guardada en: {out_path}")

    persist_activation_comparison(results)
    print("Comparacion de activaciones persistida en DuckDB (data/processed/metrics.duckdb).")


if __name__ == "__main__":
    main()
