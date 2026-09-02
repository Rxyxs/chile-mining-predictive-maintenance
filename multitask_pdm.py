"""Entrenamiento de produccion + explicabilidad DeepSHAP para la red
multi-task de PyTorch (RUL + tipo de falla).

La definicion del modelo y la funcion de entrenamiento viven en
`src/models/multi_task_net.py` (probado en `tests/test_multi_task_net.py`);
este script es el punto de entrada de produccion que:

    1. Entrena la red multi-task y persiste sus pesos + scaler (mismos
       artefactos que usa `src.models.multi_task_net`, para que cualquiera
       de los dos entrypoints deje el modelo en el mismo estado).
    2. Construye explicadores **DeepSHAP** (`shap.DeepExplainer`) para cada
       cabeza (RUL y tipo de falla) y los persiste listos para servir en la
       API sin tener que reentrenar.
    3. Guarda el historial de perdida por epoca (total, componente RUL,
       componente clasificacion; train y validacion) para las curvas del
       notebook `02_PyTorch_MultiTask_DeepSHAP.ipynb`.
    4. Guarda una tabla de importancia SHAP agregada (CSV + grafico de barras)
       sobre una muestra del set de prueba, siguiendo el mismo patron que
       `train_survival_pipeline.py` usa para el TreeSHAP de LightGBM.

Por que DeepSHAP y no KernelSHAP: DeepSHAP (DeepLIFT) requiere una entrada
continua y diferenciable de principio a fin, lo que no es compatible
directamente con los indices de embedding categoricos (`equipment_type`,
`faena`) que usa la red. Se resuelve separando el modelo en dos partes: (a)
el lookup de embeddings, que se resuelve una vez por fila (no diferenciable,
no lo necesita), y (b) el tronco compartido + cada cabeza, que si es una
funcion continua de [features numericas ++ vectores de embedding] y es sobre
esa funcion que corre `shap.DeepExplainer`. Los valores SHAP de cada
dimension de embedding se suman de vuelta a una unica atribucion por
variable categorica (`equipment_type`, `faena`) usando la propiedad de
aditividad de Shapley -- la suma de las contribuciones de las dimensiones de
un bloque de embedding es exactamente la contribucion total de esa variable.

Ejecutar desde la raiz del repositorio con:
    python multitask_pdm.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

if sys.platform == "win32" and hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import polars as pl
import shap
import torch
from torch import nn

from src.data.mining_data_generator import FAILURE_TYPES
from src.models.multi_task_net import (
    NUMERIC_COLUMNS,
    RUL_SCALE_HOURS,
    MultiTaskDegradationNet,
    build_multi_task_table,
    train_multi_task_model,
    _to_tensors,
)
from src.models.train_survival_pipeline import MODELS_DIR, PROCESSED_DIR, equipment_train_test_split

EMBEDDING_DIM = 4  # debe calzar con MultiTaskDegradationNet(embedding_dim=...)
SHAP_BACKGROUND_SIZE = 100
SHAP_EXPLAIN_SAMPLE_SIZE = 200

CONTINUOUS_FEATURE_NAMES = (
    NUMERIC_COLUMNS
    + [f"equipment_type_emb_{i}" for i in range(EMBEDDING_DIM)]
    + [f"faena_emb_{i}" for i in range(EMBEDDING_DIM)]
)
# Grupos de columnas continuas que se suman de vuelta a una sola variable
# interpretable (una entrada por dimension de embedding no es interpretable
# por si sola: cada dimension es una coordenada latente, no una categoria).
FEATURE_GROUPS = {name: [name] for name in NUMERIC_COLUMNS}
FEATURE_GROUPS["equipment_type"] = [f"equipment_type_emb_{i}" for i in range(EMBEDDING_DIM)]
FEATURE_GROUPS["faena"] = [f"faena_emb_{i}" for i in range(EMBEDDING_DIM)]


class _TrunkHead(nn.Module):
    """Tronco + una cabeza, expuesto como funcion continua x -> salida.

    Comparte los pesos de `trunk`/`head` con el modelo original (no los
    copia): es solo un envoltorio para que DeepSHAP vea una funcion de una
    sola salida sobre un tensor continuo, sin pasar por el lookup de
    embeddings (que no es diferenciable de forma util para DeepLIFT).
    """

    def __init__(self, trunk: nn.Module, head: nn.Module) -> None:
        super().__init__()
        self.trunk = trunk
        self.head = head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.trunk(x))


def to_continuous_input(
    model: MultiTaskDegradationNet,
    numeric: torch.Tensor,
    equipment_type_idx: torch.Tensor,
    faena_idx: torch.Tensor,
) -> torch.Tensor:
    """Resuelve el lookup de embeddings (no diferenciable, no hace falta que
    lo sea) y concatena con las features numericas -- el mismo tensor que
    `model.trunk` recibe internamente, pero calculado por fuera para poder
    envolver solo la parte continua en `_TrunkHead`."""
    model.eval()
    with torch.no_grad():
        emb = torch.cat([model.equipment_type_emb(equipment_type_idx), model.faena_emb(faena_idx)], dim=1)
    return torch.cat([numeric, emb], dim=1)


def build_deep_explainers(
    model: MultiTaskDegradationNet, background_continuous: torch.Tensor
) -> tuple[shap.DeepExplainer, shap.DeepExplainer]:
    model.eval()
    rul_explainer = shap.DeepExplainer(_TrunkHead(model.trunk, model.rul_head), background_continuous)
    failure_explainer = shap.DeepExplainer(_TrunkHead(model.trunk, model.failure_head), background_continuous)
    return rul_explainer, failure_explainer


def aggregate_grouped_shap(shap_values: np.ndarray) -> dict[str, float]:
    """Suma las columnas de cada grupo de `FEATURE_GROUPS` (columna a
    columna, para un vector de shap_values de una sola instancia/salida) --
    valido porque los valores SHAP son aditivos: la suma de las
    contribuciones de las dimensiones de un embedding es su contribucion
    total a la prediccion."""
    by_name = dict(zip(CONTINUOUS_FEATURE_NAMES, shap_values))
    return {group: float(sum(by_name[col] for col in cols)) for group, cols in FEATURE_GROUPS.items()}


def compute_deep_shap_importance(
    rul_explainer: shap.DeepExplainer,
    failure_explainer: shap.DeepExplainer,
    sample_continuous: torch.Tensor,
    failure_type_names: list[str],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Importancia global (media |SHAP| agregada por variable) sobre una
    muestra del set de prueba, para la cabeza de RUL y para la cabeza de
    clasificacion (promediada entre clases, como en
    `train_survival_pipeline.compute_shap_importance_multiclass`)."""
    rul_shap = np.array(rul_explainer.shap_values(sample_continuous)).squeeze(-1)  # (n, d)
    failure_shap = np.array(failure_explainer.shap_values(sample_continuous))  # (n, d, n_classes)

    rul_grouped = np.array([list(aggregate_grouped_shap(row).values()) for row in rul_shap])
    failure_grouped = np.stack(
        [
            np.array([list(aggregate_grouped_shap(failure_shap[i, :, c]).values()) for i in range(failure_shap.shape[0])])
            for c in range(len(failure_type_names))
        ],
        axis=0,
    )  # (n_classes, n_samples, n_groups)

    group_names = list(FEATURE_GROUPS.keys())
    rul_importance = (
        pd.DataFrame({"feature": group_names, "mean_abs_shap": np.abs(rul_grouped).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    failure_importance = (
        pd.DataFrame({"feature": group_names, "mean_abs_shap": np.abs(failure_grouped).mean(axis=(0, 1))})
        .sort_values("mean_abs_shap", ascending=False)
        .reset_index(drop=True)
    )
    return rul_importance, failure_importance


def _save_bar_plot(importance: pd.DataFrame, title: str, path: Path) -> None:
    plt.figure(figsize=(8, 5))
    ordered = importance.sort_values("mean_abs_shap")
    plt.barh(ordered["feature"], ordered["mean_abs_shap"], color="#2E86AB")
    plt.xlabel("Media |SHAP| (DeepSHAP)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=120)
    plt.close()


def explain_instance(
    model: MultiTaskDegradationNet,
    rul_explainer: shap.DeepExplainer,
    failure_explainer: shap.DeepExplainer,
    numeric: torch.Tensor,
    equipment_type_idx: torch.Tensor,
    faena_idx: torch.Tensor,
    failure_type_names: list[str] = FAILURE_TYPES,
) -> dict:
    """Explicacion DeepSHAP de una sola fila: usada tanto por el notebook
    como por el endpoint `/multitask/explicar` de la API."""
    x = to_continuous_input(model, numeric, equipment_type_idx, faena_idx)

    with torch.no_grad():
        rul_pred_scaled, failure_logits = model(numeric, equipment_type_idx, faena_idx)
        failure_proba = torch.softmax(failure_logits, dim=1)[0]
        predicted_class_idx = int(failure_proba.argmax())

    rul_shap = np.array(rul_explainer.shap_values(x)).squeeze()  # (d,)
    failure_shap = np.array(failure_explainer.shap_values(x))[0, :, predicted_class_idx]  # (d,)

    return {
        "predicted_rul_hours": float(rul_pred_scaled.item() * RUL_SCALE_HOURS),
        "predicted_failure_type": failure_type_names[predicted_class_idx],
        "failure_type_probabilities": {
            name: round(float(p), 4) for name, p in zip(failure_type_names, failure_proba.tolist())
        },
        "shap_rul": aggregate_grouped_shap(rul_shap),
        "shap_failure_type": aggregate_grouped_shap(failure_shap),
    }


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    equipment_metadata = pl.read_parquet(PROCESSED_DIR / "equipment_metadata.parquet")
    maintenance_logs = pl.read_parquet(PROCESSED_DIR / "maintenance_logs.parquet")
    features = pl.read_parquet(PROCESSED_DIR / "telemetry_features.parquet")

    mt_df = build_multi_task_table(features, equipment_metadata, maintenance_logs)
    failed_ids = mt_df["equipment_id"].unique().to_list()
    train_ids, test_ids = equipment_train_test_split(failed_ids, test_size=0.25, seed=42)

    print(f"Entrenando red multi-task (PyTorch) sobre {mt_df.height} lecturas de {len(failed_ids)} equipos fallados...")
    model, scaler, metrics = train_multi_task_model(mt_df, train_ids, test_ids)
    loss_history = metrics.pop("loss_history")

    print(f"\n[Multi-task PyTorch] MAE RUL: {metrics['mae_rul_hours']:.1f} horas")
    print(
        f"[Multi-task PyTorch] Clasificacion de falla: "
        f"Accuracy {metrics['accuracy_failure_type']:.3f} | F1-macro {metrics['f1_macro_failure_type']:.3f}"
    )

    torch.save(model.state_dict(), MODELS_DIR / "multi_task_net.pt")
    joblib.dump(scaler, MODELS_DIR / "multi_task_scaler.joblib")

    print("\nConstruyendo explicadores DeepSHAP (tronco + cabeza RUL / cabeza clasificacion)...")
    test_df = mt_df.filter(pl.col("equipment_id").is_in(test_ids))
    train_df = mt_df.filter(pl.col("equipment_id").is_in(train_ids))

    background_df = train_df.sample(n=min(SHAP_BACKGROUND_SIZE, train_df.height), seed=42)
    background_t = _to_tensors(background_df, scaler, fit_scaler=False)
    background_continuous = to_continuous_input(
        model, background_t["numeric"], background_t["equipment_type_idx"], background_t["faena_idx"]
    )

    rul_explainer, failure_explainer = build_deep_explainers(model, background_continuous)
    torch.save(background_continuous, MODELS_DIR / "multi_task_shap_background.pt")

    sample_df = test_df.sample(n=min(SHAP_EXPLAIN_SAMPLE_SIZE, test_df.height), seed=42)
    sample_t = _to_tensors(sample_df, scaler, fit_scaler=False)
    sample_continuous = to_continuous_input(
        model, sample_t["numeric"], sample_t["equipment_type_idx"], sample_t["faena_idx"]
    )
    rul_importance, failure_importance = compute_deep_shap_importance(
        rul_explainer, failure_explainer, sample_continuous, FAILURE_TYPES
    )
    rul_importance.to_csv(PROCESSED_DIR / "shap_multitask_rul_importance.csv", index=False)
    failure_importance.to_csv(PROCESSED_DIR / "shap_multitask_failure_importance.csv", index=False)
    _save_bar_plot(rul_importance, "DeepSHAP - importancia global (RUL, red multi-task)", MODELS_DIR / "shap_multitask_rul_summary.png")
    _save_bar_plot(
        failure_importance,
        "DeepSHAP - importancia global (tipo de falla, red multi-task)",
        MODELS_DIR / "shap_multitask_failure_summary.png",
    )
    print(f"  Top feature RUL (DeepSHAP): {rul_importance.iloc[0]['feature']}")
    print(f"  Top feature tipo de falla (DeepSHAP): {failure_importance.iloc[0]['feature']}")

    with open(PROCESSED_DIR / "multitask_loss_history.json", "w", encoding="utf-8") as f:
        json.dump(loss_history, f, indent=2)

    metrics_path = PROCESSED_DIR / "metrics.json"
    existing = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing["multitask_pdm"] = metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\nModelo, scaler y artefactos DeepSHAP guardados en: {MODELS_DIR}")
    print(f"Historial de perdida por epoca: {PROCESSED_DIR / 'multitask_loss_history.json'}")
    print(f"Metricas combinadas: {metrics_path}")


if __name__ == "__main__":
    main()
