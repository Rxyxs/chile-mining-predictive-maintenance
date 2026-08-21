"""Red multi-task en PyTorch (tronco compartido) para RUL + clasificacion de
tipo de falla, entrenada y evaluada como alternativa comparable a los dos
modelos LightGBM independientes de `train_survival_pipeline.py`.

Un tronco denso comparte representacion entre ambas tareas: procesa
features numericas de degradacion + embeddings de `equipment_type`/`faena`,
y dos cabezas lineales predicen RUL (regresion) y tipo de falla
(clasificacion multiclase) a partir de esa representacion compartida.

Ejecutar desde la raiz del repositorio con:
    python -m src.models.multi_task_net
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import polars as pl
import torch
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.data.mining_data_generator import EQUIPMENT_TYPES, FAENAS, FAILURE_TYPES
from src.features.engineering import FEATURE_COLUMNS
from src.models.train_survival_pipeline import (
    MODELS_DIR,
    PROCESSED_DIR,
    build_rul_training_table,
    equipment_train_test_split,
)

NUMERIC_COLUMNS = FEATURE_COLUMNS + ["operating_hours", "age_years"]
EQUIPMENT_TYPE_TO_IDX = {name: i for i, name in enumerate(EQUIPMENT_TYPES)}
FAENA_TO_IDX = {name: i for i, name in enumerate(FAENAS)}
FAILURE_TYPE_TO_IDX = {name: i for i, name in enumerate(FAILURE_TYPES)}
RUL_SCALE_HOURS = 1000.0  # normaliza el target de RUL para balancear ambas perdidas


class MultiTaskDegradationNet(nn.Module):
    """Tronco compartido + cabeza de regresion (RUL) y cabeza de clasificacion (tipo de falla)."""

    def __init__(
        self,
        n_numeric_features: int,
        n_equipment_types: int = len(EQUIPMENT_TYPES),
        n_faenas: int = len(FAENAS),
        n_failure_classes: int = len(FAILURE_TYPES),
        embedding_dim: int = 4,
        hidden_dim: int = 64,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.equipment_type_emb = nn.Embedding(n_equipment_types, embedding_dim)
        self.faena_emb = nn.Embedding(n_faenas, embedding_dim)

        trunk_input_dim = n_numeric_features + 2 * embedding_dim
        self.trunk = nn.Sequential(
            nn.Linear(trunk_input_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.BatchNorm1d(hidden_dim),
            nn.Dropout(dropout),
        )
        self.rul_head = nn.Linear(hidden_dim, 1)
        self.failure_head = nn.Linear(hidden_dim, n_failure_classes)

    def forward(
        self, x_numeric: torch.Tensor, equipment_type_idx: torch.Tensor, faena_idx: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        emb = torch.cat([self.equipment_type_emb(equipment_type_idx), self.faena_emb(faena_idx)], dim=1)
        shared = self.trunk(torch.cat([x_numeric, emb], dim=1))
        rul_pred = self.rul_head(shared).squeeze(-1)
        failure_logits = self.failure_head(shared)
        return rul_pred, failure_logits


def build_multi_task_table(
    features: pl.DataFrame, equipment_metadata: pl.DataFrame, maintenance_logs: pl.DataFrame
) -> pl.DataFrame:
    """Filas de telemetria de equipos fallados con ambas etiquetas: `rul_hours` y `failure_type`.

    Reutiliza `build_rul_training_table` y le suma el tipo de falla real
    (conocido retrospectivamente para cualquier lectura de un equipo que
    efectivamente fallo), habilitando entrenamiento multi-task por fila en
    vez de solo en la ultima lectura de cada equipo.
    """
    rul_df = build_rul_training_table(features, equipment_metadata)
    failure_events = maintenance_logs.filter(pl.col("event_type") == "falla_no_planificada").select(
        ["equipment_id", "failure_type"]
    )
    return rul_df.join(failure_events, on="equipment_id", how="inner")


def _map_to_idx(series: pl.Series, mapping: dict[str, int]) -> np.ndarray:
    return np.array([mapping[v] for v in series.to_list()], dtype=np.int64)


def _to_tensors(df: pl.DataFrame, scaler: StandardScaler, fit_scaler: bool) -> dict[str, torch.Tensor]:
    numeric = df.select(NUMERIC_COLUMNS).to_pandas().to_numpy(dtype=np.float32)
    numeric = scaler.fit_transform(numeric) if fit_scaler else scaler.transform(numeric)

    return {
        "numeric": torch.tensor(numeric, dtype=torch.float32),
        "equipment_type_idx": torch.tensor(_map_to_idx(df["equipment_type"], EQUIPMENT_TYPE_TO_IDX)),
        "faena_idx": torch.tensor(_map_to_idx(df["faena"], FAENA_TO_IDX)),
        "failure_idx": torch.tensor(_map_to_idx(df["failure_type"], FAILURE_TYPE_TO_IDX)),
        "rul_scaled": torch.tensor((df["rul_hours"].to_numpy() / RUL_SCALE_HOURS).astype(np.float32)),
    }


def train_multi_task_model(
    mt_df: pl.DataFrame,
    train_ids: set[str],
    test_ids: set[str],
    epochs: int = 60,
    batch_size: int = 256,
    lr: float = 1e-3,
    classification_loss_weight: float = 1.0,
    seed: int = 42,
) -> tuple[MultiTaskDegradationNet, StandardScaler, dict]:
    torch.manual_seed(seed)

    train_df = mt_df.filter(pl.col("equipment_id").is_in(train_ids))
    test_df = mt_df.filter(pl.col("equipment_id").is_in(test_ids))

    scaler = StandardScaler()
    train_t = _to_tensors(train_df, scaler, fit_scaler=True)
    test_t = _to_tensors(test_df, scaler, fit_scaler=False)

    model = MultiTaskDegradationNet(n_numeric_features=len(NUMERIC_COLUMNS))
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    mse_loss = nn.MSELoss()
    ce_loss = nn.CrossEntropyLoss()

    train_dataset = TensorDataset(
        train_t["numeric"], train_t["equipment_type_idx"], train_t["faena_idx"], train_t["failure_idx"], train_t["rul_scaled"]
    )
    # drop_last evita que un ultimo batch de tamano 1 rompa BatchNorm en modo train.
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)

    n_batches = len(train_loader)
    model.train()
    for epoch in range(epochs):
        epoch_loss = 0.0
        for x_num, eq_idx, faena_idx, fail_idx, rul_scaled in train_loader:
            optimizer.zero_grad()
            rul_pred, failure_logits = model(x_num, eq_idx, faena_idx)
            loss = mse_loss(rul_pred, rul_scaled) + classification_loss_weight * ce_loss(failure_logits, fail_idx)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        epoch_loss /= max(n_batches, 1)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"  epoch {epoch + 1}/{epochs} - loss combinado: {epoch_loss:.4f}")

    model.eval()
    with torch.no_grad():
        rul_pred_scaled, failure_logits = model(test_t["numeric"], test_t["equipment_type_idx"], test_t["faena_idx"])
        rul_pred_hours = (rul_pred_scaled * RUL_SCALE_HOURS).numpy()
        rul_true_hours = (test_t["rul_scaled"] * RUL_SCALE_HOURS).numpy()
        failure_pred_idx = failure_logits.argmax(dim=1).numpy()
        failure_true_idx = test_t["failure_idx"].numpy()

    metrics = {
        "mae_rul_hours": float(mean_absolute_error(rul_true_hours, rul_pred_hours)),
        "accuracy_failure_type": float(accuracy_score(failure_true_idx, failure_pred_idx)),
        "f1_macro_failure_type": float(f1_score(failure_true_idx, failure_pred_idx, average="macro")),
        "n_train_rows": train_df.height,
        "n_test_rows": test_df.height,
    }
    return model, scaler, metrics


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    equipment_metadata = pl.read_parquet(PROCESSED_DIR / "equipment_metadata.parquet")
    maintenance_logs = pl.read_parquet(PROCESSED_DIR / "maintenance_logs.parquet")
    features = pl.read_parquet(PROCESSED_DIR / "telemetry_features.parquet")

    mt_df = build_multi_task_table(features, equipment_metadata, maintenance_logs)

    # Mismo criterio de split (semilla + equipos fallados) que train_survival_pipeline,
    # para que la comparacion de MAE de RUL sea sobre el mismo holdout.
    failed_ids = mt_df["equipment_id"].unique().to_list()
    train_ids, test_ids = equipment_train_test_split(failed_ids, test_size=0.25, seed=42)

    print(f"Entrenando red multi-task (PyTorch) sobre {mt_df.height} lecturas de {len(failed_ids)} equipos fallados...")
    model, scaler, metrics = train_multi_task_model(mt_df, train_ids, test_ids)

    print(f"\n[Multi-task PyTorch] MAE RUL: {metrics['mae_rul_hours']:.1f} horas")
    print(
        f"[Multi-task PyTorch] Clasificacion de falla (por lectura, no solo la ultima): "
        f"Accuracy {metrics['accuracy_failure_type']:.3f} | F1-macro {metrics['f1_macro_failure_type']:.3f}"
    )

    torch.save(model.state_dict(), MODELS_DIR / "multi_task_net.pt")
    joblib.dump(scaler, MODELS_DIR / "multi_task_scaler.joblib")

    metrics_path = PROCESSED_DIR / "metrics.json"
    existing = {}
    if metrics_path.exists():
        with open(metrics_path, encoding="utf-8") as f:
            existing = json.load(f)
    existing["multi_task_pytorch"] = metrics
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(existing, f, indent=2, ensure_ascii=False)

    print(f"\nModelo guardado en: {MODELS_DIR / 'multi_task_net.pt'}")
    print(f"Metricas combinadas guardadas en: {metrics_path}")


if __name__ == "__main__":
    main()
