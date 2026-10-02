"""Carga del dataset real SCANIA Component X (IDA 2024 Industrial Challenge).

Flota de 33.641 camiones pesados con un componente de motor anonimizado
("Component X"): lecturas operacionales irregulares (contadores e histogramas),
especificaciones categoricas del vehiculo y, para cada camion, si el componente
fue reparado durante el estudio y cuando (``time to event``).

Fuente: https://researchdata.se/en/catalogue/dataset/2024-34  (CC BY 4.0, sin registro)
Descripcion: "SCANIA Component X dataset: a real-world multivariate time series dataset for
predictive maintenance", Scientific Data (2025), https://www.nature.com/articles/s41597-025-04802-6

Los archivos de validacion y test traen UNA sola lectura por camion, elegida al azar
como "la ultima" (simula usar el modelo en produccion con informacion hasta hoy).
Los de train traen toda la historia hasta el fin del estudio.
"""

from __future__ import annotations

import time
from pathlib import Path

import polars as pl
import requests

ROOT = Path(__file__).resolve().parents[2]
RAW_DIR = ROOT / "data" / "raw" / "scania"
BASE_URL = "https://api.researchdata.se/dataset/2024-34/3/file/data/"

FILES = [
    "train_tte.csv",
    "train_specifications.csv",
    "validation_labels.csv",
    "validation_specifications.csv",
    "test_labels.csv",
    "test_specifications.csv",
    "validation_operational_readouts.csv",
    "test_operational_readouts.csv",
    "train_operational_readouts.csv",
]

# Ventanas de la etiqueta oficial del reto: tiempo restante hasta la reparacion.
# clase 4: 0-6 | clase 3: 6-12 | clase 2: 12-24 | clase 1: 24-48 | clase 0: mas de 48 o sin falla
CLASS_UPPER_BOUNDS = [(6.0, 4), (12.0, 3), (24.0, 2), (48.0, 1)]

# Matriz de costo oficial: COST[real][predicho]. Revisar una camion sano cuesta poco
# (7-10); no detectar una falla cuesta mucho (200-500).
COST_MATRIX = [
    [0, 7, 8, 9, 10],
    [200, 0, 7, 8, 9],
    [300, 200, 0, 7, 8],
    [400, 300, 200, 0, 7],
    [500, 400, 300, 200, 0],
]


def download(raw_dir: Path = RAW_DIR, retries: int = 3) -> list[Path]:
    """Descarga los archivos que faltan o estan incompletos. Idempotente."""
    raw_dir.mkdir(parents=True, exist_ok=True)
    paths = []
    for name in FILES:
        target = raw_dir / name
        head = requests.head(BASE_URL + name, allow_redirects=True, timeout=60)
        expected = int(head.headers.get("Content-Length", 0))
        if not (target.exists() and target.stat().st_size > 0 and (not expected or target.stat().st_size == expected)):
            for attempt in range(1, retries + 1):
                with requests.get(BASE_URL + name, stream=True, timeout=120) as resp:
                    resp.raise_for_status()
                    with target.open("wb") as fh:
                        for chunk in resp.iter_content(chunk_size=1 << 20):
                            fh.write(chunk)
                if not expected or target.stat().st_size == expected:
                    break
                time.sleep(2 * attempt)
            else:
                raise RuntimeError(f"Descarga incompleta: {name}")
        paths.append(target)
    return paths


def read_readouts(split: str, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Lecturas operacionales de ``train``, ``validation`` o ``test``."""
    df = pl.read_csv(raw_dir / f"{split}_operational_readouts.csv", null_values=["na", "NA", ""])
    return df.sort(["vehicle_id", "time_step"])


def read_specs(split: str, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.read_csv(raw_dir / f"{split}_specifications.csv")


def read_tte(raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    """Tiempo hasta reparacion de train: ``duration`` y ``event`` (1 = reparado, 0 = censurado)."""
    return pl.read_csv(raw_dir / "train_tte.csv").rename(
        {"length_of_study_time_step": "duration", "in_study_repair": "event"}
    )


def read_labels(split: str, raw_dir: Path = RAW_DIR) -> pl.DataFrame:
    return pl.read_csv(raw_dir / f"{split}_labels.csv")


def class_from_time_to_failure(time_to_failure: float) -> int:
    """Clase 0-4 a partir del tiempo que falta para la reparacion (no negativo)."""
    for upper, label in CLASS_UPPER_BOUNDS:
        if time_to_failure <= upper:
            return label
    return 0


def label_train_readouts(readouts: pl.DataFrame, tte: pl.DataFrame) -> pl.DataFrame:
    """Etiqueta cada lectura de train con su clase 0-4 y su tiempo restante.

    Un camion con ``event == 0`` esta censurado: no se sabe si fallo despues, asi que
    todas sus lecturas son clase 0 (el mismo criterio que usa el reto) y su tiempo
    restante queda nulo.
    """
    df = readouts.join(tte, on="vehicle_id", how="inner")
    ttf = pl.when(pl.col("event") == 1).then(pl.col("duration") - pl.col("time_step")).otherwise(None)
    df = df.with_columns(ttf.alias("time_to_failure"))
    label = pl.lit(0, dtype=pl.Int8)
    for upper, cls in reversed(CLASS_UPPER_BOUNDS):
        label = pl.when(pl.col("time_to_failure") <= upper).then(pl.lit(cls, dtype=pl.Int8)).otherwise(label)
    return df.with_columns(label.alias("class_label"))


def total_cost(y_true, y_pred) -> int:
    """Costo total oficial del reto: suma de COST[real][predicho] sobre todos los camiones."""
    return int(sum(COST_MATRIX[int(t)][int(p)] for t, p in zip(y_true, y_pred)))
