"""Generador sintetico de la base relacional de mineria chilena.

Produce tres tablas interrelacionadas para mantenimiento predictivo de
flotillas de camiones de extraccion (CAEX) y chancadores primarios:

- `equipment_metadata`: especificaciones del equipo y su reloj de
  supervivencia (horas en el ciclo de mantenimiento actual + evento
  observado).
- `sensor_telemetry`: serie temporal de sensores para el ciclo de vida
  completo de cada equipo (0 horas hasta `hours_in_current_cycle`), a
  resolucion adaptativa -- ver `build_sensor_telemetry`.
- `maintenance_logs`: eventos de mantenimiento programado y fallas no
  planificadas.

El reloj de supervivencia usado por Cox Proportional Hazards es
"horas desde la ultima gran intervencion" (`hours_in_current_cycle`), no
las horas de vida total del equipo -- asi se modela un proceso de renovacion
realista en el que los componentes se reparan/reemplazan periodicamente.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path

import numpy as np
import polars as pl

DATA_DIR = Path(__file__).resolve().parents[2] / "data"
PROCESSED_DIR = DATA_DIR / "processed"

DEFAULT_TARGET_READINGS_PER_EQUIPMENT = 600  # cubre el ciclo de vida completo a resolucion adaptativa

FAENAS: dict[str, float] = {
    "Chuquicamata": 0.90,
    "Escondida": 1.00,
    "Los Bronces": 0.95,
    "El Teniente": 1.05,
    "Radomiro Tomic": 0.92,
}

EQUIPMENT_TYPES: dict[str, dict] = {
    "CAEX": {
        "weight": 0.85,
        "eta_hours": 9000.0,
        "shape": 2.4,
        "models": ["Caterpillar 797F", "Komatsu 930E", "Liebherr T 284"],
        "rpm_mean": 1800.0,
        "rpm_std": 60.0,
        "fuel_mean_lph": 180.0,
    },
    "Chancador Primario": {
        "weight": 0.15,
        "eta_hours": 13000.0,
        "shape": 2.8,
        "models": ["Metso Superior MKIII", "FLSmidth Gyratory", "ThyssenKrupp TS"],
        "rpm_mean": 950.0,
        "rpm_std": 40.0,
        "fuel_mean_lph": 25.0,
    },
}

FAILURE_TYPES: list[str] = [
    "sobrecalentamiento_motor",
    "falla_rodamiento",
    "fuga_hidraulica",
    "falla_estructural",
    "falla_electrica",
]
FAILURE_TYPE_WEIGHTS: list[float] = [0.25, 0.25, 0.20, 0.15, 0.15]

MAINTENANCE_COMPONENTS = [
    "cambio_aceite_motor",
    "cambio_filtros",
    "revision_sistema_frenos",
    "inspeccion_estructural",
    "calibracion_sensores",
    "cambio_neumaticos",
]


@dataclass
class EquipmentProfile:
    """Estado interno completo de un equipo (incluye variables ocultas de simulacion)."""

    equipment_id: str
    equipment_type: str
    model: str
    faena: str
    manufacture_year: int
    install_date: date
    hours_in_current_cycle: float  # duration = min(T, C) del proceso de censura
    event_observed: bool
    true_failure_hours: float  # T: horas hasta la falla dentro del ciclo actual
    failure_type: str | None  # solo definido si event_observed=True


def _sample_equipment_type(rng: random.Random) -> str:
    types, weights = zip(*[(k, v["weight"]) for k, v in EQUIPMENT_TYPES.items()])
    return rng.choices(types, weights=weights, k=1)[0]


def _build_profiles(n_equipment: int, seed: int, reference_date: date) -> list[EquipmentProfile]:
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)
    profiles: list[EquipmentProfile] = []

    for i in range(n_equipment):
        equipment_type = _sample_equipment_type(rng)
        type_cfg = EQUIPMENT_TYPES[equipment_type]
        faena = rng.choice(list(FAENAS.keys()))
        model = rng.choice(type_cfg["models"])

        manufacture_year = rng.randint(reference_date.year - 18, reference_date.year - 1)
        install_date = date(manufacture_year, rng.randint(1, 12), rng.randint(1, 28))
        age_years = (reference_date - install_date).days / 365.25
        age_factor = 0.90 if age_years > 12 else (0.97 if age_years > 6 else 1.0)

        eta_local = type_cfg["eta_hours"] * FAENAS[faena] * age_factor
        shape_local = max(1.2, type_cfg["shape"] + np_rng.normal(0, 0.15))

        true_failure_hours = float(np_rng.weibull(shape_local) * eta_local)
        censor_hours = float(np_rng.uniform(600.0, 1.4 * eta_local))

        event_observed = true_failure_hours <= censor_hours
        duration = true_failure_hours if event_observed else censor_hours
        failure_type = (
            rng.choices(FAILURE_TYPES, weights=FAILURE_TYPE_WEIGHTS, k=1)[0]
            if event_observed
            else None
        )

        profiles.append(
            EquipmentProfile(
                equipment_id=f"EQ-{i:04d}",
                equipment_type=equipment_type,
                model=model,
                faena=faena,
                manufacture_year=manufacture_year,
                install_date=install_date,
                hours_in_current_cycle=round(duration, 2),
                event_observed=event_observed,
                true_failure_hours=round(true_failure_hours, 2),
                failure_type=failure_type,
            )
        )

    return profiles


def build_equipment_metadata(
    profiles: list[EquipmentProfile], reference_date: date
) -> pl.DataFrame:
    """Tabla publica de metadatos de equipo (sin variables ocultas de simulacion)."""
    return pl.DataFrame(
        {
            "equipment_id": [p.equipment_id for p in profiles],
            "equipment_type": [p.equipment_type for p in profiles],
            "model": [p.model for p in profiles],
            "faena": [p.faena for p in profiles],
            "manufacture_year": [p.manufacture_year for p in profiles],
            "install_date": [p.install_date for p in profiles],
            "reference_date": [reference_date for _ in profiles],
            "hours_in_current_cycle": [p.hours_in_current_cycle for p in profiles],
            "event_observed": [p.event_observed for p in profiles],
        }
    )


def build_maintenance_logs(profiles: list[EquipmentProfile], seed: int) -> pl.DataFrame:
    """Eventos de mantenimiento programado + falla no planificada (si aplica)."""
    rng = random.Random(seed + 1)
    rows: list[dict] = []
    log_id = 0

    for p in profiles:
        # Mantenciones programadas distribuidas en lo que va del ciclo actual.
        n_scheduled = max(1, int(p.hours_in_current_cycle // rng.randint(1800, 2600)))
        for _ in range(n_scheduled):
            event_hours = rng.uniform(0, p.hours_in_current_cycle)
            rows.append(
                {
                    "log_id": log_id,
                    "equipment_id": p.equipment_id,
                    "event_type": "mantenimiento_programado",
                    "component": rng.choice(MAINTENANCE_COMPONENTS),
                    "failure_type": None,
                    "event_timestamp": datetime.combine(p.install_date, datetime.min.time())
                    + timedelta(hours=event_hours),
                    "operating_hours_at_event": round(event_hours, 2),
                    "downtime_hours": round(rng.uniform(2, 12), 1),
                }
            )
            log_id += 1

        if p.event_observed:
            severity = {"falla_estructural": (48, 120), "sobrecalentamiento_motor": (24, 72)}
            downtime_range = severity.get(p.failure_type, (12, 48))
            rows.append(
                {
                    "log_id": log_id,
                    "equipment_id": p.equipment_id,
                    "event_type": "falla_no_planificada",
                    "component": p.failure_type,
                    "failure_type": p.failure_type,
                    "event_timestamp": datetime.combine(p.install_date, datetime.min.time())
                    + timedelta(hours=p.hours_in_current_cycle),
                    "operating_hours_at_event": p.hours_in_current_cycle,
                    "downtime_hours": round(rng.uniform(*downtime_range), 1),
                }
            )
            log_id += 1

    return pl.DataFrame(rows)


def _degradation_curve(life_fraction: np.ndarray, failure_type: str | None, kind: str) -> np.ndarray:
    """Curva de degradacion no lineal en funcion de la fraccion de vida consumida."""
    f = np.clip(life_fraction, 0.0, 1.0)
    boost = {
        ("sobrecalentamiento_motor", "temp"): 2.0,
        ("falla_rodamiento", "vibration"): 2.2,
        ("falla_estructural", "vibration"): 1.6,
        ("fuga_hidraulica", "pressure"): 2.0,
    }.get((failure_type, kind), 1.0)

    if kind == "temp":
        return boost * 25.0 * f**3
    if kind == "vibration":
        return boost * 6.0 * f**2.5
    if kind == "pressure":
        return -boost * 40.0 * f**2
    if kind == "rpm_instability":
        return 1.0 + 3.0 * f  # multiplica el std de RPM
    if kind == "fuel":
        return 1.0 + 0.25 * f  # multiplica el consumo base
    raise ValueError(kind)


def build_sensor_telemetry(
    profiles: list[EquipmentProfile],
    seed: int,
    target_readings: int = DEFAULT_TARGET_READINGS_PER_EQUIPMENT,
) -> pl.DataFrame:
    """Telemetria del ciclo de vida completo de cada equipo (0 hasta `hours_in_current_cycle`).

    La resolucion es adaptativa por equipo: si el ciclo dura menos que
    `target_readings` horas, se muestrea cada hora (alta frecuencia); si es
    mas largo, el intervalo crece para mantener el numero de lecturas
    acotado (`target_readings`), evitando que equipos con ciclos muy largos
    (miles de horas) disparen el volumen de datos. Esto permite que el RUL
    de entrenamiento cubra el ciclo completo, no solo una ventana reciente.
    """
    np_rng = np.random.default_rng(seed + 2)
    frames: list[pl.DataFrame] = []

    for p in profiles:
        n_readings = int(min(target_readings, max(2, round(p.hours_in_current_cycle))))
        hours = np.linspace(0.0, p.hours_in_current_cycle, n_readings)
        # T de referencia para la curva de degradacion: la falla real simulada
        # (oculta), o el propio ciclo si el equipo aun no muestra desgaste medible.
        t_ref = max(p.true_failure_hours, p.hours_in_current_cycle + 1.0)
        life_fraction = hours / t_ref

        type_cfg = EQUIPMENT_TYPES[p.equipment_type]

        engine_temp = (
            np_rng.normal(85.0, 3.0, n_readings)
            + _degradation_curve(life_fraction, p.failure_type, "temp")
        )
        vibration = np.clip(
            np_rng.normal(2.5, 0.4, n_readings)
            + _degradation_curve(life_fraction, p.failure_type, "vibration"),
            0.1,
            None,
        )
        hydraulic_pressure = np.clip(
            np_rng.normal(210.0, 10.0, n_readings)
            + _degradation_curve(life_fraction, p.failure_type, "pressure"),
            20.0,
            None,
        )
        rpm_std_mult = _degradation_curve(life_fraction, p.failure_type, "rpm_instability")
        rpm = np_rng.normal(type_cfg["rpm_mean"], type_cfg["rpm_std"] * rpm_std_mult, n_readings)
        fuel_mult = _degradation_curve(life_fraction, p.failure_type, "fuel")
        fuel = np.clip(
            np_rng.normal(type_cfg["fuel_mean_lph"], type_cfg["fuel_mean_lph"] * 0.05, n_readings)
            * fuel_mult,
            0.0,
            None,
        )

        timestamps = [
            datetime.combine(p.install_date, datetime.min.time()) + timedelta(hours=float(h))
            for h in hours
        ]

        frames.append(
            pl.DataFrame(
                {
                    "equipment_id": [p.equipment_id] * n_readings,
                    "timestamp": timestamps,
                    "operating_hours": hours.round(2),
                    "engine_temp_c": engine_temp.round(2),
                    "vibration_rms_mm_s": vibration.round(3),
                    "hydraulic_pressure_bar": hydraulic_pressure.round(2),
                    "rpm": rpm.round(1),
                    "fuel_consumption_lph": fuel.round(2),
                }
            )
        )

    return pl.concat(frames)


def save_processed_tables(
    equipment_metadata: pl.DataFrame,
    sensor_telemetry: pl.DataFrame,
    maintenance_logs: pl.DataFrame,
) -> None:
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    equipment_metadata.write_parquet(PROCESSED_DIR / "equipment_metadata.parquet")
    sensor_telemetry.write_parquet(PROCESSED_DIR / "sensor_telemetry.parquet")
    maintenance_logs.write_parquet(PROCESSED_DIR / "maintenance_logs.parquet")


def generate_mining_dataset(
    n_equipment: int = 520,
    seed: int = 42,
    reference_date: date | None = None,
    target_readings: int = DEFAULT_TARGET_READINGS_PER_EQUIPMENT,
) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Genera las tres tablas relacionales sinteticas completas."""
    reference_date = reference_date or date.today()
    profiles = _build_profiles(n_equipment, seed, reference_date)

    equipment_metadata = build_equipment_metadata(profiles, reference_date)
    maintenance_logs = build_maintenance_logs(profiles, seed)
    sensor_telemetry = build_sensor_telemetry(profiles, seed, target_readings=target_readings)

    return equipment_metadata, sensor_telemetry, maintenance_logs


def main() -> None:
    equipment_metadata, sensor_telemetry, maintenance_logs = generate_mining_dataset()
    save_processed_tables(equipment_metadata, sensor_telemetry, maintenance_logs)

    n_events = int(equipment_metadata["event_observed"].sum())
    print(f"Equipos generados: {equipment_metadata.height}")
    print(f"  - Fallas observadas (event=1): {n_events}")
    print(f"  - Censurados (event=0, aun operando): {equipment_metadata.height - n_events}")
    print(f"Lecturas de telemetria: {sensor_telemetry.height}")
    print(f"Eventos de mantenimiento/falla: {maintenance_logs.height}")
    print(f"\nArchivos guardados en: {PROCESSED_DIR}")


if __name__ == "__main__":
    main()
