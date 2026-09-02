"""Extraccion de features de degradacion temporal sobre `sensor_telemetry`.

Combina expresiones nativas de Polars (rolling stats, deltas, varianza
acumulada -- todas vectorizadas y evaluadas por equipo) con una
transformada de Fourier aplicada en ventanas deslizantes sobre la senal de
vibracion, para capturar componentes periodicas asociadas a defectos de
rodamientos / desbalance mecanico.
"""

from __future__ import annotations

import numpy as np
import polars as pl

SHORT_WINDOW = 6
MED_WINDOW = 24
LONG_WINDOW = 168
FFT_WINDOW = 24

SENSOR_COLUMNS = [
    "engine_temp_c",
    "vibration_rms_mm_s",
    "hydraulic_pressure_bar",
    "rpm",
    "fuel_consumption_lph",
]

FEATURE_COLUMNS = [
    "engine_temp_roll_mean_short",
    "engine_temp_roll_mean_med",
    "engine_temp_roll_std_med",
    "engine_temp_delta_long",
    "vibration_roll_mean_short",
    "vibration_roll_mean_med",
    "vibration_roll_std_med",
    "vibration_cum_var",
    "vibration_fft_dominant_amp",
    "vibration_fft_spectral_energy",
    "hydraulic_pressure_roll_mean_med",
    "hydraulic_pressure_delta_long",
    "rpm_roll_std_med",
    "fuel_consumption_roll_mean_med",
]


def _rolling_and_delta_features(telemetry: pl.DataFrame) -> pl.DataFrame:
    df = telemetry.sort(["equipment_id", "timestamp"])

    return df.with_columns(
        [
            pl.col("engine_temp_c")
            .rolling_mean(window_size=SHORT_WINDOW, min_samples=1)
            .over("equipment_id")
            .alias("engine_temp_roll_mean_short"),
            pl.col("engine_temp_c")
            .rolling_mean(window_size=MED_WINDOW, min_samples=1)
            .over("equipment_id")
            .alias("engine_temp_roll_mean_med"),
            pl.col("engine_temp_c")
            .rolling_std(window_size=MED_WINDOW, min_samples=2)
            .over("equipment_id")
            .alias("engine_temp_roll_std_med"),
            (
                pl.col("engine_temp_c")
                - pl.col("engine_temp_c")
                .rolling_mean(window_size=LONG_WINDOW, min_samples=1)
                .over("equipment_id")
            ).alias("engine_temp_delta_long"),
            pl.col("vibration_rms_mm_s")
            .rolling_mean(window_size=SHORT_WINDOW, min_samples=1)
            .over("equipment_id")
            .alias("vibration_roll_mean_short"),
            pl.col("vibration_rms_mm_s")
            .rolling_mean(window_size=MED_WINDOW, min_samples=1)
            .over("equipment_id")
            .alias("vibration_roll_mean_med"),
            pl.col("vibration_rms_mm_s")
            .rolling_std(window_size=MED_WINDOW, min_samples=2)
            .over("equipment_id")
            .alias("vibration_roll_std_med"),
            pl.col("hydraulic_pressure_bar")
            .rolling_mean(window_size=MED_WINDOW, min_samples=1)
            .over("equipment_id")
            .alias("hydraulic_pressure_roll_mean_med"),
            (
                pl.col("hydraulic_pressure_bar")
                - pl.col("hydraulic_pressure_bar")
                .rolling_mean(window_size=LONG_WINDOW, min_samples=1)
                .over("equipment_id")
            ).alias("hydraulic_pressure_delta_long"),
            pl.col("rpm")
            .rolling_std(window_size=MED_WINDOW, min_samples=2)
            .over("equipment_id")
            .alias("rpm_roll_std_med"),
            pl.col("fuel_consumption_lph")
            .rolling_mean(window_size=MED_WINDOW, min_samples=1)
            .over("equipment_id")
            .alias("fuel_consumption_roll_mean_med"),
        ]
    )


def _cumulative_variance(df: pl.DataFrame) -> pl.DataFrame:
    """Varianza acumulada (expanding) de la vibracion por equipo, via sumas acumuladas."""
    cum_n = pl.int_range(1, pl.len() + 1).over("equipment_id")
    cum_sum = pl.col("vibration_rms_mm_s").cum_sum().over("equipment_id")
    cum_sum_sq = (pl.col("vibration_rms_mm_s") ** 2).cum_sum().over("equipment_id")
    mean = cum_sum / cum_n
    var = (cum_sum_sq / cum_n) - mean**2

    return df.with_columns(var.clip(lower_bound=0).alias("vibration_cum_var"))


def _fft_features_for_group(group: pl.DataFrame) -> pl.DataFrame:
    """Amplitud dominante y energia espectral de la vibracion en ventanas deslizantes."""
    values = group["vibration_rms_mm_s"].to_numpy()
    n = len(values)
    dominant_amp = np.full(n, np.nan)
    spectral_energy = np.full(n, np.nan)

    if n >= FFT_WINDOW:
        windows = np.lib.stride_tricks.sliding_window_view(values, FFT_WINDOW)
        spectrum = np.fft.rfft(windows, axis=1)
        magnitude = np.abs(spectrum)[:, 1:]  # excluye componente DC
        dominant = magnitude.max(axis=1)
        energy = (magnitude**2).sum(axis=1) / FFT_WINDOW
        dominant_amp[FFT_WINDOW - 1 :] = dominant
        spectral_energy[FFT_WINDOW - 1 :] = energy

    return group.with_columns(
        [
            pl.Series("vibration_fft_dominant_amp", dominant_amp).fill_nan(None),
            pl.Series("vibration_fft_spectral_energy", spectral_energy).fill_nan(None),
        ]
    )


def _fft_features(df: pl.DataFrame) -> pl.DataFrame:
    return df.group_by("equipment_id", maintain_order=True).map_groups(_fft_features_for_group)


def engineer_features(telemetry: pl.DataFrame) -> pl.DataFrame:
    """Aplica el pipeline completo de feature engineering sobre telemetria cruda.

    Devuelve el DataFrame original con las columnas de `FEATURE_COLUMNS`
    anadidas, ordenado por (equipment_id, timestamp).
    """
    df = _rolling_and_delta_features(telemetry)
    df = _cumulative_variance(df)
    df = _fft_features(df)
    return df.sort(["equipment_id", "timestamp"])


def main() -> None:
    from pathlib import Path

    processed_dir = Path(__file__).resolve().parents[2] / "data" / "processed"
    telemetry = pl.read_parquet(processed_dir / "sensor_telemetry.parquet")

    features = engineer_features(telemetry)
    features.write_parquet(processed_dir / "telemetry_features.parquet")

    print(f"Filas procesadas: {features.height}")
    print(f"Columnas de features generadas: {len(FEATURE_COLUMNS)}")
    print(features.select(["equipment_id", "timestamp", *FEATURE_COLUMNS]).tail(5))


if __name__ == "__main__":
    main()
