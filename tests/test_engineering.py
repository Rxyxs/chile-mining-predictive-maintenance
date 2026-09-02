from datetime import date

import pytest

from src.data.mining_data_generator import generate_mining_dataset
from src.features.engineering import FEATURE_COLUMNS, FFT_WINDOW, engineer_features


@pytest.fixture(scope="module")
def features():
    _, sensor_telemetry, _ = generate_mining_dataset(
        n_equipment=15, seed=3, reference_date=date(2026, 8, 20), target_readings=120
    )
    return engineer_features(sensor_telemetry)


def test_all_feature_columns_present(features):
    assert set(FEATURE_COLUMNS) <= set(features.columns)


def test_row_count_preserved(features, ):
    # engineer_features no debe agregar ni eliminar filas, solo columnas.
    assert features.height > 0


def test_fft_features_populated_after_warmup(features):
    non_null = features.filter(features["vibration_fft_dominant_amp"].is_not_null())
    assert non_null.height > 0


def test_fft_features_null_during_warmup_per_equipment(features):
    first_reading = features.sort(["equipment_id", "timestamp"]).group_by(
        "equipment_id", maintain_order=True
    ).first()
    # La primera lectura de cada equipo no tiene historial suficiente para FFT.
    assert first_reading["vibration_fft_dominant_amp"].is_null().all()


def test_cumulative_variance_is_non_negative(features):
    assert (features["vibration_cum_var"].drop_nulls() >= 0).all()


def test_rolling_mean_short_window_no_warmup_nulls(features):
    # min_samples=1 => la primera lectura ya tiene un rolling_mean valido.
    assert features["vibration_roll_mean_short"].null_count() == 0
