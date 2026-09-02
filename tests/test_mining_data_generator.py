from datetime import date

import pytest

from src.data.mining_data_generator import (
    EQUIPMENT_TYPES,
    FAENAS,
    generate_mining_dataset,
)


@pytest.fixture(scope="module")
def dataset():
    return generate_mining_dataset(
        n_equipment=40, seed=7, reference_date=date(2026, 8, 20), target_readings=120
    )


def test_equipment_metadata_row_count(dataset):
    equipment_metadata, _, _ = dataset
    assert equipment_metadata.height == 40


def test_equipment_metadata_valid_categories(dataset):
    equipment_metadata, _, _ = dataset
    assert set(equipment_metadata["equipment_type"].unique()) <= set(EQUIPMENT_TYPES)
    assert set(equipment_metadata["faena"].unique()) <= set(FAENAS)


def test_survival_clock_is_non_negative(dataset):
    equipment_metadata, _, _ = dataset
    assert (equipment_metadata["hours_in_current_cycle"] >= 0).all()


def test_telemetry_operating_hours_within_equipment_cycle(dataset):
    equipment_metadata, sensor_telemetry, _ = dataset
    joined = sensor_telemetry.join(
        equipment_metadata.select(["equipment_id", "hours_in_current_cycle"]), on="equipment_id"
    )
    assert (joined["operating_hours"] <= joined["hours_in_current_cycle"] + 1e-6).all()
    assert (joined["operating_hours"] >= 0).all()


def test_every_equipment_has_telemetry(dataset):
    equipment_metadata, sensor_telemetry, _ = dataset
    equipment_with_telemetry = set(sensor_telemetry["equipment_id"].unique())
    assert equipment_with_telemetry == set(equipment_metadata["equipment_id"])


def test_failed_equipment_has_failure_log(dataset):
    equipment_metadata, _, maintenance_logs = dataset
    failed_ids = set(
        equipment_metadata.filter(equipment_metadata["event_observed"])["equipment_id"]
    )
    logged_failures = set(
        maintenance_logs.filter(maintenance_logs["event_type"] == "falla_no_planificada")[
            "equipment_id"
        ]
    )
    assert failed_ids == logged_failures


def test_censored_equipment_has_no_failure_log(dataset):
    equipment_metadata, _, maintenance_logs = dataset
    censored_ids = set(
        equipment_metadata.filter(~equipment_metadata["event_observed"])["equipment_id"]
    )
    logged_failures = set(
        maintenance_logs.filter(maintenance_logs["event_type"] == "falla_no_planificada")[
            "equipment_id"
        ]
    )
    assert censored_ids.isdisjoint(logged_failures)


def test_every_equipment_has_scheduled_maintenance(dataset):
    _, _, maintenance_logs = dataset
    scheduled = maintenance_logs.filter(
        maintenance_logs["event_type"] == "mantenimiento_programado"
    )
    assert scheduled.height > 0
