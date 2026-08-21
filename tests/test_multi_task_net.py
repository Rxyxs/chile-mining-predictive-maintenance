from datetime import date

import pytest

from src.data.mining_data_generator import generate_mining_dataset
from src.features.engineering import engineer_features
from src.models.multi_task_net import build_multi_task_table, train_multi_task_model
from src.models.train_survival_pipeline import equipment_train_test_split


@pytest.fixture(scope="module")
def multi_task_table():
    equipment_metadata, sensor_telemetry, maintenance_logs = generate_mining_dataset(
        n_equipment=60, seed=11, reference_date=date(2026, 8, 20), target_readings=150
    )
    features = engineer_features(sensor_telemetry)
    return build_multi_task_table(features, equipment_metadata, maintenance_logs)


def test_multi_task_table_has_both_labels(multi_task_table):
    assert multi_task_table.height > 0
    assert "rul_hours" in multi_task_table.columns
    assert "failure_type" in multi_task_table.columns
    assert multi_task_table["failure_type"].null_count() == 0


def test_train_multi_task_model_produces_finite_metrics(multi_task_table):
    ids = multi_task_table["equipment_id"].unique().to_list()
    train_ids, test_ids = equipment_train_test_split(ids, test_size=0.3, seed=42)

    _, _, metrics = train_multi_task_model(multi_task_table, train_ids, test_ids, epochs=3, batch_size=32)

    assert metrics["mae_rul_hours"] >= 0
    assert metrics["mae_rul_hours"] == metrics["mae_rul_hours"]  # not NaN
    assert 0.0 <= metrics["accuracy_failure_type"] <= 1.0
    assert 0.0 <= metrics["f1_macro_failure_type"] <= 1.0
    assert metrics["n_train_rows"] > 0
    assert metrics["n_test_rows"] > 0
