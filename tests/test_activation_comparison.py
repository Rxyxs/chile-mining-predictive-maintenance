from datetime import date

import pytest
import torch

from src.data.mining_data_generator import generate_mining_dataset
from src.features.engineering import engineer_features
from src.models.activation_comparison import run_activation_comparison
from src.models.multi_task_net import ACTIVATIONS, MultiTaskDegradationNet, build_multi_task_table
from src.models.train_survival_pipeline import equipment_train_test_split


@pytest.fixture(scope="module")
def multi_task_table():
    equipment_metadata, sensor_telemetry, maintenance_logs = generate_mining_dataset(
        n_equipment=60, seed=11, reference_date=date(2026, 8, 20), target_readings=150
    )
    features = engineer_features(sensor_telemetry)
    return build_multi_task_table(features, equipment_metadata, maintenance_logs)


def test_all_activations_are_valid_torch_modules():
    for name, layer_cls in ACTIVATIONS.items():
        layer = layer_cls()
        out = layer(torch.tensor([-1.0, 0.0, 1.0]))
        assert out.shape == (3,)


def test_unknown_activation_raises():
    with pytest.raises(ValueError):
        MultiTaskDegradationNet(n_numeric_features=5, activation="tanh")


def test_gelu_and_swish_change_the_trunk_architecture():
    relu_net = MultiTaskDegradationNet(n_numeric_features=5, activation="relu")
    gelu_net = MultiTaskDegradationNet(n_numeric_features=5, activation="gelu")
    swish_net = MultiTaskDegradationNet(n_numeric_features=5, activation="swish")

    assert isinstance(relu_net.trunk[1], torch.nn.ReLU)
    assert isinstance(gelu_net.trunk[1], torch.nn.GELU)
    assert isinstance(swish_net.trunk[1], torch.nn.SiLU)


def test_run_activation_comparison_produces_finite_metrics_for_every_activation(multi_task_table):
    ids = multi_task_table["equipment_id"].unique().to_list()
    train_ids, test_ids = equipment_train_test_split(ids, test_size=0.3, seed=42)

    results = run_activation_comparison(multi_task_table, train_ids, test_ids, epochs=2)

    assert set(results.keys()) == set(ACTIVATIONS.keys())
    for activation_name, metrics in results.items():
        assert metrics["mae_rul_hours"] >= 0
        assert metrics["mae_rul_hours"] == metrics["mae_rul_hours"]  # not NaN
        assert 0.0 <= metrics["accuracy_failure_type"] <= 1.0
        assert metrics["activation"] == activation_name
        assert "loss_history" not in metrics
