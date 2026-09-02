from datetime import date

import polars as pl
import pytest

from src.data.mining_data_generator import FAILURE_TYPES, generate_mining_dataset
from src.features.engineering import engineer_features
from src.models.multi_task_net import NUMERIC_COLUMNS, MultiTaskDegradationNet, build_multi_task_table
from src.models.train_survival_pipeline import equipment_train_test_split
from multitask_pdm import (
    CONTINUOUS_FEATURE_NAMES,
    FEATURE_GROUPS,
    aggregate_grouped_shap,
    build_deep_explainers,
    explain_instance,
    to_continuous_input,
)


def test_feature_groups_cover_every_continuous_column_exactly_once():
    grouped_cols = [col for cols in FEATURE_GROUPS.values() for col in cols]
    assert sorted(grouped_cols) == sorted(CONTINUOUS_FEATURE_NAMES)
    assert len(grouped_cols) == len(set(grouped_cols))


def test_aggregate_grouped_shap_sums_embedding_dimensions():
    import numpy as np

    values = np.arange(len(CONTINUOUS_FEATURE_NAMES), dtype=float)
    aggregated = aggregate_grouped_shap(values)

    assert set(aggregated) == set(FEATURE_GROUPS)
    by_name = dict(zip(CONTINUOUS_FEATURE_NAMES, values))
    expected_equipment_type = sum(by_name[c] for c in FEATURE_GROUPS["equipment_type"])
    assert aggregated["equipment_type"] == pytest.approx(expected_equipment_type)


@pytest.fixture(scope="module")
def trained_tiny_model():
    equipment_metadata, sensor_telemetry, maintenance_logs = generate_mining_dataset(
        n_equipment=60, seed=11, reference_date=date(2026, 8, 20), target_readings=150
    )
    features = engineer_features(sensor_telemetry)
    mt_df = build_multi_task_table(features, equipment_metadata, maintenance_logs)
    ids = mt_df["equipment_id"].unique().to_list()
    train_ids, test_ids = equipment_train_test_split(ids, test_size=0.3, seed=42)

    from src.models.multi_task_net import _to_tensors, train_multi_task_model

    model, scaler, _ = train_multi_task_model(mt_df, train_ids, test_ids, epochs=2, batch_size=32)

    train_df = mt_df.filter(pl.col("equipment_id").is_in(train_ids))
    background_t = _to_tensors(train_df.sample(n=min(20, train_df.height), seed=42), scaler, fit_scaler=False)
    background_continuous = to_continuous_input(
        model, background_t["numeric"], background_t["equipment_type_idx"], background_t["faena_idx"]
    )
    return model, scaler, background_continuous, background_t


def test_to_continuous_input_has_expected_width(trained_tiny_model):
    _, _, background_continuous, _ = trained_tiny_model
    assert background_continuous.shape[1] == len(CONTINUOUS_FEATURE_NAMES)


def test_deep_explainers_explain_a_single_instance(trained_tiny_model):
    model, _, background_continuous, background_t = trained_tiny_model
    rul_explainer, failure_explainer = build_deep_explainers(model, background_continuous)

    one_numeric = background_t["numeric"][:1]
    one_eq_idx = background_t["equipment_type_idx"][:1]
    one_faena_idx = background_t["faena_idx"][:1]

    result = explain_instance(model, rul_explainer, failure_explainer, one_numeric, one_eq_idx, one_faena_idx)

    assert result["predicted_failure_type"] in FAILURE_TYPES
    assert set(result["shap_rul"]) == set(FEATURE_GROUPS)
    assert set(result["shap_failure_type"]) == set(FEATURE_GROUPS)
    assert abs(sum(result["failure_type_probabilities"].values()) - 1.0) < 1e-3
