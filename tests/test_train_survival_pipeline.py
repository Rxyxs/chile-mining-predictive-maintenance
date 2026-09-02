from datetime import date

import pytest

from src.data.mining_data_generator import generate_mining_dataset
from src.features.engineering import engineer_features
from src.models.train_survival_pipeline import (
    build_failure_classification_table,
    build_rul_training_table,
    build_survival_table,
    compute_shap_importance_multiclass,
    equipment_train_test_split,
    fit_survival_model,
    train_failure_classifier,
    train_rul_model,
)


@pytest.fixture(scope="module")
def pipeline_inputs():
    equipment_metadata, sensor_telemetry, maintenance_logs = generate_mining_dataset(
        n_equipment=60, seed=11, reference_date=date(2026, 8, 20), target_readings=150
    )
    features = engineer_features(sensor_telemetry)
    return equipment_metadata, features, maintenance_logs


def test_equipment_train_test_split_is_disjoint_and_covers_all():
    ids = [f"EQ-{i:04d}" for i in range(40)]
    train_ids, test_ids = equipment_train_test_split(ids, test_size=0.25, seed=1)
    assert train_ids.isdisjoint(test_ids)
    assert train_ids | test_ids == set(ids)
    assert len(test_ids) == 10


def test_survival_model_c_index_in_valid_range(pipeline_inputs):
    equipment_metadata, _, _ = pipeline_inputs
    all_ids = equipment_metadata["equipment_id"].to_list()
    train_ids, test_ids = equipment_train_test_split(all_ids, test_size=0.3, seed=42)

    survival_df = build_survival_table(equipment_metadata)
    _, c_index = fit_survival_model(survival_df, train_ids, test_ids)
    assert 0.0 <= c_index <= 1.0


def test_rul_training_labels_are_non_negative(pipeline_inputs):
    equipment_metadata, features, _ = pipeline_inputs
    rul_df = build_rul_training_table(features, equipment_metadata)
    assert rul_df.height > 0
    assert (rul_df["rul_hours"] >= 0).all()


def test_rul_model_produces_finite_mae(pipeline_inputs):
    equipment_metadata, features, _ = pipeline_inputs
    rul_df = build_rul_training_table(features, equipment_metadata)

    failed_ids = equipment_metadata.filter(equipment_metadata["event_observed"])["equipment_id"].to_list()
    train_ids, test_ids = equipment_train_test_split(failed_ids, test_size=0.3, seed=42)

    _, mae, X_test = train_rul_model(rul_df, train_ids, test_ids)
    assert mae >= 0
    assert mae == mae  # not NaN
    assert len(X_test) > 0


def test_failure_classifier_accuracy_in_valid_range(pipeline_inputs):
    equipment_metadata, features, maintenance_logs = pipeline_inputs
    clf_df = build_failure_classification_table(features, equipment_metadata, maintenance_logs)

    failed_ids = equipment_metadata.filter(equipment_metadata["event_observed"])["equipment_id"].to_list()
    train_ids, test_ids = equipment_train_test_split(failed_ids, test_size=0.3, seed=42)

    _, accuracy, f1, X_test = train_failure_classifier(clf_df, train_ids, test_ids)
    assert 0.0 <= accuracy <= 1.0
    assert 0.0 <= f1 <= 1.0
    assert len(X_test) > 0


def test_failure_classification_table_one_row_per_failed_equipment(pipeline_inputs):
    equipment_metadata, features, maintenance_logs = pipeline_inputs
    clf_df = build_failure_classification_table(features, equipment_metadata, maintenance_logs)
    assert clf_df["equipment_id"].n_unique() == clf_df.height


def test_shap_importance_multiclass_matches_model_classes(pipeline_inputs):
    equipment_metadata, features, maintenance_logs = pipeline_inputs
    clf_df = build_failure_classification_table(features, equipment_metadata, maintenance_logs)

    failed_ids = equipment_metadata.filter(equipment_metadata["event_observed"])["equipment_id"].to_list()
    train_ids, test_ids = equipment_train_test_split(failed_ids, test_size=0.3, seed=42)

    clf_model, _, _, X_test = train_failure_classifier(clf_df, train_ids, test_ids)
    global_importance, per_class_importance = compute_shap_importance_multiclass(clf_model, X_test)

    assert set(per_class_importance.columns) == set(clf_model.classes_)
    assert set(global_importance["feature"]) <= set(X_test.columns)
    assert (global_importance["mean_abs_shap"] >= 0).all()
    assert (per_class_importance.to_numpy() >= 0).all()
