from datetime import datetime, timezone

import pytest

from src.models.metrics_db import (
    load_model_comparison,
    persist_activation_comparison,
    persist_model_comparison,
)

SAMPLE_METRICS = {
    "n_equipment": 520,
    "n_events_observed": 341,
    "c_index_survival": 0.6246,
    "mae_rul_hours": 512.96,
    "accuracy_failure_type": 1.0,
    "f1_macro_failure_type": 1.0,
    "multi_task_pytorch": {
        "mae_rul_hours": 592.31,
        "accuracy_failure_type": 0.714,
        "f1_macro_failure_type": 0.697,
        "n_train_rows": 1000,
        "activation": "relu",
        "loss_history": {"epoch": [1, 2], "train_total": [1.2, 1.0]},
    },
}

SAMPLE_ACTIVATION_RESULTS = {
    "relu": {"mae_rul_hours": 592.31, "accuracy_failure_type": 0.714, "activation": "relu"},
    "gelu": {"mae_rul_hours": 580.10, "accuracy_failure_type": 0.723, "activation": "gelu"},
    "swish": {"mae_rul_hours": 575.44, "accuracy_failure_type": 0.731, "activation": "swish"},
}


@pytest.fixture
def db_path(tmp_path):
    return tmp_path / "metrics_test.duckdb"


def test_persist_model_comparison_flattens_flat_and_nested_keys(db_path):
    run_ts = datetime(2026, 8, 30, tzinfo=timezone.utc)
    persist_model_comparison(SAMPLE_METRICS, db_path=db_path, run_ts=run_ts)

    df = load_model_comparison(db_path=db_path)
    assert len(df) > 0

    coxph_rows = df[df["model"] == "coxph"]
    assert coxph_rows.iloc[0]["task"] == "survival"
    assert coxph_rows.iloc[0]["metric"] == "c_index_survival"
    assert coxph_rows.iloc[0]["value"] == pytest.approx(0.6246)

    lightgbm_rows = df[df["model"] == "lightgbm"]
    assert set(lightgbm_rows["metric"]) == {"mae_rul_hours", "accuracy_failure_type", "f1_macro_failure_type"}

    pytorch_rows = df[df["model"] == "multi_task_pytorch"]
    # loss_history (dict) and activation (str) must be dropped; numeric scalars are kept
    assert {"mae_rul_hours", "accuracy_failure_type", "f1_macro_failure_type"} <= set(pytorch_rows["metric"])
    assert "loss_history" not in set(pytorch_rows["metric"])
    assert "activation" not in set(pytorch_rows["metric"])


def test_persist_model_comparison_is_append_only(db_path):
    persist_model_comparison(SAMPLE_METRICS, db_path=db_path)
    persist_model_comparison(SAMPLE_METRICS, db_path=db_path)

    df = load_model_comparison(db_path=db_path)
    # two runs inserted -> row count doubles, nothing overwritten
    assert len(df) == 2 * len(df[df["run_ts"] == df["run_ts"].iloc[0]])


def test_persist_activation_comparison_writes_one_row_per_metric_per_activation(db_path):
    persist_activation_comparison(SAMPLE_ACTIVATION_RESULTS, db_path=db_path)

    import duckdb

    con = duckdb.connect(str(db_path))
    try:
        rows = con.execute("SELECT activation, metric, value FROM activation_comparison ORDER BY activation").fetchall()
    finally:
        con.close()

    activations_seen = {row[0] for row in rows}
    assert activations_seen == {"relu", "gelu", "swish"}
    # activation (string) column must not leak in as a numeric metric row
    assert all(row[1] != "activation" for row in rows)
