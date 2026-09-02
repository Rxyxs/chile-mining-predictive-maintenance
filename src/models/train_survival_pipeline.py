"""Pipeline hibrido de mantenimiento predictivo:

1. Survival Analysis (CoxPH, lifelines) sobre `equipment_metadata` para
   estimar riesgo de falla con datos censurados -- evaluado con C-Index.
2. Regresion de Vida Util Restante (RUL, LightGBM) sobre lecturas de
   telemetria de equipos con falla observada -- evaluada con MAE en horas.
3. Clasificacion multiclase del tipo de falla mas probable (LightGBM)
   sobre la ultima lectura conocida de cada equipo fallado.
4. Explicabilidad SHAP del modelo de RUL, para uso de mecanicos e
   ingenieros de mantenimiento en faena.

Los splits de train/test se hacen siempre a nivel de `equipment_id` (nunca
por fila), para no filtrar informacion entre lecturas correlacionadas del
mismo equipo.
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
import shap
from lifelines import CoxPHFitter
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error

from src.features.engineering import FEATURE_COLUMNS

PROCESSED_DIR = Path(__file__).resolve().parents[2] / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"

MODEL_FEATURE_COLUMNS = FEATURE_COLUMNS + ["operating_hours", "age_years"]
CATEGORICAL_COLUMNS = ["equipment_type", "faena"]


def equipment_train_test_split(
    equipment_ids: list[str], test_size: float = 0.25, seed: int = 42
) -> tuple[set[str], set[str]]:
    """Split reproducible a nivel de equipo (evita fuga entre lecturas del mismo equipo)."""
    ids = sorted(equipment_ids)
    rng = random.Random(seed)
    rng.shuffle(ids)
    n_test = max(1, int(len(ids) * test_size))
    return set(ids[n_test:]), set(ids[:n_test])


def _equipment_covariates(equipment_metadata: pl.DataFrame) -> pl.DataFrame:
    return equipment_metadata.with_columns(
        (
            (pl.col("reference_date") - pl.col("install_date")).dt.total_days() / 365.25
        ).alias("age_years")
    ).select(
        ["equipment_id", "equipment_type", "faena", "age_years", "hours_in_current_cycle", "event_observed"]
    )


def build_survival_table(equipment_metadata: pl.DataFrame) -> pl.DataFrame:
    """Tabla a nivel de equipo para CoxPH: duration, event y covariables numericas."""
    df = equipment_metadata.with_columns(
        [
            (
                (pl.col("reference_date") - pl.col("install_date")).dt.total_days() / 365.25
            ).alias("age_years"),
            pl.col("hours_in_current_cycle").alias("duration"),
            pl.col("event_observed").cast(pl.Int8).alias("event"),
        ]
    ).select(["equipment_id", "equipment_type", "faena", "age_years", "duration", "event"])

    return df.to_dummies(columns=["equipment_type", "faena"], drop_first=True)


def fit_survival_model(
    survival_df: pl.DataFrame, train_ids: set[str], test_ids: set[str]
) -> tuple[CoxPHFitter, float]:
    train_pdf = survival_df.filter(pl.col("equipment_id").is_in(train_ids)).drop("equipment_id").to_pandas()
    test_pdf = survival_df.filter(pl.col("equipment_id").is_in(test_ids)).drop("equipment_id").to_pandas()

    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_pdf, duration_col="duration", event_col="event")
    c_index = cph.score(test_pdf, scoring_method="concordance_index")
    return cph, float(c_index)


def build_rul_training_table(features: pl.DataFrame, equipment_metadata: pl.DataFrame) -> pl.DataFrame:
    """Filas de telemetria de equipos FALLADOS, con RUL real conocido como etiqueta.

    Cubre el ciclo de vida completo de cada equipo fallado (la telemetria ya
    no esta acotada a una ventana reciente -- ver
    `mining_data_generator.build_sensor_telemetry`), por lo que `rul_hours`
    va desde 0 (en la falla) hasta miles de horas (en equipos jovenes dentro
    de su ciclo).
    """
    meta = _equipment_covariates(equipment_metadata)
    df = features.join(meta, on="equipment_id", how="inner")
    df = df.filter(pl.col("event_observed"))
    df = df.with_columns(
        (pl.col("hours_in_current_cycle") - pl.col("operating_hours")).clip(lower_bound=0).alias("rul_hours")
    )
    return df.drop_nulls(subset=FEATURE_COLUMNS)


def build_failure_classification_table(
    features: pl.DataFrame, equipment_metadata: pl.DataFrame, maintenance_logs: pl.DataFrame
) -> pl.DataFrame:
    """Ultima lectura conocida de cada equipo fallado + tipo de falla real (etiqueta)."""
    meta = _equipment_covariates(equipment_metadata)
    failure_events = maintenance_logs.filter(pl.col("event_type") == "falla_no_planificada").select(
        ["equipment_id", "failure_type"]
    )
    last_readings = features.sort(["equipment_id", "timestamp"]).group_by("equipment_id", maintain_order=True).last()

    df = last_readings.join(meta, on="equipment_id", how="inner").join(failure_events, on="equipment_id", how="inner")
    df = df.filter(pl.col("event_observed"))
    return df.drop_nulls(subset=FEATURE_COLUMNS)


def prepare_model_frame(df: pl.DataFrame) -> pd.DataFrame:
    pdf = df.select(MODEL_FEATURE_COLUMNS + CATEGORICAL_COLUMNS).to_pandas()
    for col in CATEGORICAL_COLUMNS:
        pdf[col] = pdf[col].astype("category")
    return pdf


def train_rul_model(
    rul_df: pl.DataFrame, train_ids: set[str], test_ids: set[str]
) -> tuple[lgb.LGBMRegressor, float, pd.DataFrame]:
    train_df = rul_df.filter(pl.col("equipment_id").is_in(train_ids))
    test_df = rul_df.filter(pl.col("equipment_id").is_in(test_ids))

    X_train, y_train = prepare_model_frame(train_df), train_df["rul_hours"].to_numpy()
    X_test, y_test = prepare_model_frame(test_df), test_df["rul_hours"].to_numpy()

    model = lgb.LGBMRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    mae = mean_absolute_error(y_test, model.predict(X_test))
    return model, float(mae), X_test


def train_failure_classifier(
    clf_df: pl.DataFrame, train_ids: set[str], test_ids: set[str]
) -> tuple[lgb.LGBMClassifier, float, float, pd.DataFrame]:
    train_df = clf_df.filter(pl.col("equipment_id").is_in(train_ids))
    test_df = clf_df.filter(pl.col("equipment_id").is_in(test_ids))

    X_train, y_train = prepare_model_frame(train_df), train_df["failure_type"].to_numpy()
    X_test, y_test = prepare_model_frame(test_df), test_df["failure_type"].to_numpy()

    model = lgb.LGBMClassifier(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=6,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
        verbosity=-1,
    )
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    accuracy = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")
    return model, float(accuracy), float(f1), X_test


def compute_shap_importance(
    model: lgb.LGBMRegressor, X_sample: pd.DataFrame, top_n: int = 15
) -> tuple[pd.DataFrame, np.ndarray]:
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    importance = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return importance, shap_values


def save_shap_summary_plot(shap_values: np.ndarray, X_sample: pd.DataFrame, path: Path) -> None:
    import matplotlib.pyplot as plt

    shap.summary_plot(shap_values, X_sample, plot_type="bar", show=False)
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()


def _stack_multiclass_shap(shap_values, n_classes: int) -> np.ndarray:
    """Normaliza la salida de TreeExplainer.shap_values() a (n_classes, n_samples, n_features).

    Distintas versiones de `shap` devuelven una lista de arrays (una por
    clase) o un unico ndarray (n_samples, n_features, n_classes) -- se
    manejan ambos formatos.
    """
    if isinstance(shap_values, list):
        return np.stack([np.abs(sv) for sv in shap_values], axis=0)
    values = np.abs(shap_values)
    if values.ndim == 3 and values.shape[-1] == n_classes:
        return values.transpose(2, 0, 1)
    raise ValueError(f"Formato de shap_values no reconocido: shape={values.shape}")


def compute_shap_importance_multiclass(
    model: lgb.LGBMClassifier, X_sample: pd.DataFrame, top_n: int = 15
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Importancia SHAP del clasificador de tipo de falla, global y por clase.

    Devuelve (importancia_global_top_n, importancia_por_clase_todas_las_features).
    """
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X_sample)
    abs_per_class = _stack_multiclass_shap(shap_values, n_classes=len(model.classes_))  # (n_classes, n_samples, n_features)

    mean_abs_per_class = abs_per_class.mean(axis=1)  # (n_classes, n_features)
    per_class_importance = pd.DataFrame(
        mean_abs_per_class.T, index=X_sample.columns, columns=model.classes_
    )

    global_importance = (
        pd.DataFrame({"feature": X_sample.columns, "mean_abs_shap": mean_abs_per_class.mean(axis=0)})
        .sort_values("mean_abs_shap", ascending=False)
        .head(top_n)
        .reset_index(drop=True)
    )
    return global_importance, per_class_importance


def save_shap_classifier_plot(per_class_importance: pd.DataFrame, top_features: list[str], path: Path) -> None:
    import matplotlib.pyplot as plt

    subset = per_class_importance.loc[top_features].iloc[::-1]
    fig, ax = plt.subplots(figsize=(9, max(4, 0.4 * len(top_features))))
    subset.plot(kind="barh", ax=ax, width=0.75)
    ax.set_xlabel("Mean |SHAP value|")
    ax.set_title("Importancia SHAP por tipo de falla (top features)")
    ax.legend(title="Tipo de falla", bbox_to_anchor=(1.02, 1), loc="upper left")
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close(fig)


def main() -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    equipment_metadata = pl.read_parquet(PROCESSED_DIR / "equipment_metadata.parquet")
    maintenance_logs = pl.read_parquet(PROCESSED_DIR / "maintenance_logs.parquet")
    features = pl.read_parquet(PROCESSED_DIR / "telemetry_features.parquet")

    all_ids = equipment_metadata["equipment_id"].to_list()
    train_ids, test_ids = equipment_train_test_split(all_ids, test_size=0.25, seed=42)

    survival_df = build_survival_table(equipment_metadata)
    cph, c_index = fit_survival_model(survival_df, train_ids, test_ids)
    print(f"[Survival] C-Index (CoxPH, holdout): {c_index:.4f}")

    failed_ids = equipment_metadata.filter(pl.col("event_observed"))["equipment_id"].to_list()
    failed_train_ids, failed_test_ids = equipment_train_test_split(failed_ids, test_size=0.25, seed=42)

    rul_df = build_rul_training_table(features, equipment_metadata)
    rul_model, mae, X_test_rul = train_rul_model(rul_df, failed_train_ids, failed_test_ids)
    print(f"[RUL] MAE (LightGBM, holdout): {mae:.1f} horas")

    clf_df = build_failure_classification_table(features, equipment_metadata, maintenance_logs)
    clf_model, accuracy, f1, X_test_clf = train_failure_classifier(clf_df, failed_train_ids, failed_test_ids)
    print(f"[Clasificacion falla] Accuracy: {accuracy:.3f} | F1-macro: {f1:.3f}")

    shap_sample = X_test_rul.sample(n=min(200, len(X_test_rul)), random_state=42)
    importance, shap_values = compute_shap_importance(rul_model, shap_sample)
    importance.to_csv(PROCESSED_DIR / "shap_rul_importance.csv", index=False)
    save_shap_summary_plot(shap_values, shap_sample, MODELS_DIR / "shap_rul_summary.png")
    print("\n[SHAP] Top features para prediccion de RUL:")
    print(importance.to_string(index=False))

    clf_shap_sample = X_test_clf.sample(n=min(200, len(X_test_clf)), random_state=42)
    clf_importance, clf_per_class_importance = compute_shap_importance_multiclass(clf_model, clf_shap_sample)
    clf_importance.to_csv(PROCESSED_DIR / "shap_failure_classifier_importance.csv", index=False)
    clf_per_class_importance.to_csv(PROCESSED_DIR / "shap_failure_classifier_importance_by_class.csv")
    save_shap_classifier_plot(
        clf_per_class_importance, clf_importance["feature"].tolist(), MODELS_DIR / "shap_failure_classifier_summary.png"
    )
    print("\n[SHAP] Top features para clasificacion de tipo de falla:")
    print(clf_importance.to_string(index=False))

    joblib.dump(cph, MODELS_DIR / "coxph_model.joblib")
    joblib.dump(rul_model, MODELS_DIR / "rul_lightgbm.joblib")
    joblib.dump(clf_model, MODELS_DIR / "failure_classifier_lightgbm.joblib")

    metrics = {
        "n_equipment": equipment_metadata.height,
        "n_events_observed": int(equipment_metadata["event_observed"].sum()),
        "c_index_survival": round(c_index, 4),
        "mae_rul_hours": round(mae, 2),
        "accuracy_failure_type": round(accuracy, 4),
        "f1_macro_failure_type": round(f1, 4),
    }
    with open(PROCESSED_DIR / "metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

    print(f"\nModelos guardados en: {MODELS_DIR}")
    print(f"Metricas guardadas en: {PROCESSED_DIR / 'metrics.json'}")


if __name__ == "__main__":
    main()
