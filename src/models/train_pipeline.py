"""Pipeline de mantenimiento predictivo sobre SCANIA Component X (datos reales).

1. Clasificador de riesgo (LightGBM multiclase, clases 0-4 del reto) con decision de
   mantenimiento por costo esperado, evaluado con el costo oficial en validacion y test.
2. Regresion del tiempo restante hasta la reparacion (solo camiones que fallaron).
3. Supervivencia (CoxPH con entrada tardia en un "landmark") a nivel de vehiculo.
4. Explicabilidad SHAP del modelo de tiempo restante.

Todas las particiones se hacen por vehiculo, nunca por fila. Validacion y test son los
conjuntos oficiales del reto (una lectura por camion, elegida al azar como la ultima).
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import polars as pl
from lifelines import CoxPHFitter
from lifelines.utils import concordance_index
from sklearn.metrics import roc_auc_score

from src.data import scania_loader as S
from src.features import engineering as E
from src.models import cost_decision as C

ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = PROCESSED_DIR / "models"
REPORTS_DIR = ROOT / "reports"

SPEC_COLUMNS = [f"Spec_{i}" for i in range(8)]
NEG_RATE = 0.05  # fraccion de lecturas clase 0 que se usa para entrenar (se compensa con pesos)
LANDMARK = 50.0  # tiempo (unidades del dataset) en que se fija el riesgo para el modelo Cox
SEED = 42
LGB_COMMON = dict(
    learning_rate=0.05, num_leaves=31, subsample=0.8, subsample_freq=1, colsample_bytree=0.3,
    min_child_samples=20, random_state=SEED, verbosity=-1, n_jobs=-1,
)


# ----------------------------------------------------------------------------- datos
def spec_categories(train_specs: pl.DataFrame) -> dict[str, list[str]]:
    return {c: sorted(train_specs[c].unique().to_list()) for c in SPEC_COLUMNS}


def with_specs(frame: pd.DataFrame, specs: pl.DataFrame, categories: dict[str, list[str]]) -> pd.DataFrame:
    """Une las especificaciones como columnas categoricas con categorias fijadas en train."""
    spec_pdf = specs.to_pandas().set_index("vehicle_id")
    out = frame.join(spec_pdf, on="vehicle_id", how="left")
    for c in SPEC_COLUMNS:
        out[c] = pd.Categorical(out[c], categories=categories[c])
    return out


def vehicle_split(vehicle_ids, strata, test_size: float = 0.2, seed: int = SEED):
    """Split reproducible por vehiculo, estratificado (para mantener la tasa de fallas)."""
    rng = np.random.default_rng(seed)
    ids = np.asarray(vehicle_ids)
    strata = np.asarray(strata)
    test = []
    for s in np.unique(strata):
        group = ids[strata == s]
        rng.shuffle(group)
        test.extend(group[: max(1, int(len(group) * test_size))])
    test_set = set(test)
    return {v for v in ids if v not in test_set}, test_set


def load_features(split: str, force: bool = False) -> pl.DataFrame:
    cache = PROCESSED_DIR / f"features_{split}.parquet"
    if cache.exists() and not force:
        return pl.read_parquet(cache)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    feats = E.engineer_features(S.read_readouts(split))
    feats.write_parquet(cache)
    return feats


def feature_columns(frame: pd.DataFrame) -> list[str]:
    return [c for c in frame.columns if c not in ("vehicle_id", "time_to_failure", "class_label", "duration", "event")]


# ------------------------------------------------------------------- clasificador
def sample_rows(labeled: pl.DataFrame, neg_rate: float = NEG_RATE, seed: int = SEED) -> pl.DataFrame:
    """Todas las lecturas con riesgo (clase > 0) y una fraccion de las sanas, con su peso."""
    rng = np.random.default_rng(seed)
    keep = (labeled["class_label"].to_numpy() > 0) | (rng.random(labeled.height) < neg_rate)
    sampled = labeled.filter(pl.Series(keep))
    weight = np.where(sampled["class_label"].to_numpy() > 0, 1.0, 1.0 / neg_rate)
    return sampled.with_columns(pl.Series("weight", weight))


def train_classifier(train_pdf: pd.DataFrame, hold_pdf: pd.DataFrame, cols: list[str]):
    model = lgb.LGBMClassifier(objective="multiclass", num_class=5, n_estimators=800, **LGB_COMMON)
    model.fit(
        train_pdf[cols], train_pdf["class_label"], sample_weight=train_pdf["weight"],
        eval_set=[(hold_pdf[cols], hold_pdf["class_label"])], eval_sample_weight=[hold_pdf["weight"]],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


def evaluate_decisions(y_true, proba: np.ndarray) -> dict:
    """Costo oficial de tres reglas: todo sano, argmax y minimo costo esperado."""
    rules = {
        "siempre_sano": np.zeros(len(y_true), dtype=int),
        "argmax": proba.argmax(axis=1),
        "costo_esperado_minimo": C.bayes_decision(proba),
    }
    out = {}
    for name, pred in rules.items():
        matrix = C.confusion(y_true, pred)
        out[name] = {
            "costo_total": C.cost_from_confusion(matrix),
            **C.alarm_metrics(y_true, pred),
            "matriz_confusion": matrix.tolist(),
        }
    return out


# ------------------------------------------------------------------------- tiempo restante
def train_time_to_failure(train_pdf: pd.DataFrame, hold_pdf: pd.DataFrame, cols: list[str]):
    model = lgb.LGBMRegressor(objective="regression_l1", n_estimators=800, **LGB_COMMON)
    model.fit(
        train_pdf[cols], np.log1p(train_pdf["time_to_failure"]),
        eval_set=[(hold_pdf[cols], np.log1p(hold_pdf["time_to_failure"]))],
        callbacks=[lgb.early_stopping(50, verbose=False)],
    )
    return model


# ------------------------------------------------------------------------------ Cox
def landmark_table(readouts: pl.DataFrame, tte: pl.DataFrame, specs: pl.DataFrame, landmark: float = LANDMARK):
    """Una fila por vehiculo en el landmark: uso temprano por contador + especificaciones.

    Solo entra el uso observado hasta ``landmark``; el vehiculo se modela con entrada tardia
    (``entry = landmark``), asi que no se condiciona sobre informacion posterior.
    """
    single = [cols[0] for cols in E.variable_groups(readouts.columns).values() if len(cols) == 1]
    early = readouts.filter(pl.col("time_step") <= landmark).sort(E.ID_COLUMNS).group_by("vehicle_id", maintain_order=True).last()
    rates = early.select(
        "vehicle_id", *[(pl.col(c) / pl.col("time_step").clip(lower_bound=1.0)).log1p().alias(f"usage_{c}") for c in single]
    )
    df = tte.join(rates, on="vehicle_id", how="inner").join(specs, on="vehicle_id", how="inner")
    usage = [c for c in df.columns if c.startswith("usage_")]
    # Un contador nulo en la lectura del landmark se imputa con la mediana de la flota.
    df = df.with_columns([pl.col(c).fill_nan(None).fill_null(pl.col(c).median()) for c in usage])
    return df.filter(pl.col("duration") > landmark).with_columns(pl.lit(landmark).alias("entry"))


def cox_design(table: pl.DataFrame, categories: dict[str, list[str]], min_count: int = 100) -> pd.DataFrame:
    pdf = table.to_pandas()
    for c in SPEC_COLUMNS:
        counts = pdf[c].value_counts()
        rare = counts[counts < min_count].index
        pdf[c] = pdf[c].where(~pdf[c].isin(rare), "rare")
    return pdf


def fit_cox(design: pd.DataFrame, train_ids: set, hold_ids: set, use_specs: bool, use_usage: bool) -> dict:
    usage_cols = [c for c in design.columns if c.startswith("usage_")]
    parts = []
    if use_usage:
        parts.append(design[usage_cols])
    if use_specs:
        parts.append(pd.get_dummies(design[SPEC_COLUMNS], drop_first=True, dtype=float))
    X = pd.concat(parts, axis=1)
    X = X.loc[:, X.nunique() > 1]
    X["duration"], X["event"], X["entry"] = design["duration"], design["event"], design["entry"]
    mask_tr = design["vehicle_id"].isin(train_ids)
    train_x, hold_x = X[mask_tr].copy(), X[~mask_tr & design["vehicle_id"].isin(hold_ids)].copy()
    covariates = [c for c in X.columns if c not in ("duration", "event", "entry")]
    mu, sd = train_x[covariates].mean(), train_x[covariates].std().replace(0, 1)
    train_x[covariates] = (train_x[covariates] - mu) / sd
    hold_x[covariates] = (hold_x[covariates] - mu) / sd
    cph = CoxPHFitter(penalizer=0.1)
    cph.fit(train_x, duration_col="duration", event_col="event", entry_col="entry")
    risk = cph.predict_partial_hazard(hold_x)
    c_index = concordance_index(hold_x["duration"], -risk, hold_x["event"])
    return {"c_index": float(c_index), "n_covariates": len(covariates), "model": cph, "mu": mu, "sd": sd}


# ------------------------------------------------------------------------------- main
def run(force_features: bool = False) -> dict:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    tte = S.read_tte()
    train_specs = S.read_specs("train")
    cats = spec_categories(train_specs)

    # --- features y etiquetas de train
    feats_tr = load_features("train", force_features)
    readouts_tr = S.read_readouts("train")
    labeled = S.label_train_readouts(readouts_tr.select(E.ID_COLUMNS), tte).select(
        "vehicle_id", "time_step", "time_to_failure", "class_label"
    )
    full = feats_tr.join(labeled, on=["vehicle_id", "time_step"], how="inner")

    ids = tte["vehicle_id"].to_numpy()
    train_ids, hold_ids = vehicle_split(ids, tte["event"].to_numpy(), test_size=0.2)
    print(f"[Split] vehiculos train={len(train_ids)} holdout={len(hold_ids)}")

    # --- 1. clasificador de riesgo
    tr_rows = sample_rows(full.filter(pl.col("vehicle_id").is_in(list(train_ids))))
    ho_rows = sample_rows(full.filter(pl.col("vehicle_id").is_in(list(hold_ids))), seed=SEED + 1)
    tr_pdf = with_specs(tr_rows.to_pandas(), train_specs, cats)
    ho_pdf = with_specs(ho_rows.to_pandas(), train_specs, cats)
    cols = feature_columns(tr_pdf.drop(columns=["weight"]))
    print(f"[Clasificador] filas train={len(tr_pdf)} (clase>0: {(tr_pdf['class_label'] > 0).sum()}) | features={len(cols)}")
    clf = train_classifier(tr_pdf, ho_pdf, cols)
    print(f"[Clasificador] iteraciones usadas: {clf.best_iteration_}")

    results = {"datos": {
        "vehiculos_train": int(len(ids)), "reparaciones_train": int(tte["event"].sum()),
        "lecturas_train": int(readouts_tr.height), "features": len(cols),
    }, "clasificador": {"iteraciones": int(clf.best_iteration_)}}

    for split in ("validation", "test"):
        feats = E.last_readout_per_vehicle(load_features(split, force_features))
        labels = S.read_labels(split)
        frame = with_specs(feats.join(labels, on="vehicle_id", how="inner").to_pandas(), S.read_specs(split), cats)
        proba = clf.predict_proba(frame[cols])
        results["clasificador"][split] = evaluate_decisions(frame["class_label"].to_numpy(), proba)
        results["clasificador"][split]["n_vehiculos"] = int(len(frame))
        # Discriminacion pura, sin depender de la regla de decision ni de los costos.
        results["clasificador"][split]["auc_algun_riesgo"] = float(
            roc_auc_score(frame["class_label"].to_numpy() > 0, 1.0 - proba[:, 0])
        )
        res = results["clasificador"][split]
        print(f"[{split}] costo: todo-sano={res['siempre_sano']['costo_total']:,} argmax={res['argmax']['costo_total']:,} "
              f"min-costo-esperado={res['costo_esperado_minimo']['costo_total']:,} "
              f"(recall alarma {res['costo_esperado_minimo']['recall_alarma']:.2f}, falsas alarmas {res['costo_esperado_minimo']['falsas_alarmas']})")

    # --- 2. tiempo restante (solo camiones con reparacion observada)
    failed = full.filter(pl.col("time_to_failure").is_not_null())
    failed_ids = np.array(sorted(set(failed["vehicle_id"].to_list())))
    ftr, fho = vehicle_split(failed_ids, np.zeros(len(failed_ids)), test_size=0.25)
    ftr_pdf = with_specs(failed.filter(pl.col("vehicle_id").is_in(list(ftr))).to_pandas(), train_specs, cats)
    fho_pdf = with_specs(failed.filter(pl.col("vehicle_id").is_in(list(fho))).to_pandas(), train_specs, cats)
    ttf = train_time_to_failure(ftr_pdf, fho_pdf, cols)
    pred = np.expm1(ttf.predict(fho_pdf[cols])).clip(min=0)
    truth = fho_pdf["time_to_failure"].to_numpy()
    baseline = np.full_like(truth, np.median(ftr_pdf["time_to_failure"]))
    near = truth <= 48
    results["tiempo_restante"] = {
        "camiones_fallados_train": len(ftr), "camiones_fallados_holdout": len(fho),
        "mae": float(np.abs(pred - truth).mean()), "mae_baseline_mediana": float(np.abs(baseline - truth).mean()),
        "mae_ultimas_48": float(np.abs(pred[near] - truth[near]).mean()),
        "mae_ultimas_48_baseline": float(np.abs(baseline[near] - truth[near]).mean()),
        "lecturas_holdout": int(len(truth)), "lecturas_ultimas_48": int(near.sum()),
    }
    print(f"[Tiempo restante] MAE={results['tiempo_restante']['mae']:.1f} (baseline {results['tiempo_restante']['mae_baseline_mediana']:.1f}) "
          f"| ultimas 48: {results['tiempo_restante']['mae_ultimas_48']:.1f} (baseline {results['tiempo_restante']['mae_ultimas_48_baseline']:.1f})")

    # --- 3. supervivencia con entrada tardia
    design = cox_design(landmark_table(readouts_tr, tte, train_specs), cats)
    cox = {}
    for name, (sp, us) in {"especificaciones": (True, False), "uso_temprano": (False, True), "ambos": (True, True)}.items():
        fit = fit_cox(design, train_ids, hold_ids, use_specs=sp, use_usage=us)
        cox[name] = {"c_index": round(fit["c_index"], 4), "n_covariables": fit["n_covariates"]}
        if name == "ambos":
            joblib.dump({k: fit[k] for k in ("model", "mu", "sd")}, MODELS_DIR / "coxph_landmark.joblib")
    results["supervivencia"] = {"landmark": LANDMARK, "vehiculos_en_riesgo": int(len(design)), **cox}
    print(f"[Supervivencia] C-index holdout: {cox}")

    # --- 4. SHAP del modelo de tiempo restante
    import shap

    sample = fho_pdf[cols].sample(n=min(1000, len(fho_pdf)), random_state=SEED)
    shap_values = shap.TreeExplainer(ttf).shap_values(sample)
    imp = pd.DataFrame({"feature": cols, "mean_abs_shap": np.abs(shap_values).mean(axis=0)})
    imp = imp.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    imp.to_csv(REPORTS_DIR / "shap_time_to_failure.csv", index=False)
    results["shap_top"] = imp.head(10).round(4).to_dict(orient="records")

    joblib.dump(clf, MODELS_DIR / "risk_classifier.joblib")
    joblib.dump(ttf, MODELS_DIR / "time_to_failure.joblib")
    (MODELS_DIR / "feature_columns.json").write_text(json.dumps(cols), encoding="utf-8")
    readout_columns = [c for c in readouts_tr.columns if c != "vehicle_id"]
    (MODELS_DIR / "readout_columns.json").write_text(json.dumps(readout_columns), encoding="utf-8")
    (MODELS_DIR / "spec_categories.json").write_text(json.dumps(cats), encoding="utf-8")
    (REPORTS_DIR / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    from src.models.make_figures import make_all

    make_all(results)
    print(f"Resultados en {REPORTS_DIR / 'results.json'}")
    return results


if __name__ == "__main__":
    import sys

    run(force_features="--force" in sys.argv)
