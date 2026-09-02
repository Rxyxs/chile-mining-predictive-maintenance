"""Genera un grafico Plotly interactivo (predicho vs. real, RUL en horas) a partir
del holdout REAL usado por train_survival_pipeline.py (mismo split, mismo modelo
LightGBM ya entrenado y persistido). No genera datos nuevos: reconstruye la misma
tabla de evaluacion y la exporta a un HTML autocontenido.

    python -m src.models.make_interactive_chart

Autor: Pablo Reyes
"""
from __future__ import annotations

from pathlib import Path

import joblib
import plotly.graph_objects as go
import polars as pl

from src.models.train_survival_pipeline import (
    MODELS_DIR,
    PROCESSED_DIR,
    build_rul_training_table,
    equipment_train_test_split,
    prepare_model_frame,
)

OUTPUT_DIR = Path(__file__).resolve().parents[2] / "outputs" / "interactive"


def main() -> None:
    equipment_metadata = pl.read_parquet(PROCESSED_DIR / "equipment_metadata.parquet")
    features = pl.read_parquet(PROCESSED_DIR / "telemetry_features.parquet")

    rul_df = build_rul_training_table(features, equipment_metadata)
    equipment_ids = rul_df["equipment_id"].unique().to_list()
    train_ids, test_ids = equipment_train_test_split(equipment_ids)

    test_df = rul_df.filter(pl.col("equipment_id").is_in(test_ids))
    X_test = prepare_model_frame(test_df)
    y_test = test_df["rul_hours"].to_numpy()
    equip_ids = test_df["equipment_id"].to_list()
    faenas = test_df["faena"].to_list()

    model = joblib.load(MODELS_DIR / "rul_lightgbm.joblib")
    y_pred = model.predict(X_test)

    mae = float(abs(y_pred - y_test).mean())
    max_val = float(max(y_test.max(), y_pred.max()) * 1.05)

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=[0, max_val],
            y=[0, max_val],
            mode="lines",
            line=dict(color="#999999", dash="dash", width=1.5),
            name="Perfect prediction",
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=y_test,
            y=y_pred,
            mode="markers",
            marker=dict(
                size=6,
                color=y_test,
                colorscale="Viridis",
                showscale=True,
                colorbar=dict(title="Actual RUL (h)"),
                opacity=0.75,
                line=dict(width=0.5, color="rgba(0,0,0,0.3)"),
            ),
            text=[f"{eid} — {f}" for eid, f in zip(equip_ids, faenas)],
            hovertemplate=(
                "<b>%{text}</b><br>Actual RUL: %{x:.0f} h<br>"
                "Predicted RUL: %{y:.0f} h<extra></extra>"
            ),
            name="Test units",
        )
    )
    fig.update_layout(
        title=(
            f"01 — Mining fleet RUL: predicted vs. actual (LightGBM, holdout, "
            f"n={len(y_test)}, MAE={mae:.1f} h)"
        ),
        xaxis_title="Actual RUL (hours)",
        yaxis_title="Predicted RUL (hours)",
        template="plotly_white",
        width=900,
        height=650,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    fig.update_xaxes(range=[0, max_val])
    fig.update_yaxes(range=[0, max_val])

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / "rul-predicted-vs-actual.html"
    fig.write_html(out_path, include_plotlyjs="inline", full_html=True)
    print(f"MAE holdout (recomputed for this chart): {mae:.2f} h over {len(y_test)} units")
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
