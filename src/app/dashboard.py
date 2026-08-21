"""Dashboard de telemetria de flota minera con heatmap de riesgo por equipo.

Ejecutar desde la raiz del repositorio con:
    streamlit run src/app/dashboard.py
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import streamlit as st

from src.models.scoring import PROCESSED_DIR, FleetScorer

RISK_ORDER = ["CRITICO", "ALTO", "MEDIO", "BAJO"]
RISK_COLORS = {"CRITICO": "#d62728", "ALTO": "#ff7f0e", "MEDIO": "#e0c341", "BAJO": "#2ca02c"}

st.set_page_config(page_title="Chile Mining Predictive Maintenance", layout="wide")


@st.cache_resource
def load_scorer() -> FleetScorer:
    return FleetScorer()


@st.cache_data
def load_fleet_scores(_scorer: FleetScorer) -> pl.DataFrame:
    return _scorer.score_fleet()


@st.cache_data
def load_raw_telemetry() -> pl.DataFrame:
    return pl.read_parquet(PROCESSED_DIR / "sensor_telemetry.parquet")


def build_risk_heatmap(fleet_pdf) -> go.Figure:
    risk_rank = {level: i for i, level in enumerate(RISK_ORDER)}
    faenas = sorted(fleet_pdf["faena"].unique())
    max_n = fleet_pdf.groupby("faena").size().max()

    z = np.full((len(faenas), max_n), np.nan)
    hover = np.full((len(faenas), max_n), "", dtype=object)

    for i, faena in enumerate(faenas):
        sub = fleet_pdf[fleet_pdf["faena"] == faena].sort_values("predicted_rul_hours").reset_index(drop=True)
        for j, row in sub.iterrows():
            z[i, j] = risk_rank[row["risk_level"]]
            hover[i, j] = (
                f"{row['equipment_id']} ({row['equipment_type']})<br>"
                f"RUL estimado: {row['predicted_rul_hours']:.0f} h<br>"
                f"Riesgo: {row['risk_level']}<br>"
                f"Falla probable: {row['predicted_failure_type']}"
            )

    fig = go.Figure(
        go.Heatmap(
            z=z,
            y=faenas,
            hovertext=hover,
            hoverinfo="text",
            colorscale=[[0, RISK_COLORS["CRITICO"]], [0.33, RISK_COLORS["ALTO"]], [0.66, RISK_COLORS["MEDIO"]], [1, RISK_COLORS["BAJO"]]],
            zmin=0,
            zmax=3,
            colorbar=dict(tickvals=[0, 1, 2, 3], ticktext=RISK_ORDER, title="Riesgo"),
            xgap=2,
            ygap=4,
        )
    )
    fig.update_layout(
        title="Heatmap de riesgo por equipo (cada celda = un equipo, ordenado por RUL dentro de su faena)",
        xaxis_title="Equipo (indice dentro de la faena, ordenado por RUL ascendente)",
        yaxis_title="Faena",
        height=380,
        margin=dict(l=10, r=10, t=60, b=10),
    )
    fig.update_xaxes(showticklabels=False)
    return fig


def main() -> None:
    st.title("⛏️ Chile Mining Predictive Maintenance")
    st.caption("Score de salud de flota en tiempo real — CAEX y chancadores primarios")

    scorer = load_scorer()
    fleet_scores = load_fleet_scores(scorer)
    fleet_pdf = fleet_scores.to_pandas()

    with st.sidebar:
        st.header("Filtros")
        faenas_sel = st.multiselect("Faena", sorted(fleet_pdf["faena"].unique()), default=None)
        types_sel = st.multiselect("Tipo de equipo", sorted(fleet_pdf["equipment_type"].unique()), default=None)
        risk_sel = st.multiselect("Nivel de riesgo", RISK_ORDER, default=None)

    filtered = fleet_pdf.copy()
    if faenas_sel:
        filtered = filtered[filtered["faena"].isin(faenas_sel)]
    if types_sel:
        filtered = filtered[filtered["equipment_type"].isin(types_sel)]
    if risk_sel:
        filtered = filtered[filtered["risk_level"].isin(risk_sel)]

    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Equipos monitoreados", len(filtered))
    col2.metric("Riesgo CRITICO", int((filtered["risk_level"] == "CRITICO").sum()))
    col3.metric("Riesgo ALTO", int((filtered["risk_level"] == "ALTO").sum()))
    col4.metric("RUL promedio (h)", f"{filtered['predicted_rul_hours'].mean():.0f}" if len(filtered) else "-")

    st.plotly_chart(build_risk_heatmap(filtered if len(filtered) else fleet_pdf), use_container_width=True)

    st.subheader("Flota filtrada")
    st.dataframe(
        filtered[
            [
                "equipment_id",
                "equipment_type",
                "faena",
                "operating_hours",
                "predicted_rul_hours",
                "risk_level",
                "predicted_failure_type",
            ]
        ].sort_values("predicted_rul_hours"),
        use_container_width=True,
        height=300,
    )

    st.subheader("Detalle de equipo")
    equipment_options = filtered["equipment_id"].tolist() or fleet_pdf["equipment_id"].tolist()
    selected_id = st.selectbox("Seleccionar equipo", equipment_options)

    if selected_id:
        score = scorer.score_equipment(selected_id)
        detail_col, chart_col = st.columns([1, 2])

        with detail_col:
            st.metric("RUL estimado", f"{score['predicted_rul_hours']:.0f} h")
            st.metric("Riesgo", score["risk_level"])
            st.metric("Falla mas probable", score["predicted_failure_type"])
            if score["survival_probability_30d"] is not None:
                st.metric("P(sobrevive 30 dias)", f"{score['survival_probability_30d'] * 100:.1f}%")

            proba_df = (
                pl.DataFrame(
                    {
                        "failure_type": list(score["failure_type_probabilities"].keys()),
                        "probability": list(score["failure_type_probabilities"].values()),
                    }
                )
                .sort("probability", descending=True)
                .to_pandas()
            )
            st.plotly_chart(
                px.bar(proba_df, x="probability", y="failure_type", orientation="h", title="Probabilidad por tipo de falla"),
                use_container_width=True,
            )

        with chart_col:
            raw_telemetry = load_raw_telemetry()
            history = raw_telemetry.filter(pl.col("equipment_id") == selected_id).sort("timestamp").to_pandas()
            fig = px.line(
                history,
                x="timestamp",
                y=["engine_temp_c", "vibration_rms_mm_s", "hydraulic_pressure_bar"],
                title=f"Tendencia de sensores — {selected_id}",
            )
            st.plotly_chart(fig, use_container_width=True)


if __name__ == "__main__":
    main()
