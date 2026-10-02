"""Figuras de resultados a partir de ``reports/results.json`` y el CSV de SHAP."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

REPORTS_DIR = Path(__file__).resolve().parents[2] / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

SURFACE, INK, INK_2, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e3df"
GRAY, ORANGE, BLUE = "#9a9994", "#eb6834", "#2a78d6"
SEQ = LinearSegmentedColormap.from_list("seq_blue", ["#e8f0fb", "#9cc0ef", "#2a78d6", "#123f78"])


def _style(ax):
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=INK_2, labelsize=9, length=0)


def _save(fig, name):
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FIGURES_DIR / name
    fig.savefig(path, dpi=160, facecolor=SURFACE, bbox_inches="tight")
    plt.close(fig)
    return path


def fig_costs(results: dict) -> Path:
    rules = [("siempre_sano", "Siempre «sano»", GRAY), ("argmax", "Clase más probable", ORANGE), ("costo_esperado_minimo", "Costo esperado mínimo", BLUE)]
    splits = [("validation", "Validación"), ("test", "Test")]
    fig, ax = plt.subplots(figsize=(8.5, 4.6), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="y", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    width = 0.26
    for i, (key, label, color) in enumerate(rules):
        vals = [results["clasificador"][s][key]["costo_total"] for s, _ in splits]
        x = np.arange(len(splits)) + (i - 1) * (width + 0.02)
        ax.bar(x, vals, width=width, color=color, label=label)
        for xi, v in zip(x, vals):
            ax.text(xi, v + 600, f"{v:,}".replace(",", "."), ha="center", fontsize=9, color=INK)
    ax.set_xticks(range(len(splits)), [n for _, n in splits])
    ax.set_ylabel("Costo total oficial del reto (menor es mejor)", color=INK_2, fontsize=9)
    ax.set_ylim(0, 66000)
    ax.set_title("Costo de la decisión de mantenimiento, por regla", loc="left", color=INK, fontsize=12)
    ax.legend(frameon=False, fontsize=9, labelcolor=INK, loc="upper center", bbox_to_anchor=(0.5, -0.1), ncol=3)
    return _save(fig, "01_costo_decision.png")


def fig_confusion(results: dict, split: str = "test") -> Path:
    m = np.array(results["clasificador"][split]["costo_esperado_minimo"]["matriz_confusion"])
    rows = m / m.sum(axis=1, keepdims=True)
    fig, ax = plt.subplots(figsize=(6.2, 5), facecolor=SURFACE)
    ax.imshow(rows, cmap=SEQ, vmin=0, vmax=1)
    for i in range(5):
        for j in range(5):
            ax.text(j, i, f"{rows[i, j]:.0%}\n({m[i, j]})", ha="center", va="center", fontsize=9,
                    color="#ffffff" if rows[i, j] > 0.5 else INK)
    ax.set_xticks(range(5), range(5))
    ax.set_yticks(range(5), range(5))
    ax.set_xlabel("Clase recomendada (acción)", color=INK_2, fontsize=9)
    ax.set_ylabel("Clase real", color=INK_2, fontsize=9)
    ax.tick_params(length=0, colors=INK_2)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_title(f"Matriz de confusión en {split} (porcentaje por fila, conteo entre paréntesis)", loc="left", color=INK, fontsize=10)
    return _save(fig, "02_matriz_confusion.png")


def fig_shap(top: int = 15) -> Path:
    imp = pd.read_csv(REPORTS_DIR / "shap_time_to_failure.csv").head(top).iloc[::-1]
    fig, ax = plt.subplots(figsize=(8, 5), facecolor=SURFACE)
    _style(ax)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.barh(imp["feature"], imp["mean_abs_shap"], color=BLUE, height=0.62)
    ax.set_xlabel("|SHAP| medio sobre log(1 + tiempo restante)", color=INK_2, fontsize=9)
    ax.set_title("Variables que más mueven el pronóstico de tiempo restante", loc="left", color=INK, fontsize=12)
    return _save(fig, "03_shap_tiempo_restante.png")


def make_all(results: dict | None = None) -> list[Path]:
    results = results or json.loads((REPORTS_DIR / "results.json").read_text(encoding="utf-8"))
    return [fig_costs(results), fig_confusion(results), fig_shap()]


if __name__ == "__main__":
    for p in make_all():
        print(p)
