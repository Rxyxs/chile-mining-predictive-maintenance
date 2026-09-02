"""Genera el GIF animado de las curvas de perdida (train/val) de la red
multi-task de PyTorch, a partir del historial REAL guardado por
`multitask_pdm.py` en `data/processed/multitask_loss_history.json`.

Es un complemento del grafico estatico `multitask_loss_curves.png` (celda 3
de `02_PyTorch_MultiTask_DeepSHAP.ipynb`): mismos datos, mismos paneles,
pero como "racing line chart" -- la linea se dibuja progresivamente por
epoca, con una etiqueta flotante en la punta de cada serie que muestra su
valor actual.

Ejecutar desde la raiz del repositorio con:
    python generate_multitask_loss_animation.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

from src.models.train_survival_pipeline import MODELS_DIR, PROCESSED_DIR

LOSS_HISTORY_PATH = PROCESSED_DIR / "multitask_loss_history.json"
OUTPUT_PATH = MODELS_DIR / "multitask_loss_curves_animated.gif"

PANELS = [
    ("total", "Perdida combinada (MSE RUL + CE falla)"),
    ("rul_mse", "Componente RUL (MSE, escala normalizada)"),
    ("cls_ce", "Componente clasificacion (Cross-Entropy)"),
]


def main() -> None:
    if not LOSS_HISTORY_PATH.exists():
        raise FileNotFoundError(
            f"No se encontro {LOSS_HISTORY_PATH}. Corre primero: python multitask_pdm.py"
        )
    with open(LOSS_HISTORY_PATH, encoding="utf-8") as f:
        loss_history = json.load(f)

    epochs = loss_history["epoch"]
    n_frames = len(epochs)  # 60 epocas reales -> 1 frame por epoca (dentro de 30-60)

    plt.style.use("dark_background")
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))

    lines = {}
    labels = {}
    for ax, (key, title) in zip(axes, PANELS):
        train_y = loss_history[f"train_{key}"]
        val_y = loss_history[f"val_{key}"]
        (train_line,) = ax.plot([], [], color="#4CC9F0", label="train", linewidth=2)
        (val_line,) = ax.plot([], [], color="#F72585", label="validacion", linewidth=2)
        train_label = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, 10),
            textcoords="offset points",
            color="#4CC9F0",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="#4CC9F0", alpha=0.85),
        )
        val_label = ax.annotate(
            "",
            xy=(0, 0),
            xytext=(10, -20),
            textcoords="offset points",
            color="#F72585",
            fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="black", ec="#F72585", alpha=0.85),
        )
        ax.set_xlim(min(epochs), max(epochs))
        y_all = train_y + val_y
        pad = (max(y_all) - min(y_all)) * 0.1 or 0.1
        ax.set_ylim(min(y_all) - pad, max(y_all) + pad)
        ax.set_xlabel("Epoca")
        ax.set_ylabel("Perdida")
        ax.set_title(title)
        ax.legend(loc="upper right")
        ax.grid(alpha=0.2)
        lines[key] = (train_line, val_line, train_y, val_y)
        labels[key] = (train_label, val_label)

    fig.suptitle("Entrenamiento red multi-task (PyTorch) -- perdida por epoca (real)")
    plt.tight_layout()

    def update(frame: int):
        idx = frame + 1  # al menos 1 punto en el primer frame
        x = epochs[:idx]
        artists = []
        for key, _ in PANELS:
            train_line, val_line, train_y, val_y = lines[key]
            train_label, val_label = labels[key]
            train_line.set_data(x, train_y[:idx])
            val_line.set_data(x, val_y[:idx])
            train_label.xy = (x[-1], train_y[idx - 1])
            train_label.set_text(f"train {train_y[idx - 1]:.3f}")
            val_label.xy = (x[-1], val_y[idx - 1])
            val_label.set_text(f"val {val_y[idx - 1]:.3f}")
            artists.extend([train_line, val_line, train_label, val_label])
        return artists

    ani = FuncAnimation(fig, update, frames=n_frames, interval=120, blit=False)
    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    ani.save(OUTPUT_PATH, writer="pillow")
    plt.close(fig)
    print(f"GIF animado guardado en: {OUTPUT_PATH}")


if __name__ == "__main__":
    main()
