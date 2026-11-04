"""Generate result figures into results/figures/ from saved metrics.

    python scripts/make_figures.py

Produces (only for models that exist):
  - training_curves_<model>.png   (loss + val F1 per epoch)
  - model_comparison_f1.png       (entity F1 bars across models)
  - per_class_f1.png              (PER/ORG/LOC/MISC F1 per model)
  - efficiency.png                (F1 vs latency)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts"
RES = ROOT / "results"
FIG = RES / "figures"
NAVY, BLUE = "#0f172a", "#2563eb"
TYPE_COLORS = {"PER": "#2563eb", "ORG": "#7c3aed", "LOC": "#059669", "MISC": "#d97706"}


def training_curves():
    for name in ["lstm", "bilstm", "bilstm_crf", "transformer"]:
        f = ART / name / "training_summary.json"
        if not f.exists():
            continue
        hist = json.loads(f.read_text()).get("history", [])
        if not hist:
            continue
        ep = [h["epoch"] for h in hist]
        fig, ax1 = plt.subplots(figsize=(7, 4))
        ax1.plot(ep, [h["train_loss"] for h in hist], color=NAVY, marker="o", label="train loss")
        ax1.set_xlabel("epoch"); ax1.set_ylabel("train loss", color=NAVY)
        ax2 = ax1.twinx()
        ax2.plot(ep, [h["val_f1"] for h in hist], color=BLUE, marker="s", label="val F1")
        ax2.set_ylabel("val entity F1", color=BLUE)
        plt.title(f"Training curve — {name}")
        fig.tight_layout(); fig.savefig(FIG / f"training_curves_{name}.png", dpi=130)
        plt.close(fig)


def comparison_figures():
    comp_f = RES / "model_comparison.json"
    if not comp_f.exists():
        return
    rows = json.loads(comp_f.read_text())["comparison"]
    names = [r["model"] for r in rows]

    plt.figure(figsize=(7, 4))
    plt.bar(names, [r["f1"] for r in rows], color=BLUE)
    for i, r in enumerate(rows):
        plt.text(i, r["f1"] + 0.005, f'{r["f1"]:.3f}', ha="center", fontsize=9)
    plt.ylabel("entity-level F1 (test)"); plt.title("Model comparison — entity F1")
    plt.ylim(0, 1.0); plt.tight_layout()
    plt.savefig(FIG / "model_comparison_f1.png", dpi=130); plt.close()

    types = ["PER", "ORG", "LOC", "MISC"]
    import numpy as np
    x = np.arange(len(names)); w = 0.2
    plt.figure(figsize=(8, 4.5))
    for j, t in enumerate(types):
        plt.bar(x + (j - 1.5) * w, [r["per_type"].get(t, 0) for r in rows], w,
                label=t, color=TYPE_COLORS[t])
    plt.xticks(x, names); plt.ylabel("F1"); plt.title("Per-class F1 by model")
    plt.legend(); plt.ylim(0, 1.0); plt.tight_layout()
    plt.savefig(FIG / "per_class_f1.png", dpi=130); plt.close()

    plt.figure(figsize=(6.5, 4.5))
    for r in rows:
        plt.scatter(r.get("latency_ms", 0), r["f1"], s=80, color=BLUE)
        plt.annotate(r["model"], (r.get("latency_ms", 0), r["f1"]),
                     textcoords="offset points", xytext=(6, 4), fontsize=9)
    plt.xlabel("CPU latency (ms/sentence)"); plt.ylabel("entity F1 (test)")
    plt.title("Efficiency: accuracy vs latency"); plt.tight_layout()
    plt.savefig(FIG / "efficiency.png", dpi=130); plt.close()


def main():
    FIG.mkdir(parents=True, exist_ok=True)
    training_curves()
    comparison_figures()
    print("Figures written to", FIG)


if __name__ == "__main__":
    main()
