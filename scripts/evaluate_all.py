"""Evaluate every trained model on the held-out test set and build the
comparison table + model selection.

For each model found under ``artifacts/`` this computes strict entity-level
seqeval metrics (overall + per type), measures CPU inference latency and
on-disk size, reads training time / params / best validation F1 from the saved
training summary, and writes:
    results/model_comparison.json   (machine-readable, used by the app)
    results/<model>_test_report.json

Model selection is by **validation** F1 (test metrics are reported, never used
for selection).  Run after training:  python scripts/evaluate_all.py
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entitylens.data.dataset import load_split  # noqa: E402
from entitylens.evaluation.metrics import bio_transition_validity, entity_level_report  # noqa: E402
from entitylens.inference.predictor import Predictor  # noqa: E402
from entitylens.labels import ID2LABEL  # noqa: E402

PROCESSED = ROOT / "data" / "processed"
ARTIFACTS = ROOT / "artifacts"
RESULTS = ROOT / "results"

# (display name, path, kind)
MODEL_LOCATIONS = [
    ("LSTM", ARTIFACTS / "lstm", "rnn"),
    ("BiLSTM", ARTIFACTS / "bilstm", "rnn"),
    ("BiLSTM+CRF", ARTIFACTS / "bilstm_crf", "rnn"),
    ("Transformer", ARTIFACTS / "transformer" / "model", "transformer"),
]


def _dir_size_mb(path: Path) -> float:
    if path.is_file():
        total = path.stat().st_size
    else:
        total = sum(f.stat().st_size for f in path.rglob("*") if f.is_file())
    return round(total / 1e6, 2)


def _summary_for(path: Path, kind: str) -> dict:
    # training summaries live next to the artifacts
    base = path.parent if kind == "transformer" else path
    f = base / "training_summary.json"
    return json.loads(f.read_text()) if f.exists() else {}


def evaluate_model(name: str, path: Path, kind: str, test, n_latency: int = 100) -> dict:
    predictor = Predictor.load(path)
    y_true, y_pred = [], []
    for ex in test:
        words = ex["tokens"]
        y_true.append([ID2LABEL[i] for i in ex["ner_tags"]])
        y_pred.append(predictor.predict_pretokenized(words))
    rep = entity_level_report(y_true, y_pred)
    valid = bio_transition_validity(y_pred)

    # latency: mean ms per sentence over a sample (CPU)
    sample = test[:n_latency]
    t0 = time.time()
    for ex in sample:
        predictor.predict_pretokenized(ex["tokens"])
    latency_ms = round((time.time() - t0) / max(1, len(sample)) * 1000, 2)

    summ = _summary_for(path, kind)
    size_mb = _dir_size_mb(path if kind == "transformer" else path / "best.pt")
    return {
        "model": name,
        "precision": round(rep["micro"]["precision"], 4),
        "recall": round(rep["micro"]["recall"], 4),
        "f1": round(rep["micro"]["f1"], 4),
        "token_accuracy": rep["token_accuracy"],
        "per_type": {k: round(v["f1"], 4) for k, v in rep["per_type"].items()},
        "per_type_full": rep["per_type"],
        "invalid_bio_rate": valid["invalid_rate"],
        "val_f1": summ.get("best_val_f1"),
        "train_time_sec": summ.get("train_time_sec"),
        "params": summ.get("trainable_params"),
        "latency_ms": latency_ms,
        "size_mb": size_mb,
    }


def select_model(rows: list) -> tuple:
    scored = [r for r in rows if r.get("val_f1") is not None]
    pool = scored or rows
    best = max(pool, key=lambda r: (r.get("val_f1") or r["f1"]))
    reason = ("highest entity-level validation F1"
              + (f" ({best['val_f1']:.4f})" if best.get("val_f1") else ""))
    return best["model"], reason


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latency-n", type=int, default=100)
    args = ap.parse_args()
    RESULTS.mkdir(parents=True, exist_ok=True)
    test = load_split(PROCESSED, "test")

    rows = []
    for name, path, kind in MODEL_LOCATIONS:
        exists = (path / "best.pt").exists() or (path / "config.json").exists()
        if not exists:
            print(f"[skip] {name}: no artifacts at {path}")
            continue
        print(f"[eval] {name} ...", flush=True)
        row = evaluate_model(name, path, kind, test, args.latency_n)
        rows.append(row)
        (RESULTS / f"{name.replace('+','_').replace(' ','_')}_test_report.json").write_text(
            json.dumps(row, indent=2))
        print(f"       F1={row['f1']:.4f} P={row['precision']:.4f} "
              f"R={row['recall']:.4f} latency={row['latency_ms']}ms")

    if not rows:
        print("No trained models found. Train first.")
        return
    selected, reason = select_model(rows)
    comparison = {
        "comparison": rows,
        "selected_model": selected,
        "selection_reason": reason,
        "protocol": ("Strict entity-level seqeval (IOB2) on the frozen test split; "
                     "selection by validation F1, test used only for final report."),
    }
    (RESULTS / "model_comparison.json").write_text(json.dumps(comparison, indent=2))
    print(f"\nSelected production model: {selected} ({reason})")
    print("Wrote results/model_comparison.json")


if __name__ == "__main__":
    main()
