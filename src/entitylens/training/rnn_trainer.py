"""Training loop for the recurrent taggers (LSTM / BiLSTM / BiLSTM+CRF).

Features required by the brief: fixed seed, Adam optimizer, LR scheduling
(ReduceLROnPlateau on val F1), gradient clipping, dropout (in the model),
early stopping on entity-level validation F1, full training history, best
checkpoint saving with all metadata needed for inference, and resume support.
Padding never contributes to the loss (CrossEntropy ``ignore_index`` / CRF mask).
"""
from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Dict, Optional

import torch
from torch.utils.data import DataLoader

from ..evaluation.metrics import bio_transition_validity, entity_level_report
from .utils import count_params, ids_to_tags, select_device, set_seed


def _batch_to_device(batch, device):
    return {k: v.to(device) for k, v in batch.items()}


@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    model.eval()
    y_true, y_pred = [], []
    for batch in loader:
        b = _batch_to_device(batch, device)
        preds = model.decode(b["word_ids"], b["char_ids"], b["lengths"], b["mask"])
        lengths = b["lengths"].tolist()
        gold = [batch["labels"][i, :lengths[i]].tolist() for i in range(len(lengths))]
        y_true.extend(ids_to_tags(gold))
        y_pred.extend(ids_to_tags(preds))
    report = entity_level_report(y_true, y_pred)
    report["bio_validity"] = bio_transition_validity(y_pred)
    return report


def train_model(model, train_loader: DataLoader, val_loader: DataLoader,
                config: Dict, out_dir: str | Path, vocab_path: str | Path,
                resume: bool = False) -> Dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.get("seed", 42))
    device = select_device(config.get("device", "auto"))
    model.to(device)

    lr = config.get("lr", 1e-3)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=config.get("weight_decay", 0.0))
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=config.get("sched_patience", 1))
    clip = config.get("grad_clip", 5.0)
    max_epochs = config.get("epochs", 30)
    patience = config.get("early_stop_patience", 5)
    is_crf = getattr(model, "arch", "") == "bilstm_crf"

    ckpt_path = out_dir / "best.pt"
    history, best_f1, best_epoch, bad = [], -1.0, -1, 0
    start_epoch = 0

    if resume and (out_dir / "state.pt").exists():
        state = torch.load(out_dir / "state.pt", map_location=device)
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        start_epoch = state["epoch"] + 1
        best_f1 = state["best_f1"]
        history = state["history"]
        print(f"Resumed from epoch {start_epoch} (best F1 {best_f1:.4f})")

    t0 = time.time()
    for epoch in range(start_epoch, max_epochs):
        model.train()
        ep_loss, n = 0.0, 0
        for batch in train_loader:
            b = _batch_to_device(batch, device)
            optimizer.zero_grad()
            out = model(b["word_ids"], b["char_ids"], b["lengths"],
                        mask=b["mask"], labels=b["labels"])
            loss = out["loss"]
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()
            if is_crf:
                model.apply_constraints()
            ep_loss += loss.item() * len(b["lengths"]); n += len(b["lengths"])
        val = evaluate(model, val_loader, device)
        f1 = val["micro"]["f1"]
        scheduler.step(f1)
        row = {"epoch": epoch, "train_loss": round(ep_loss / n, 4),
               "val_f1": round(f1, 4), "val_precision": round(val["micro"]["precision"], 4),
               "val_recall": round(val["micro"]["recall"], 4),
               "lr": optimizer.param_groups[0]["lr"]}
        history.append(row)
        print(f"epoch {epoch:02d} | loss {row['train_loss']:.4f} | "
              f"val P/R/F1 {row['val_precision']:.4f}/{row['val_recall']:.4f}/{f1:.4f}")

        improved = f1 > best_f1
        if improved:
            best_f1, best_epoch, bad = f1, epoch, 0
            torch.save({
                "model_state": model.state_dict(),
                "arch": model.arch,
                "config": config,
                "val_metrics": val,
                "vocab_file": Path(vocab_path).name,
                "epoch": epoch,
            }, ckpt_path)
        else:
            bad += 1
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "epoch": epoch, "best_f1": best_f1, "history": history},
                   out_dir / "state.pt")
        if bad >= patience:
            print(f"Early stopping at epoch {epoch} (best epoch {best_epoch})")
            break

    train_time = round(time.time() - t0, 1)
    summary = {
        "arch": model.arch, "best_epoch": best_epoch, "best_val_f1": round(best_f1, 4),
        "epochs_run": len(history), "train_time_sec": train_time,
        "trainable_params": count_params(model), "device": str(device),
        "history": history,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
