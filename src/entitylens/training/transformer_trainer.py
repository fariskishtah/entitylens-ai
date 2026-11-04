"""Fine-tune a Hugging Face Transformer for token classification.

Genuine fine-tuning of ``bert-base-cased`` (or a configured alternative) with
``AutoModelForTokenClassification`` + the HF ``Trainer``.  Subword labels are
aligned so only the first subword of each word is scored; ``compute_metrics``
runs strict entity-level seqeval after dropping ignored (-100) positions, so the
reported F1 is word-level and directly comparable to the recurrent models.
``id2label`` / ``label2id`` are written into the saved config, so the deployed
checkpoint carries the exact mapping used in training.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import numpy as np

from ..evaluation.metrics import bio_transition_validity, entity_level_report
from ..labels import ID2LABEL, LABEL2ID, LABELS


def _build_compute_metrics():
    def compute_metrics(eval_pred):
        logits, labels = eval_pred
        preds = np.argmax(logits, axis=-1)
        y_true, y_pred = [], []
        for pred_row, lab_row in zip(preds, labels):
            t, p = [], []
            for pi, li in zip(pred_row, lab_row):
                if li == -100:
                    continue
                t.append(ID2LABEL[int(li)])
                p.append(ID2LABEL[int(pi)])
            y_true.append(t)
            y_pred.append(p)
        rep = entity_level_report(y_true, y_pred)
        return {
            "precision": rep["micro"]["precision"],
            "recall": rep["micro"]["recall"],
            "f1": rep["micro"]["f1"],
            "token_accuracy": rep["token_accuracy"],
            "invalid_bio_rate": bio_transition_validity(y_pred)["invalid_rate"],
        }
    return compute_metrics


def fine_tune(config: Dict, processed_dir: str | Path, out_dir: str | Path,
              smoke: bool = False) -> Dict:
    import time

    import torch
    from datasets import Dataset
    from transformers import (AutoModelForTokenClassification, AutoTokenizer,
                              DataCollatorForTokenClassification, Trainer,
                              TrainingArguments, set_seed)

    from ..data.alignment import tokenize_and_align_labels
    from ..data.dataset import load_split

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    set_seed(config.get("seed", 42))
    model_name = config.get("model_name", "bert-base-cased")

    raw = {s: load_split(processed_dir, s) for s in ["train", "validation", "test"]}
    if smoke:
        model_name = config.get("smoke_model_name", "prajjwal1/bert-tiny")
        raw = {s: ex[:200] for s, ex in raw.items()}
        config = {**config, "epochs": 1, "batch_size": 8}

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds = {s: Dataset.from_dict({"tokens": [e["tokens"] for e in ex],
                                "ner_tags": [e["ner_tags"] for e in ex]})
          for s, ex in raw.items()}
    max_len = config.get("max_length", 256)
    label_all = config.get("label_all_subwords", False)
    ds = {s: d.map(lambda b: tokenize_and_align_labels(b, tokenizer, max_len, label_all),
                   batched=True, remove_columns=["tokens", "ner_tags"])
          for s, d in ds.items()}

    model = AutoModelForTokenClassification.from_pretrained(
        model_name, num_labels=len(LABELS), id2label=ID2LABEL, label2id=LABEL2ID)

    args = TrainingArguments(
        output_dir=str(out_dir / "hf_trainer"),
        eval_strategy="epoch", save_strategy="epoch",
        learning_rate=config.get("lr", 5e-5),
        per_device_train_batch_size=config.get("batch_size", 16),
        per_device_eval_batch_size=config.get("batch_size", 16),
        num_train_epochs=config.get("epochs", 4),
        weight_decay=config.get("weight_decay", 0.01),
        warmup_ratio=config.get("warmup_ratio", 0.1),
        load_best_model_at_end=True, metric_for_best_model="f1", greater_is_better=True,
        logging_steps=50, seed=config.get("seed", 42),
        report_to=[], fp16=torch.cuda.is_available(),
    )
    trainer = Trainer(
        model=model, args=args, train_dataset=ds["train"], eval_dataset=ds["validation"],
        tokenizer=tokenizer,
        data_collator=DataCollatorForTokenClassification(tokenizer),
        compute_metrics=_build_compute_metrics(),
    )
    t0 = time.time()
    trainer.train()
    train_time = round(time.time() - t0, 1)

    val_metrics = trainer.evaluate(ds["validation"])
    final_dir = out_dir / "model"
    trainer.save_model(str(final_dir))
    tokenizer.save_pretrained(str(final_dir))

    summary = {
        "arch": "transformer", "model_name": model_name,
        "best_val_f1": round(val_metrics.get("eval_f1", 0.0), 4),
        "val_metrics": {k: v for k, v in val_metrics.items() if isinstance(v, (int, float))},
        "train_time_sec": train_time,
        "trainable_params": sum(p.numel() for p in model.parameters()),
        "epochs": config.get("epochs", 4), "max_length": max_len,
    }
    (out_dir / "training_summary.json").write_text(json.dumps(summary, indent=2))
    return summary
