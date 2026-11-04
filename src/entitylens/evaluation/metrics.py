"""Strict entity-level metrics via seqeval, plus BIO-validity diagnostics.

``entity_level_report`` takes *tag-string* sequences (not ids) and returns
micro precision/recall/F1 and per-type (PER/ORG/LOC/MISC) P/R/F1 under
seqeval's strict IOB2 scheme — the correct primary measure for NER.  Token-level
accuracy is reported only as a secondary diagnostic.
"""
from __future__ import annotations

from typing import Dict, List

from seqeval.metrics import classification_report, f1_score, precision_score, recall_score
from seqeval.scheme import IOB2

from ..labels import ENTITY_TYPES, is_valid_transition


def entity_level_report(y_true: List[List[str]], y_pred: List[List[str]]) -> Dict:
    micro = {
        "precision": float(precision_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, mode="strict", scheme=IOB2, zero_division=0)),
    }
    report = classification_report(y_true, y_pred, mode="strict", scheme=IOB2,
                                   output_dict=True, zero_division=0)
    per_type = {}
    for et in ENTITY_TYPES:
        row = report.get(et, {})
        per_type[et] = {
            "precision": float(row.get("precision", 0.0)),
            "recall": float(row.get("recall", 0.0)),
            "f1": float(row.get("f1-score", 0.0)),
            "support": int(row.get("support", 0)),
        }
    # token-level accuracy (secondary diagnostic only)
    correct = total = 0
    for t_seq, p_seq in zip(y_true, y_pred):
        for t, p in zip(t_seq, p_seq):
            total += 1
            correct += int(t == p)
    return {
        "micro": micro,
        "per_type": per_type,
        "token_accuracy": round(correct / total, 4) if total else 0.0,
    }


def bio_transition_validity(y_pred: List[List[str]]) -> Dict:
    """Fraction of predicted tag transitions that are legal IOB2 moves."""
    total = invalid = 0
    for seq in y_pred:
        prev = "O"
        for tag in seq:
            if not is_valid_transition(prev, tag):
                invalid += 1
            total += 1
            prev = tag
    return {
        "transitions": total,
        "invalid_transitions": invalid,
        "invalid_rate": round(invalid / total, 6) if total else 0.0,
    }
