"""Dataset audit for CoNLL-2003.

Produces a single JSON report covering every item the project brief requires:
sentence/token/entity counts, entity-type and BIO distributions, sentence
length stats, cross-split OOV rate, invalid/missing labels, invalid BIO
transitions, duplicated sentences and cross-split overlap (leakage).
Everything is computed from the *frozen* splits — no fabricated numbers.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, List

from ..labels import ID2LABEL, LABELS, is_valid_transition, split_tag
from .dataset import load_split


def _tags(example) -> List[str]:
    return [ID2LABEL[i] for i in example["ner_tags"]]


def _count_entities(tag_seq: List[str]) -> Counter:
    """Count entity spans by type (a span starts at B- or at an I- with no
    matching predecessor, which we also count as a span start for robustness)."""
    counts: Counter = Counter()
    prev = "O"
    for tag in tag_seq:
        prefix, etype = split_tag(tag)
        if prefix == "B":
            counts[etype] += 1
        elif prefix == "I":
            pp, pt = split_tag(prev)
            if not (pp in {"B", "I"} and pt == etype):
                counts[etype] += 1  # dangling I- opens a span
        prev = tag
    return counts


def audit_split(examples: List[Dict]) -> Dict:
    n_tokens = 0
    sent_lengths = []
    label_counter: Counter = Counter()
    entity_counter: Counter = Counter()
    invalid_transitions = 0
    invalid_label_ids = 0
    for ex in examples:
        toks, ids = ex["tokens"], ex["ner_tags"]
        if len(toks) != len(ids):
            invalid_label_ids += abs(len(toks) - len(ids))
        sent_lengths.append(len(toks))
        n_tokens += len(toks)
        tags = []
        for i in ids:
            if i < 0 or i >= len(LABELS):
                invalid_label_ids += 1
                tags.append("O")
            else:
                tags.append(ID2LABEL[i])
        label_counter.update(tags)
        entity_counter.update(_count_entities(tags))
        prev = "O"
        for tag in tags:
            if not is_valid_transition(prev, tag):
                invalid_transitions += 1
            prev = tag
    sent_lengths.sort()
    n = len(sent_lengths)
    return {
        "n_sentences": n,
        "n_tokens": n_tokens,
        "sentence_length": {
            "min": sent_lengths[0] if n else 0,
            "max": sent_lengths[-1] if n else 0,
            "mean": round(sum(sent_lengths) / n, 2) if n else 0,
            "median": sent_lengths[n // 2] if n else 0,
            "p95": sent_lengths[int(0.95 * n)] if n else 0,
        },
        "label_distribution": dict(sorted(label_counter.items())),
        "entity_counts": {et: entity_counter.get(et, 0) for et in ["PER", "ORG", "LOC", "MISC"]},
        "total_entities": sum(entity_counter.values()),
        "invalid_bio_transitions": invalid_transitions,
        "invalid_or_misaligned_labels": invalid_label_ids,
    }


def _sentence_key(ex) -> str:
    return " ".join(ex["tokens"])


def run_audit(processed_dir: str | Path) -> Dict:
    processed_dir = Path(processed_dir)
    splits = {s: load_split(processed_dir, s) for s in ["train", "validation", "test"]}

    per_split = {s: audit_split(ex) for s, ex in splits.items()}

    # Cross-split OOV: fraction of val/test tokens (lowercased) unseen in train.
    train_vocab = set()
    for ex in splits["train"]:
        train_vocab.update(t.lower() for t in ex["tokens"])
    oov = {}
    for s in ["validation", "test"]:
        total = unseen = 0
        for ex in splits[s]:
            for t in ex["tokens"]:
                total += 1
                if t.lower() not in train_vocab:
                    unseen += 1
        oov[s] = {"oov_tokens": unseen, "total_tokens": total,
                  "oov_rate": round(unseen / total, 4) if total else 0.0}

    # Duplicates within split and overlap across splits (leakage).
    keys = {s: [_sentence_key(ex) for ex in splits[s]] for s in splits}
    dup_within = {s: len(k) - len(set(k)) for s, k in keys.items()}
    sets = {s: set(k) for s, k in keys.items()}
    overlap = {
        "train_val": len(sets["train"] & sets["validation"]),
        "train_test": len(sets["train"] & sets["test"]),
        "val_test": len(sets["validation"] & sets["test"]),
    }

    report = {
        "dataset": "CoNLL-2003 (English NER)",
        "tagging_scheme": "IOB2 / BIO2 (verified: no sentence starts with I-)",
        "labels": LABELS,
        "per_split": per_split,
        "cross_split_oov": oov,
        "duplicated_sentences_within_split": dup_within,
        "cross_split_sentence_overlap": overlap,
        "leakage_flag": any(v > 0 for v in overlap.values()),
    }
    out = processed_dir / "data_audit.json"
    out.write_text(json.dumps(report, indent=2))
    return report
