"""Word <-> subword label alignment for the Transformer.

Uses the ``word_ids()`` offsets from a fast tokenizer.  The *first* subword of
each word carries the real BIO label; all remaining subwords and special tokens
get ``PAD_LABEL_ID`` so they are ignored by the loss and excluded from metrics.
At inference we read the label back from that first subword, which guarantees
predicted spans map onto the original words with no subword leakage.
"""
from __future__ import annotations

from typing import Dict, List

from ..labels import PAD_LABEL_ID


def tokenize_and_align_labels(examples: Dict[str, List], tokenizer,
                              max_length: int = 256,
                              label_all_subwords: bool = False) -> Dict:
    """Batch map fn for a HF dataset with ``tokens`` + ``ner_tags`` columns."""
    tokenized = tokenizer(
        examples["tokens"],
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    aligned_labels = []
    for i, labels in enumerate(examples["ner_tags"]):
        word_ids = tokenized.word_ids(batch_index=i)
        prev = None
        label_ids = []
        for wid in word_ids:
            if wid is None:
                label_ids.append(PAD_LABEL_ID)
            elif wid != prev:
                label_ids.append(labels[wid])
            else:
                if label_all_subwords:
                    lab = labels[wid]
                    # convert B- to I- for continuation subwords
                    label_ids.append(lab + 1 if 1 <= lab and lab % 2 == 1 else lab)
                else:
                    label_ids.append(PAD_LABEL_ID)
            prev = wid
        aligned_labels.append(label_ids)
    tokenized["labels"] = aligned_labels
    return tokenized
