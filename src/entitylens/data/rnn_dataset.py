"""Torch Dataset + collate for the recurrent models.

Produces, per batch: word ids ``[B, T]``, char ids ``[B, T, W]``, label ids
``[B, T]`` (padded with ``PAD_LABEL_ID``), an attention ``mask [B, T]`` (bool,
True on real tokens) and the true ``lengths``.  The mask keeps padded positions
from contributing to the loss and to the CRF.
"""
from __future__ import annotations

from typing import Dict, List

import torch
from torch.utils.data import Dataset

from ..labels import PAD_LABEL_ID
from .vocab import Vocabularies


class RNNNerDataset(Dataset):
    def __init__(self, examples: List[Dict], vocabs: Vocabularies):
        self.examples = examples
        self.vocabs = vocabs

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        ex = self.examples[idx]
        toks = ex["tokens"]
        word_ids = [self.vocabs.encode_word(t) for t in toks]
        char_ids = [self.vocabs.encode_chars(t) for t in toks]
        labels = list(ex["ner_tags"])
        return {"word_ids": word_ids, "char_ids": char_ids, "labels": labels,
                "length": len(toks)}


def make_collate(pad_word_id: int, pad_char_id: int, max_word_len: int):
    def collate(batch: List[Dict]) -> Dict[str, torch.Tensor]:
        B = len(batch)
        T = max(item["length"] for item in batch)
        word = torch.full((B, T), pad_word_id, dtype=torch.long)
        char = torch.full((B, T, max_word_len), pad_char_id, dtype=torch.long)
        labels = torch.full((B, T), PAD_LABEL_ID, dtype=torch.long)
        mask = torch.zeros((B, T), dtype=torch.bool)
        lengths = torch.zeros(B, dtype=torch.long)
        for i, item in enumerate(batch):
            L = item["length"]
            lengths[i] = L
            word[i, :L] = torch.tensor(item["word_ids"], dtype=torch.long)
            labels[i, :L] = torch.tensor(item["labels"], dtype=torch.long)
            mask[i, :L] = True
            for j, cids in enumerate(item["char_ids"]):
                w = min(len(cids), max_word_len)
                char[i, j, :w] = torch.tensor(cids[:w], dtype=torch.long)
        return {"word_ids": word, "char_ids": char, "labels": labels,
                "mask": mask, "lengths": lengths}

    return collate
