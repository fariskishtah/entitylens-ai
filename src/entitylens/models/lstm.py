"""Model 1 — word-level (unidirectional) LSTM sequence tagger."""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn

from ..labels import NUM_LABELS, PAD_LABEL_ID
from .components import WordCharEncoder


class LSTMTagger(nn.Module):
    arch = "lstm"
    bidirectional = False

    def __init__(self, n_words: int, n_chars: int, num_labels: int = NUM_LABELS,
                 word_emb_dim: int = 100, hidden_dim: int = 200, num_layers: int = 1,
                 dropout: float = 0.5, use_char: bool = True,
                 pretrained: Optional[torch.Tensor] = None, **kw):
        super().__init__()
        self.encoder = WordCharEncoder(
            n_words, n_chars, word_emb_dim=word_emb_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, bidirectional=self.bidirectional, dropout=dropout,
            use_char=use_char, pretrained=pretrained, **kw)
        self.classifier = nn.Linear(self.encoder.output_dim, num_labels)
        self.loss_fn = nn.CrossEntropyLoss(ignore_index=PAD_LABEL_ID)
        self.num_labels = num_labels

    def emissions(self, word_ids, char_ids, lengths) -> torch.Tensor:
        feats = self.encoder(word_ids, char_ids, lengths)
        return self.classifier(feats)  # [B, T, C]

    def forward(self, word_ids, char_ids, lengths, mask=None, labels=None):
        logits = self.emissions(word_ids, char_ids, lengths)
        loss = None
        if labels is not None:
            loss = self.loss_fn(logits.view(-1, self.num_labels), labels.view(-1))
        return {"logits": logits, "loss": loss}

    @torch.no_grad()
    def decode(self, word_ids, char_ids, lengths, mask) -> List[List[int]]:
        logits = self.emissions(word_ids, char_ids, lengths)
        preds = logits.argmax(-1)
        return [preds[i, :int(lengths[i])].tolist() for i in range(len(lengths))]
