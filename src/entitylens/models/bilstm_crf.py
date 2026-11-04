"""Model 3 — BiLSTM + CRF.

Word+char embeddings -> BiLSTM -> linear emission layer -> linear-chain CRF.
The CRF (``pytorch-crf``) provides a real sequence-level negative-log-likelihood
loss and Viterbi decoding, so the model scores whole tag *sequences* rather than
each token independently.

IOB2 transition constraints: illegal transitions (e.g. ``O -> I-PER``,
``B-ORG -> I-PER``) and illegal starts (``<start> -> I-*``) are forced to a
large negative score.  Because the CRF's transition matrix is learnable, we
re-apply the constraints after every optimizer step (``apply_constraints``),
giving a genuine hard constraint rather than a one-off initialization.
"""
from __future__ import annotations

from typing import List, Optional

import torch
import torch.nn as nn
from torchcrf import CRF

from ..labels import ID2LABEL, LABELS, NUM_LABELS, PAD_LABEL_ID, is_valid_transition, split_tag
from .components import WordCharEncoder

NEG = -1.0e4


class BiLSTMCRFTagger(nn.Module):
    arch = "bilstm_crf"
    bidirectional = True

    def __init__(self, n_words: int, n_chars: int, num_labels: int = NUM_LABELS,
                 word_emb_dim: int = 100, hidden_dim: int = 200, num_layers: int = 1,
                 dropout: float = 0.5, use_char: bool = True,
                 constrain_transitions: bool = True,
                 pretrained: Optional[torch.Tensor] = None, **kw):
        super().__init__()
        self.encoder = WordCharEncoder(
            n_words, n_chars, word_emb_dim=word_emb_dim, hidden_dim=hidden_dim,
            num_layers=num_layers, bidirectional=True, dropout=dropout,
            use_char=use_char, pretrained=pretrained, **kw)
        self.classifier = nn.Linear(self.encoder.output_dim, num_labels)
        self.crf = CRF(num_labels, batch_first=True)
        self.num_labels = num_labels
        self.constrain_transitions = constrain_transitions
        self.register_buffer("_trans_illegal", self._build_trans_mask())
        self.register_buffer("_start_illegal", self._build_start_mask())
        if constrain_transitions:
            self.apply_constraints()

    # ---- IOB2 constraint masks ------------------------------------------
    def _build_trans_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_labels, self.num_labels, dtype=torch.bool)
        for i, prev in enumerate(LABELS):
            for j, curr in enumerate(LABELS):
                if not is_valid_transition(prev, curr):
                    mask[i, j] = True
        return mask

    def _build_start_mask(self) -> torch.Tensor:
        mask = torch.zeros(self.num_labels, dtype=torch.bool)
        for j, curr in enumerate(LABELS):
            prefix, _ = split_tag(curr)
            if prefix == "I":  # a sequence may not start on an I- tag
                mask[j] = True
        return mask

    @torch.no_grad()
    def apply_constraints(self) -> None:
        if not self.constrain_transitions:
            return
        self.crf.transitions.data.masked_fill_(self._trans_illegal, NEG)
        self.crf.start_transitions.data.masked_fill_(self._start_illegal, NEG)

    # ---- forward / decode ----------------------------------------------
    def emissions(self, word_ids, char_ids, lengths) -> torch.Tensor:
        return self.classifier(self.encoder(word_ids, char_ids, lengths))

    def forward(self, word_ids, char_ids, lengths, mask, labels=None):
        emissions = self.emissions(word_ids, char_ids, lengths)
        loss = None
        if labels is not None:
            crf_tags = labels.clone()
            crf_tags[labels == PAD_LABEL_ID] = 0  # dummy tag on masked positions
            ll = self.crf(emissions, crf_tags, mask=mask, reduction="mean")
            loss = -ll
        return {"emissions": emissions, "loss": loss}

    @torch.no_grad()
    def decode(self, word_ids, char_ids, lengths, mask) -> List[List[int]]:
        emissions = self.emissions(word_ids, char_ids, lengths)
        return self.crf.decode(emissions, mask=mask)  # Viterbi, variable length
