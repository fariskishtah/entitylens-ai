"""Shared building blocks for the recurrent taggers.

``CharCNN`` turns each word's characters into a fixed vector (OOV handling:
even words absent from GloVe get a morphology-aware representation).
``WordCharEncoder`` concatenates pretrained word embeddings with the char-CNN
output and runs a (bi)directional LSTM, returning per-token features.
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class CharCNN(nn.Module):
    def __init__(self, n_chars: int, char_emb_dim: int = 30,
                 n_filters: int = 30, kernel_size: int = 3, pad_id: int = 0,
                 dropout: float = 0.25):
        super().__init__()
        self.embedding = nn.Embedding(n_chars, char_emb_dim, padding_idx=pad_id)
        self.conv = nn.Conv1d(char_emb_dim, n_filters, kernel_size,
                              padding=kernel_size // 2)
        self.dropout = nn.Dropout(dropout)
        self.out_dim = n_filters

    def forward(self, char_ids: torch.Tensor) -> torch.Tensor:
        # char_ids: [B, T, W]
        B, T, W = char_ids.shape
        x = self.embedding(char_ids.view(B * T, W))      # [B*T, W, E]
        x = self.dropout(x).transpose(1, 2)              # [B*T, E, W]
        x = self.conv(x)                                 # [B*T, F, W]
        x, _ = x.max(dim=2)                              # max-pool -> [B*T, F]
        return x.view(B, T, self.out_dim)


class WordCharEncoder(nn.Module):
    def __init__(self, n_words: int, n_chars: int, word_emb_dim: int = 100,
                 hidden_dim: int = 200, num_layers: int = 1,
                 bidirectional: bool = True, dropout: float = 0.5,
                 char_emb_dim: int = 30, char_filters: int = 30,
                 use_char: bool = True, word_pad_id: int = 0, char_pad_id: int = 0,
                 pretrained: Optional[torch.Tensor] = None, freeze_emb: bool = False):
        super().__init__()
        self.word_emb = nn.Embedding(n_words, word_emb_dim, padding_idx=word_pad_id)
        if pretrained is not None:
            self.word_emb.weight.data.copy_(pretrained)
            if freeze_emb:
                self.word_emb.weight.requires_grad_(False)
        self.use_char = use_char
        enc_in = word_emb_dim
        if use_char:
            self.char_cnn = CharCNN(n_chars, char_emb_dim, char_filters,
                                    pad_id=char_pad_id, dropout=dropout)
            enc_in += self.char_cnn.out_dim
        self.emb_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(enc_in, hidden_dim, num_layers=num_layers,
                            batch_first=True, bidirectional=bidirectional,
                            dropout=dropout if num_layers > 1 else 0.0)
        self.out_dropout = nn.Dropout(dropout)
        self.output_dim = hidden_dim * (2 if bidirectional else 1)

    def forward(self, word_ids, char_ids, lengths) -> torch.Tensor:
        emb = self.word_emb(word_ids)
        if self.use_char:
            emb = torch.cat([emb, self.char_cnn(char_ids)], dim=-1)
        emb = self.emb_dropout(emb)
        packed = pack_padded_sequence(emb, lengths.cpu(), batch_first=True,
                                      enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = pad_packed_sequence(out, batch_first=True,
                                     total_length=word_ids.size(1))
        return self.out_dropout(out)
