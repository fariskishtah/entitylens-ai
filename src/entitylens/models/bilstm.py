"""Model 2 — bidirectional LSTM sequence tagger.

Identical head to the unidirectional LSTM but with a BiLSTM encoder, so each
token's representation sees both left and right context — important for NER,
where boundary decisions (e.g. whether "New" starts a LOC) depend on following
words as much as preceding ones.
"""
from __future__ import annotations

from .lstm import LSTMTagger


class BiLSTMTagger(LSTMTagger):
    arch = "bilstm"
    bidirectional = True
