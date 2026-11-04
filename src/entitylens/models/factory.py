"""Build a recurrent tagger from a config + vocabularies (+ GloVe)."""
from __future__ import annotations

from typing import Dict, Optional

import torch

from ..data.vocab import Vocabularies
from . import BiLSTMCRFTagger, BiLSTMTagger, LSTMTagger

_REGISTRY = {"lstm": LSTMTagger, "bilstm": BiLSTMTagger, "bilstm_crf": BiLSTMCRFTagger}


def build_rnn_model(config: Dict, vocabs: Vocabularies,
                    pretrained: Optional[torch.Tensor] = None):
    name = config["model"]
    if name not in _REGISTRY:
        raise ValueError(f"Unknown RNN model '{name}'. Options: {list(_REGISTRY)}")
    cls = _REGISTRY[name]
    kwargs = dict(
        n_words=len(vocabs.word), n_chars=len(vocabs.char),
        word_emb_dim=config.get("word_emb_dim", 100),
        hidden_dim=config.get("hidden_dim", 200),
        num_layers=config.get("num_layers", 1),
        dropout=config.get("dropout", 0.5),
        use_char=config.get("use_char", True),
        word_pad_id=vocabs.word.pad_id, char_pad_id=vocabs.char.pad_id,
        pretrained=pretrained, freeze_emb=config.get("freeze_emb", False),
    )
    if name == "bilstm_crf":
        kwargs["constrain_transitions"] = config.get("constrain_transitions", True)
    return cls(**kwargs)
