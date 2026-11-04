"""Reproducibility, device selection and id<->tag helpers."""
from __future__ import annotations

import os
import random
from typing import List

import numpy as np
import torch

from ..labels import ID2LABEL


def set_seed(seed: int = 42) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def select_device(prefer: str = "auto") -> torch.device:
    if prefer not in ("auto", "cpu"):
        return torch.device(prefer)
    if prefer == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def ids_to_tags(id_seqs: List[List[int]]) -> List[List[str]]:
    return [[ID2LABEL[i] for i in seq] for seq in id_seqs]


def count_params(model) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
