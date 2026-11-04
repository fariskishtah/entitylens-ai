"""Train one recurrent NER model (LSTM / BiLSTM / BiLSTM+CRF).

    python scripts/train_rnn.py --config configs/bilstm_crf.yaml \
        --out artifacts/bilstm_crf [--smoke] [--resume]

``--smoke`` trims the data and epochs for a fast CPU/MPS sanity run.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entitylens.config import load_config  # noqa: E402
from entitylens.data.dataset import load_split  # noqa: E402
from entitylens.data.embeddings import build_embedding_matrix  # noqa: E402
from entitylens.data.rnn_dataset import RNNNerDataset, make_collate  # noqa: E402
from entitylens.data.vocab import build_vocabularies  # noqa: E402
from entitylens.models.factory import build_rnn_model  # noqa: E402
from entitylens.training.rnn_trainer import train_model  # noqa: E402
from entitylens.training.utils import set_seed  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    cfg = load_config(args.config, args.base)
    set_seed(cfg.get("seed", 42))
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    processed = ROOT / cfg["processed_dir"]
    raw = ROOT / cfg["raw_dir"]

    train_ex = load_split(processed, "train")
    val_ex = load_split(processed, "validation")
    if args.smoke:
        train_ex, val_ex = train_ex[:400], val_ex[:200]
        cfg["epochs"] = min(cfg.get("epochs", 3), 2)
        cfg["glove_dim"] = cfg["word_emb_dim"] = 50

    vocabs = build_vocabularies(train_ex, lower=cfg.get("lower", True),
                                max_word_len=cfg.get("max_word_len", 20))
    vocab_path = out_dir / "vocab.json"
    vocabs.save(vocab_path)

    pretrained = None
    if not args.smoke:
        matrix, stats = build_embedding_matrix(
            vocabs.word.itos, raw, dim=cfg.get("glove_dim", 100),
            cache_dir=processed)
        pretrained = torch.tensor(matrix)
        (out_dir / "embedding_stats.json").write_text(json.dumps(stats, indent=2))
        print("GloVe:", stats)

    model = build_rnn_model(cfg, vocabs, pretrained=pretrained)
    collate = make_collate(vocabs.word.pad_id, vocabs.char.pad_id,
                           vocabs.max_word_len)
    bs = cfg.get("batch_size", 32)
    train_loader = DataLoader(RNNNerDataset(train_ex, vocabs), batch_size=bs,
                              shuffle=True, collate_fn=collate)
    val_loader = DataLoader(RNNNerDataset(val_ex, vocabs), batch_size=bs,
                            shuffle=False, collate_fn=collate)

    summary = train_model(model, train_loader, val_loader, cfg, out_dir,
                          vocab_path, resume=args.resume)
    print("DONE:", json.dumps({k: summary[k] for k in
          ("arch", "best_val_f1", "best_epoch", "train_time_sec", "trainable_params")}))


if __name__ == "__main__":
    main()
