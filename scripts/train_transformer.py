"""Fine-tune the Transformer NER model.

    python scripts/train_transformer.py --config configs/transformer.yaml \
        --out artifacts/transformer [--smoke]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entitylens.config import load_config  # noqa: E402
from entitylens.training.transformer_trainer import fine_tune  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--base", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--smoke", action="store_true")
    args = ap.parse_args()
    cfg = load_config(args.config, args.base)
    summary = fine_tune(cfg, ROOT / cfg["processed_dir"], args.out, smoke=args.smoke)
    print("DONE:", json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
