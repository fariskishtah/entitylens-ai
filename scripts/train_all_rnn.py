"""Train all three recurrent models sequentially into artifacts/.

    python scripts/train_all_rnn.py           # full training
    python scripts/train_all_rnn.py --smoke   # fast sanity
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MODELS = ["lstm", "bilstm", "bilstm_crf"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--only", nargs="*", default=MODELS)
    args = ap.parse_args()
    for name in args.only:
        cfg = ROOT / "configs" / f"{name}.yaml"
        out = ROOT / "artifacts" / name
        cmd = [sys.executable, str(ROOT / "scripts" / "train_rnn.py"),
               "--config", str(cfg), "--out", str(out)]
        if args.smoke:
            cmd.append("--smoke")
        print(f"\n===== TRAIN {name} =====", flush=True)
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
