"""Push and run the Transformer training notebook on Kaggle GPU, headlessly.

Needs ~/.kaggle/kaggle.json (Kaggle -> Settings -> API -> Create New Token).
The code + frozen CoNLL-2003 parquet splits travel as a PRIVATE Kaggle dataset,
so the kernel needs no GitHub token and no secrets.

    python scripts/kaggle_launch.py --push      # create/update dataset + kernel, start the GPU run
    python scripts/kaggle_launch.py --status    # poll kernel status
    python scripts/kaggle_launch.py --pull      # download kernel output into artifacts/transformer

Nothing sensitive is uploaded: only configs, src, scripts, notebooks,
requirements and the frozen parquet/json under data/processed.
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CODE_DATASET = "entitylens-ai-code"
KERNEL = "entitylens-transformer-training"
NOTEBOOK = ROOT / "notebooks" / "kaggle_train_transformer.ipynb"

INCLUDE_DIRS = ["configs", "src", "scripts", "notebooks"]
INCLUDE_FILES = ["requirements.txt"]
# frozen data needed by the kernel
DATA_GLOBS = ["data/processed/*.parquet", "data/processed/*.json"]

KAGGLE_BIN = str(Path(sys.executable).with_name("kaggle"))


def _kaggle(*args, check=True):
    cmd = [KAGGLE_BIN if Path(KAGGLE_BIN).exists() else "kaggle", *args]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.stdout.strip():
        print(r.stdout.strip())
    if r.returncode != 0 and r.stderr.strip():
        print(r.stderr.strip(), file=sys.stderr)
    if check and r.returncode != 0:
        raise SystemExit(f"kaggle {' '.join(args)} failed")
    return r


def _username() -> str:
    cfg = Path.home() / ".kaggle" / "kaggle.json"
    if not cfg.exists():
        raise SystemExit("Missing ~/.kaggle/kaggle.json (Kaggle > Settings > API > Create New Token).")
    return json.loads(cfg.read_text())["username"]


def _stage_code(tmp: Path):
    for d in INCLUDE_DIRS:
        shutil.copytree(ROOT / d, tmp / d,
                        ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    for f in INCLUDE_FILES:
        shutil.copy(ROOT / f, tmp / f)
    (tmp / "data" / "processed").mkdir(parents=True, exist_ok=True)
    for glob in DATA_GLOBS:
        for src in ROOT.glob(glob):
            shutil.copy(src, tmp / "data" / "processed" / src.name)


def push():
    user = _username()
    with tempfile.TemporaryDirectory() as td:
        tmp = Path(td)
        _stage_code(tmp)
        (tmp / "dataset-metadata.json").write_text(json.dumps({
            "title": CODE_DATASET, "id": f"{user}/{CODE_DATASET}",
            "licenses": [{"name": "CC0-1.0"}],
        }, indent=2))
        existing = _kaggle("datasets", "status", f"{user}/{CODE_DATASET}", check=False)
        if "404" in (existing.stderr + existing.stdout) or existing.returncode != 0:
            _kaggle("datasets", "create", "-p", str(tmp), "--dir-mode", "zip")
        else:
            _kaggle("datasets", "version", "-p", str(tmp), "-m", "update", "--dir-mode", "zip")

    with tempfile.TemporaryDirectory() as td:
        ktmp = Path(td)
        shutil.copy(NOTEBOOK, ktmp / NOTEBOOK.name)
        (ktmp / "kernel-metadata.json").write_text(json.dumps({
            "id": f"{user}/{KERNEL}",
            "title": KERNEL,
            "code_file": NOTEBOOK.name,
            "language": "python", "kernel_type": "notebook",
            "is_private": True, "enable_gpu": True, "enable_internet": True,
            "dataset_sources": [f"{user}/{CODE_DATASET}"],
        }, indent=2))
        _kaggle("kernels", "push", "-p", str(ktmp))
    print(f"\nPushed. Monitor: python scripts/kaggle_launch.py --status\n"
          f"Kernel: https://www.kaggle.com/code/{user}/{KERNEL}")


def status():
    user = _username()
    _kaggle("kernels", "status", f"{user}/{KERNEL}")


def pull():
    user = _username()
    out = ROOT / "artifacts" / "transformer"
    out.mkdir(parents=True, exist_ok=True)
    _kaggle("kernels", "output", f"{user}/{KERNEL}", "-p", str(out))
    print(f"Pulled kernel output into {out}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--status", action="store_true")
    ap.add_argument("--pull", action="store_true")
    args = ap.parse_args()
    if args.push:
        push()
    elif args.status:
        status()
    elif args.pull:
        pull()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
