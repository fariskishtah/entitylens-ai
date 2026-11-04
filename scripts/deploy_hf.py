"""Assemble and (optionally) push the EntityLens AI Hugging Face Space.

    python scripts/deploy_hf.py --build                 # stage ./deploy/space
    python scripts/deploy_hf.py --push --private        # create + upload PRIVATE Space

The Space bundles: app.py, src/, requirements.txt, results/model_comparison.json
and the selected production model copied to artifacts/production/. Deployment is
kept PRIVATE by default; pushing requires HF_TOKEN (env or cached login) and is
only done on explicit request.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SPACE = ROOT / "deploy" / "space"


def _production_dir() -> Path:
    env = os.environ.get("ENTITYLENS_MODEL_DIR")
    if env and Path(env).exists():
        return Path(env)
    prod = ROOT / "artifacts" / "production"
    if prod.exists():
        return prod
    comp = ROOT / "results" / "model_comparison.json"
    if comp.exists():
        name = json.loads(comp.read_text())["selected_model"]
        mapping = {"LSTM": "lstm", "BiLSTM": "bilstm", "BiLSTM+CRF": "bilstm_crf",
                   "Transformer": "transformer/model"}
        cand = ROOT / "artifacts" / mapping.get(name, "")
        if cand.exists():
            return cand
    raise SystemExit("No production model found. Train + select first, or set ENTITYLENS_MODEL_DIR.")


def build() -> Path:
    if SPACE.exists():
        shutil.rmtree(SPACE)
    SPACE.mkdir(parents=True)
    shutil.copy(ROOT / "app.py", SPACE / "app.py")
    shutil.copy(ROOT / "requirements.txt", SPACE / "requirements.txt")
    shutil.copytree(ROOT / "src", SPACE / "src",
                    ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    shutil.copy(ROOT / "deploy" / "space_readme.md", SPACE / "README.md")
    (SPACE / "results").mkdir(exist_ok=True)
    comp = ROOT / "results" / "model_comparison.json"
    if comp.exists():
        shutil.copy(comp, SPACE / "results" / "model_comparison.json")
    prod = _production_dir()
    dest = SPACE / "artifacts" / "production"
    dest.mkdir(parents=True)
    for f in prod.iterdir():
        if f.is_file():
            shutil.copy(f, dest / f.name)
    print(f"Staged Space at {SPACE} (model from {prod})")
    return SPACE


def push(private: bool) -> None:
    from huggingface_hub import HfApi, create_repo
    build()
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    user = api.whoami()["name"]
    repo_id = f"{user}/entitylens-ai"
    create_repo(repo_id, repo_type="space", space_sdk="gradio",
                private=private, exist_ok=True, token=token)
    api.upload_folder(folder_path=str(SPACE), repo_id=repo_id, repo_type="space")
    print(f"Pushed {'PRIVATE ' if private else ''}Space: https://huggingface.co/spaces/{repo_id}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--push", action="store_true")
    ap.add_argument("--private", action="store_true", default=True)
    ap.add_argument("--public", dest="private", action="store_false")
    args = ap.parse_args()
    if args.push:
        push(args.private)
    else:
        build()


if __name__ == "__main__":
    main()
