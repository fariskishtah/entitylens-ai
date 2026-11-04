"""Download + freeze CoNLL-2003 and run the data audit.

    python scripts/prepare_data.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from entitylens.data.dataset import download_and_freeze  # noqa: E402
from entitylens.data.audit import run_audit  # noqa: E402

PROCESSED = ROOT / "data" / "processed"


def main() -> None:
    prov = download_and_freeze(PROCESSED)
    print("Provenance:", json.dumps(prov["splits"]), "source=", prov["source"])
    report = run_audit(PROCESSED)
    for split, stats in report["per_split"].items():
        print(f"[{split}] sents={stats['n_sentences']} tokens={stats['n_tokens']} "
              f"entities={stats['total_entities']} "
              f"invalid_bio={stats['invalid_bio_transitions']}")
    print("Leakage flag:", report["leakage_flag"],
          "overlap:", report["cross_split_sentence_overlap"])
    print("Wrote:", PROCESSED / "data_audit.json")


if __name__ == "__main__":
    main()
