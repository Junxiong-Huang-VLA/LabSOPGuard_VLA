#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

try:
    from dotenv import load_dotenv
except Exception:
    load_dotenv = None

if load_dotenv is not None:
    load_dotenv(PROJECT_ROOT / ".env")

from labsopguard.asr import asr_diagnostics
from labsopguard.embeddings import embedding_diagnostics, get_text_embedding_provider


def main() -> int:
    parser = argparse.ArgumentParser(description="Check Qwen ASR and embedding integration status.")
    parser.add_argument("--live-embedding", action="store_true", help="Call the configured embedding provider once.")
    parser.add_argument("--text", default="pipette transfers reagent into sample bottle")
    args = parser.parse_args()

    payload = {
        "asr": asr_diagnostics(),
        "embedding": embedding_diagnostics(),
    }
    if args.live_embedding:
        provider = get_text_embedding_provider()
        vector = provider.embed(args.text)
        payload["embedding_live_probe"] = {
            "mode": provider.mode,
            "dimension": len(vector),
            "fallback_mode": "hash" in provider.mode or payload["embedding"]["fallback_mode"],
        }

    print(json.dumps(payload, ensure_ascii=False, indent=2))
    if not os.getenv("DASHSCOPE_API_KEY"):
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
