#!/usr/bin/env bash
set -euo pipefail

python -m key_action_indexer.cli run \
  --manifest examples/session_manifest.example.json \
  --dry-run

python -m key_action_indexer.cli query \
  --index-dir data/sessions/exp_20260429_172500/index \
  --query "找一下使用移液枪加样的片段" \
  --top-k 3
