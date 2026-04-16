#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OPERATOR_PY="/home/pc/miniconda3/envs/operator/bin/python"

if [[ ! -x "$OPERATOR_PY" ]]; then
  echo "operator env python not found: $OPERATOR_PY" >&2
  echo "Please create/fix the conda env first." >&2
  exit 1
fi

cd "$ROOT_DIR"
exec "$OPERATOR_PY" "$ROOT_DIR/pipeline.py" --config pipeline_config.yaml
