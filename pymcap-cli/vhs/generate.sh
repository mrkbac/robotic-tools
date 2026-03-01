#!/usr/bin/env bash
# Generate all VHS GIFs (run from repo root)
set -euo pipefail

for tape in pymcap-cli/vhs/*.tape; do
  echo "==> $tape"
  vhs "$tape"
done
