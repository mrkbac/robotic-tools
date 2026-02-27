#!/bin/bash

set -e

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "===================="
echo "Type checking with ty"
echo "===================="
echo ""

uv run --frozen ty check

echo ""
echo "🎉 All packages passed type checking!"
