#!/usr/bin/env bash

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Clone test data
mkdir -p data
cd data
git clone https://github.com/foxglove/ros-foxglove-bridge-benchmark-assets.git data --depth 1
