#!/usr/bin/env bash
set -e

# This script initializes git submodules for test data
# Test data is now managed as git submodules instead of being downloaded separately

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "Initializing git submodules for test data..."

# Initialize and update submodules
git submodule update --init --recursive

# Pull LFS files from the mcap submodule
echo "Pulling Git LFS files from mcap submodule..."
cd data/mcap
git lfs install
git lfs pull
cd "$REPO_ROOT"

echo "Test data initialization complete!"
echo ""
echo "Note: For new clones, you can use 'git clone --recurse-submodules' to automatically initialize submodules."
