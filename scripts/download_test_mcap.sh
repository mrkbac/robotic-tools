#!/usr/bin/env bash
set -e

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

# Create data directory
mkdir -p data

# Download nuScenes benchmark data
echo "Downloading nuScenes benchmark data..."
if [ ! -d "data/data" ]; then
    cd data
    git clone https://github.com/foxglove/ros-foxglove-bridge-benchmark-assets.git data --depth 1
    cd ..
else
    echo "nuScenes data already exists, skipping..."
fi

# Download MCAP conformance test data
echo "Downloading MCAP conformance test data..."
if [ ! -d "data/conformance" ]; then
    cd data

    # Clone with sparse checkout (only conformance data folder)
    git clone --filter=blob:none --sparse --depth 1 \
        https://github.com/foxglove/mcap.git conformance-repo

    cd conformance-repo
    git sparse-checkout set tests/conformance/data

    # Pull LFS files
    git lfs install
    git lfs pull

    # Move conformance data to parent and clean up
    cd ..
    mv conformance-repo/tests/conformance/data conformance
    rm -rf conformance-repo

    cd ..
    echo "Conformance data downloaded successfully!"
else
    echo "Conformance data already exists, skipping..."
fi

echo "Test data download complete!"
