#!/bin/bash

set -e

# Change to repository root
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
cd "$REPO_ROOT"

echo "===================="
echo "Type checking all packages with mypy --strict"
echo "===================="
echo ""

PACKAGES=(
    "small-mcap"
    "digitalis"
    "mcap-ros2-support-fast"
    "pymcap-cli"
    "websocket-proxy"
    "websocket-bridge"
    "ros-parser"
)

FAILED_PACKAGES=()
PASSED_PACKAGES=()

for package in "${PACKAGES[@]}"; do
    echo "📦 Checking $package..."
    if [ -d "$package/src" ]; then
        if uv run --frozen --all-extras --all-groups --all-packages --no-progress mypy "$package/src" --strict; then
            PASSED_PACKAGES+=("$package")
            echo "✅ $package passed"
        else
            FAILED_PACKAGES+=("$package")
            echo "❌ $package failed"
        fi
    else
        echo "⚠️  Skipping $package (no src/ directory found)"
    fi
    echo ""
done

echo "===================="
echo "Summary"
echo "===================="
echo "Passed: ${#PASSED_PACKAGES[@]}"
for pkg in "${PASSED_PACKAGES[@]}"; do
    echo "  ✅ $pkg"
done
echo ""
echo "Failed: ${#FAILED_PACKAGES[@]}"
for pkg in "${FAILED_PACKAGES[@]}"; do
    echo "  ❌ $pkg"
done
echo ""

if [ ${#FAILED_PACKAGES[@]} -gt 0 ]; then
    exit 1
else
    echo "🎉 All packages passed type checking!"
    exit 0
fi
