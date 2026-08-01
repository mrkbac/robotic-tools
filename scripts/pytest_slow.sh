#!/usr/bin/env bash
set -e

# Run all tests including slow ones (conformance, compat, benchmark, e2e).

echo "Running all tests (including slow tests)..."
echo "Note: This requires test data. Run ./scripts/download_test_mcap.sh first."
echo

for package in \
    small-mcap \
    mcap-codec-support \
    pymcap-cli \
    mcap-ros2-support-fast \
    ros-parser \
    pointcloud2 \
    pureini \
    robo-ws-bridge \
    digitalis
do
    echo "Testing $package..."
    uv run --frozen --all-groups --all-extras --all-packages \
        pytest "$package/tests" -v --no-cov
done

echo
echo "✓ All tests complete!"
