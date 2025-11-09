#!/usr/bin/env bash
set -e

# Run all tests including slow ones (conformance, compat, benchmark, e2e)

echo "Running all tests (including slow tests)..."
echo "Note: This requires test data. Run ./scripts/download_test_mcap.sh first."
echo

uv run pytest small-mcap/tests -v --cov=small_mcap
uv run pytest pymcap-cli/tests -v --cov=pymcap_cli --cov-append
uv run pytest mcap-ros2-support-fast/tests -v --cov=mcap_ros2_support_fast --cov-append
uv run pytest ros-parser/tests -v --cov=ros_parser --cov-append

echo
echo "✓ All tests complete!"
echo
echo "Coverage report:"
uv run coverage report
