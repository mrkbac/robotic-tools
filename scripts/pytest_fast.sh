#!/usr/bin/env bash
# Don't use set -e because pytest exit code 5 (no tests collected) should not fail the script

# Run fast tests (excludes conformance, compat, benchmark, e2e)
# This matches what pre-commit runs

echo "Running fast tests..."
echo

for dir in small-mcap pymcap-cli mcap-ros2-support-fast ros-parser; do
    if [ -d "$dir/tests" ]; then
        echo "Testing $dir..."
        set +e  # Allow pytest to exit with non-zero
        uv run pytest "$dir/tests" \
            -m "not (conformance or compat or benchmark or e2e)" \
            --ignore="$dir/tests/test_benchmark.py" \
            --ignore="$dir/tests/benchmark" \
            --no-cov \
            -v
        code=$?
        set -e  # Re-enable exit on error
        # Exit code 5 means no tests collected (OK for packages with only slow tests)
        # Exit code 0 means success
        if [ $code -ne 0 ] && [ $code -ne 5 ]; then
            echo "❌ Tests failed with exit code $code"
            exit $code
        fi
        echo
    fi
done

echo "✓ Fast tests complete!"
exit 0
