#!/usr/bin/env bash
# Don't use set -e because pytest exit code 5 (no tests collected) should not fail the script

# Run fast tests (excludes conformance, compat, benchmark, e2e).
#
# --benchmark-skip drops any test that requests the pytest-benchmark `benchmark`
# fixture even when it isn't tagged @pytest.mark.benchmark and doesn't live in a
# benchmark file, so the marker/path filters miss it (e.g.
# small-mcap/tests/test_remapper.py, mcap-ros2-support-fast/tests/test_micro_benchmark.py).
#
# pymcap-cli is the only suite large enough to profit from xdist (-n auto). The
# other suites finish in well under a second, so worker startup would cost more
# than it saves — they run serially.

echo "Running fast tests..."
echo

run_pkg() {
    local dir="$1"
    shift
    [ -d "$dir/tests" ] || return 0
    echo "Testing $dir..."
    uv run --frozen pytest "$dir/tests" \
        -m "not (conformance or compat or benchmark or e2e)" \
        --ignore="$dir/tests/test_benchmark.py" \
        --ignore="$dir/tests/benchmark" \
        --benchmark-skip \
        --no-cov \
        -q \
        "$@"
    local code=$?
    # Exit code 5 means no tests collected (OK for packages with only slow tests)
    if [ "$code" -ne 0 ] && [ "$code" -ne 5 ]; then
        echo "❌ $dir tests failed with exit code $code"
        exit "$code"
    fi
    echo
}

run_pkg small-mcap
run_pkg mcap-codec-support
run_pkg pymcap-cli -n auto
run_pkg mcap-ros2-support-fast
run_pkg ros-parser
run_pkg pointcloud2
run_pkg digitalis

echo "✓ Fast tests complete!"
exit 0
