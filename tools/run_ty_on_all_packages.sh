#!/bin/bash
set -euo pipefail

fail=0
for package in *; do
    if [ -f "$package/pyproject.toml" ]; then
        echo "Checking $package"
        if ! uv run --project "$package" --no-progress --frozen --all-extras --all-groups ty check --project "$package" "$package/src"; then
            echo "Error in $package"
            fail=1
        fi
    fi
done

exit $fail
