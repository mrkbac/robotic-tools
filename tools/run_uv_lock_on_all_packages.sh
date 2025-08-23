#!/bin/bash
set -euo pipefail

fail=0
for package in *; do
    # only run if pyproject.toml exits
    if [ -f "$package/pyproject.toml" ]; then
        echo "Locking $package"
        if ! uv lock --project "$package"; then
            echo "Errors found in $package"
            fail=1
        fi
    fi
done

exit $fail
