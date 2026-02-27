#!/usr/bin/env bash
set -euo pipefail

# Bump minor version for workspace packages that have code changes since their last version bump.
# Uses the last commit that touched <pkg>/pyproject.toml as a proxy for the last bump.

# Parse workspace members from root pyproject.toml
members=()
in_members=false
while IFS= read -r line; do
    if [[ "$line" =~ ^members ]]; then
        in_members=true
        continue
    fi
    if $in_members; then
        [[ "$line" == "]" ]] && break
        # Extract quoted string
        member=$(echo "$line" | sed 's/.*"\(.*\)".*/\1/')
        members+=("$member")
    fi
done < pyproject.toml

if [ ${#members[@]} -eq 0 ]; then
    echo "No workspace members found in pyproject.toml"
    exit 1
fi

bumped=0
skipped=0

for pkg in "${members[@]}"; do
    if [ ! -f "$pkg/pyproject.toml" ]; then
        echo "⚠ $pkg/pyproject.toml not found, skipping"
        continue
    fi

    # Find the last commit that touched this package's pyproject.toml
    last_bump_commit=$(git log -1 --format=%H -- "$pkg/pyproject.toml" 2>/dev/null || true)

    needs_bump=false

    if [ -z "$last_bump_commit" ]; then
        # No commit ever touched pyproject.toml — first time, needs bump
        needs_bump=true
    else
        # Check committed changes since last bump (exclude pyproject.toml itself)
        committed_diff=$(git diff "$last_bump_commit"..HEAD -- "$pkg/" ":(exclude)$pkg/pyproject.toml" 2>/dev/null || true)
        # Check uncommitted changes — staged + unstaged (exclude pyproject.toml itself)
        uncommitted_diff=$(git diff HEAD -- "$pkg/" ":(exclude)$pkg/pyproject.toml" 2>/dev/null || true)

        if [ -n "$committed_diff" ] || [ -n "$uncommitted_diff" ]; then
            needs_bump=true
        fi
    fi

    if $needs_bump; then
        echo "⬆ Bumping $pkg..."
        uv version --bump patch --package "$pkg" --frozen
        bumped=$((bumped + 1))
    else
        echo "— $pkg: no changes, skipping"
        skipped=$((skipped + 1))
    fi
done

echo
echo "Done: $bumped bumped, $skipped skipped"
