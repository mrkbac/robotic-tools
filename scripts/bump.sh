#!/usr/bin/env bash
set -euo pipefail

# Bump patch version for workspace packages that have code changes since their last release.
# Uses per-package git tags (pkg@version) as the reference point for detecting changes.
# Guards against double-bumps when pyproject.toml already has an uncommitted version change.

# Run pre-commit checks before bumping to avoid lint failures after version change.
echo "Running pre-commit checks..."
pre-commit run --all-files
echo

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

    # Guard: if pyproject.toml already has an uncommitted version change, skip.
    # This prevents double-bumps when a previous bump wasn't committed yet (e.g. lint hook failed).
    version_diff=$(git diff HEAD -- "$pkg/pyproject.toml" 2>/dev/null | grep -E '^\+version\s*=' || true)
    if [ -n "$version_diff" ]; then
        echo "— $pkg: version already bumped (uncommitted), skipping"
        skipped=$((skipped + 1))
        continue
    fi

    # Read current version from pyproject.toml
    current_version=$(grep -m1 '^version' "$pkg/pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')
    tag="${pkg}@${current_version}"

    # Find the commit the tag points to
    tag_commit=$(git rev-list -1 "$tag" 2>/dev/null || true)

    needs_bump=false

    if [ -z "$tag_commit" ]; then
        # No tag exists — check for any uncommitted changes in the package (excluding pyproject.toml)
        uncommitted_diff=$(git diff HEAD -- "$pkg/" ":(exclude)$pkg/pyproject.toml" 2>/dev/null || true)
        if [ -n "$uncommitted_diff" ]; then
            needs_bump=true
        fi
    else
        # Check committed changes since tag (exclude pyproject.toml itself)
        committed_diff=$(git diff "$tag_commit"..HEAD -- "$pkg/" ":(exclude)$pkg/pyproject.toml" 2>/dev/null || true)
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
uv lock
