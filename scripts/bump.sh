#!/usr/bin/env bash
set -euo pipefail

# Bump patch version for workspace packages that have code changes since their last release.
# Uses per-package git tags (pkg@version) as the reference point for detecting changes.
# Guards against double-bumps when pyproject.toml already has an uncommitted version change.

# Ensure working tree is clean before bumping
if [ -n "$(git status --porcelain)" ]; then
    echo "Error: working tree is dirty. Commit or stash your changes first."
    exit 1
fi

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
bumped_pkgs=()

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
        # No tag for current version — check for any committed or uncommitted changes
        committed_files=$(git log --oneline -- "$pkg/" ":(exclude)$pkg/pyproject.toml" 2>/dev/null | head -1 || true)
        uncommitted_diff=$(git diff HEAD -- "$pkg/" ":(exclude)$pkg/pyproject.toml" 2>/dev/null || true)
        if [ -n "$committed_files" ] || [ -n "$uncommitted_diff" ]; then
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
        uv version --bump minor --package "$pkg" --frozen
        bumped=$((bumped + 1))
        bumped_pkgs+=("$pkg")
    else
        echo "— $pkg: no changes, skipping"
        skipped=$((skipped + 1))
    fi
done

echo
echo "Done: $bumped bumped, $skipped skipped"

if [ $bumped -eq 0 ]; then
    exit 0
fi

uv lock

# Commit version bumps
git add uv.lock
for pkg in "${bumped_pkgs[@]}"; do
    git add "$pkg/pyproject.toml"
done
git commit -m "chore: bump versions for ${bumped_pkgs[*]}"
