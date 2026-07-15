#!/usr/bin/env bash
set -euo pipefail

# Bump minor versions for workspace packages with releasable changes since their last release.
# Uses per-package git tags (pkg@version) as the reference point for detecting changes.
# Package-local tests do not change the published artifact and are ignored.
# Guards against double-bumps before a prepared release receives its tag.

version_is_newer() {
    local current="$1"
    local previous="$2"
    local current_major current_minor current_patch
    local previous_major previous_minor previous_patch

    if [[ ! "$current" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    current_major="${BASH_REMATCH[1]}"
    current_minor="${BASH_REMATCH[2]}"
    current_patch="${BASH_REMATCH[3]}"

    if [[ ! "$previous" =~ ^([0-9]+)\.([0-9]+)\.([0-9]+)$ ]]; then
        return 1
    fi
    previous_major="${BASH_REMATCH[1]}"
    previous_minor="${BASH_REMATCH[2]}"
    previous_patch="${BASH_REMATCH[3]}"

    if ((current_major != previous_major)); then
        if ((current_major > previous_major)); then
            return 0
        fi
        return 1
    fi
    if ((current_minor != previous_minor)); then
        if ((current_minor > previous_minor)); then
            return 0
        fi
        return 1
    fi
    if ((current_patch > previous_patch)); then
        return 0
    fi
    return 1
}

strip_project_version() {
    awk '
        !removed && /^[[:space:]]*version[[:space:]]*=/ { removed = 1; next }
        { print }
    '
}

pyproject_changed_since() {
    local reference="$1"
    local pkg="$2"

    ! cmp -s \
        <(git show "$reference:$pkg/pyproject.toml" | strip_project_version) \
        <(strip_project_version < "$pkg/pyproject.toml")
}

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

skipped=0
to_bump=()

for pkg in "${members[@]}"; do
    if [ ! -f "$pkg/pyproject.toml" ]; then
        echo "⚠ $pkg/pyproject.toml not found, skipping"
        continue
    fi

    # Read current version from pyproject.toml
    current_version=$(grep -m1 '^version' "$pkg/pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')
    latest_tag=$(
        git for-each-ref \
            --sort=-version:refname \
            --count=1 \
            --format='%(refname:short)' \
            "refs/tags/$pkg@*"
    )

    if [ -n "$latest_tag" ]; then
        latest_version="${latest_tag#"$pkg@"}"
    else
        latest_version=""
    fi

    if [ -n "$latest_version" ] && version_is_newer "$current_version" "$latest_version"; then
        echo "— $pkg: version $current_version is already awaiting a tag, skipping"
        skipped=$((skipped + 1))
        continue
    fi

    needs_bump=false
    change_paths=(
        "$pkg/"
        ":(exclude)$pkg/pyproject.toml"
        ":(exclude)$pkg/tests"
        ":(exclude)$pkg/tests/**"
    )

    if [ -z "$latest_tag" ]; then
        # No tag for current version — any releasable package history needs a release.
        initial_release_paths=(
            "$pkg/"
            ":(exclude)$pkg/tests"
            ":(exclude)$pkg/tests/**"
        )
        if [ -n "$(git log -1 --format=%H -- "${initial_release_paths[@]}" 2>/dev/null)" ]; then
            needs_bump=true
        fi
    else
        if ! git diff --quiet "$latest_tag"..HEAD -- "${change_paths[@]}"; then
            needs_bump=true
        fi
        if pyproject_changed_since "$latest_tag" "$pkg"; then
            needs_bump=true
        fi
    fi

    if $needs_bump; then
        echo "⬆ $pkg ($current_version) has changes"
        to_bump+=("$pkg")
    else
        echo "— $pkg: no changes, skipping"
        skipped=$((skipped + 1))
    fi
done

echo
if [ ${#to_bump[@]} -eq 0 ]; then
    echo "Nothing to bump ($skipped skipped)"
    exit 0
fi

echo "Will bump minor version for: ${to_bump[*]}"
read -rp "Proceed? [y/N] " confirm
if [[ ! "$confirm" =~ ^[Yy]$ ]]; then
    echo "Aborted."
    exit 0
fi

echo
bumped_pkgs=()
for pkg in "${to_bump[@]}"; do
    echo "⬆ Bumping $pkg..."
    uv version --bump minor --package "$pkg" --frozen
    bumped_pkgs+=("$pkg")
done

echo
echo "Done: ${#bumped_pkgs[@]} bumped, $skipped skipped"

uv lock

# Commit version bumps
git add uv.lock
for pkg in "${bumped_pkgs[@]}"; do
    git add "$pkg/pyproject.toml"
done
git commit -m "chore: bump versions for ${bumped_pkgs[*]}"
