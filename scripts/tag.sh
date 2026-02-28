#!/usr/bin/env bash
set -euo pipefail

# Create pkg@version tags for all workspace packages whose current version isn't tagged yet.
# Usage: scripts/tag.sh [--push]

push=false
if [[ "${1:-}" == "--push" ]]; then
    push=true
fi

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
        member=$(echo "$line" | sed 's/.*"\(.*\)".*/\1/')
        members+=("$member")
    fi
done < pyproject.toml

if [ ${#members[@]} -eq 0 ]; then
    echo "No workspace members found in pyproject.toml"
    exit 1
fi

created=0
existed=0
tags_to_push=()

for pkg in "${members[@]}"; do
    if [ ! -f "$pkg/pyproject.toml" ]; then
        echo "⚠ $pkg/pyproject.toml not found, skipping"
        continue
    fi

    version=$(grep -m1 '^version' "$pkg/pyproject.toml" | sed 's/.*"\(.*\)".*/\1/')
    tag="${pkg}@${version}"

    if git rev-parse "$tag" >/dev/null 2>&1; then
        echo "— $tag already exists"
        existed=$((existed + 1))
    else
        git tag "$tag"
        echo "✓ Created $tag"
        tags_to_push+=("$tag")
        created=$((created + 1))
    fi
done

echo
echo "Done: $created created, $existed already existed"

if $push && [ ${#tags_to_push[@]} -gt 0 ]; then
    echo "Pushing ${#tags_to_push[@]} tags to origin..."
    git push origin "${tags_to_push[@]}"
fi
