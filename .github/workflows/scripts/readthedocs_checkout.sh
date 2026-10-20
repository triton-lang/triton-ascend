#!/usr/bin/env bash
# Read the Docs post_checkout hook.
#
# 1. Maps the project's RTD language code to a docs/<lang>/ source tree.
# 2. Cancels PR-preview builds that don't touch docs/ (safety net — the
#    GitHub Actions pipeline normally prevents this, but RTD could be
#    triggered outside of it).
# 3. Creates docs/active -> docs/<lang>/.
#
# PR-preview gating (docs-spec check, changed-files filter, RTD trigger +
# status reporting) is handled by the GitHub Actions pipeline:
#   .github/workflows/docs-spec-check.yml
#   .github/workflows/docs-rtd-preview.yml
#
# RTD treats exit code 183 as "cancel build" (not fail).

set -eu

# ── 1. Language mapping ────────────────────────────────────────────────────

case "$READTHEDOCS_LANGUAGE" in
  en)          lang="en" ;;
  zh-cn|zh_CN) lang="zh" ;;
  *)
    echo "Unrecognised READTHEDOCS_LANGUAGE=[${READTHEDOCS_LANGUAGE}]; aborting."
    exit 1
    ;;
esac
echo "Building docs/${lang}/ for project ${READTHEDOCS_PROJECT} (${READTHEDOCS_LANGUAGE})"

# ── 2. PR-preview: skip if no docs/ changes ───────────────────────────────

if [ "${READTHEDOCS_VERSION_TYPE:-}" = "external" ]; then
  echo "PR preview build detected."

  # RTD uses shallow clones; fetch with full depth so merge-base is reliable.
  git fetch --depth=0 origin 2>/dev/null || true
  base="$(git merge-base origin/HEAD HEAD 2>/dev/null || true)"

  if [ -n "$base" ] && git diff --quiet "$base" HEAD -- \
      "docs/" \
      "*.md"  \
      ".readthedocs.yaml" \
      ".github/workflows/docs-rtd-preview.yml" \
      ".github/workflows/scripts/readthedocs_checkout.sh"; then
    echo "No docs-affecting changes in this PR; cancelling build."
    exit 183
  fi
  echo "docs-affecting changes detected, proceeding with build."
fi

# ── 3. Symlink docs/active ─────────────────────────────────────────────────

ln -sfn "$lang" docs/active
ls -la docs/active
