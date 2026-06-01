#!/usr/bin/env bash
# Read the Docs post_checkout hook.
#
# 1. Maps the project's RTD language code to a docs/<lang>/ source tree.
# 2. Cancels PR-preview builds that don't touch this language's docs.
# 3. Creates docs/active -> docs/<lang>/ so the declarative sphinx/python
#    blocks in .readthedocs.yaml don't need to know which language to build.
#
# Kept as a separate script (rather than inline in .readthedocs.yaml) because
# RTD's command runner has been observed to silently abort on multi-line
# YAML scalars with case/if/$()-style constructs.

set -eu

case "$READTHEDOCS_LANGUAGE" in
  en)          lang="en" ;;
  zh-cn|zh_CN) lang="zh" ;;
  *)
    echo "Unrecognised READTHEDOCS_LANGUAGE=[${READTHEDOCS_LANGUAGE}]; aborting."
    exit 1
    ;;
esac
echo "Building docs/${lang}/ for project ${READTHEDOCS_PROJECT} (${READTHEDOCS_LANGUAGE})"

# PR-preview gate. RTD treats exit code 183 as "cancel build" (not fail).
if [ "${READTHEDOCS_VERSION_TYPE:-}" = "external" ]; then
  git fetch --depth=100 origin main:refs/remotes/origin/main 2>/dev/null || true
  base="$(git merge-base origin/main HEAD 2>/dev/null || git rev-parse origin/main 2>/dev/null || true)"
  if [ -n "$base" ] && git diff --quiet "$base" HEAD -- \
        "docs/${lang}/" .readthedocs.yaml .readthedocs/post_checkout.sh; then
    echo "No relevant changes in this PR; cancelling build."
    exit 183
  fi
fi

ln -sfn "$lang" docs/active
ls -la docs/active
