#!/bin/bash
# Fetch the AscendNPU-IR stable branch HEAD (and optionally its nested
# third-party/* submodules) into the destination directory.
#
# This replaces the old submodule-pin flow: the npuir payload now tracks the
# upstream stable branch instead of the gitlink recorded in .gitmodules.
#
# gitcode branch clones are unreliable from the CI pods, but `want <sha>`
# depth-1 fetches are served fine, so resolve stable -> SHA with ls-remote
# first and then fetch that exact SHA (the same pattern the nested-submodule
# initialization has always used).
#
# Usage: fetch-npuir-stable.sh [DEST_DIR] [--skip-nested]
#   DEST_DIR      target directory (default: third_party/ascend/AscendNPU-IR)
#   --skip-nested skip the nested third-party/* submodules (the caller
#                 initializes them itself, e.g. build_npuir.py which has
#                 its own retry handling)
#
# NPUIR_REPO_BRANCH selects what to fetch: a branch name (default: stable,
# resolved to its HEAD via ls-remote) or a full 40-hex commit SHA, which is
# fetched directly — the pinned-wheel job in wheels.yml passes the submodule
# gitlink SHA that way.
#
# All logs go to stderr; the resolved SHA is printed on stdout as the
# last line so callers can capture it for cache keys:
#   NPUIR_SHA=$(bash fetch-npuir-stable.sh)
set -eu -o pipefail

URL="${NPUIR_REPO_URL:-https://gitcode.com/Ascend/AscendNPU-IR.git}"
BRANCH="${NPUIR_REPO_BRANCH:-stable}"
DEST="third_party/ascend/AscendNPU-IR"
WITH_NESTED=1
for arg in "$@"; do
  case "$arg" in
    --skip-nested) WITH_NESTED=0 ;;
    *) DEST="$arg" ;;
  esac
done

# Abort stalled gitcode transfers instead of hanging the job.
export GIT_HTTP_LOW_SPEED_LIMIT="${GIT_HTTP_LOW_SPEED_LIMIT:-1000}"
export GIT_HTTP_LOW_SPEED_TIME="${GIT_HTTP_LOW_SPEED_TIME:-30}"

log() { echo "[fetch-npuir] $*" >&2; }

retry() { # retry <attempts> <cmd...> — retry a network command with backoff.
  local attempts=$1; shift
  local n=1
  until "$@"; do
    if [ "$n" -ge "$attempts" ]; then
      log "'$*' failed after $n attempts"
      return 1
    fi
    n=$((n + 1))
    log "'$*' failed; retrying in 10s (attempt $n/$attempts)"
    sleep 10
  done
}

fetch_at() { # fetch_at <dest> <sha> <url> — depth-1 fetch of an exact SHA.
  local dest=$1 sha=$2 url=$3
  rm -rf "$dest"
  mkdir -p "$(dirname "$dest")"
  git init -q "$dest"
  git -C "$dest" remote add origin "$url"
  retry 5 git -C "$dest" fetch --depth 1 origin "$sha"
  git -C "$dest" checkout -q FETCH_HEAD
}

if printf '%s' "$BRANCH" | grep -qE '^[0-9a-fA-F]{40}$'; then
  # An exact commit SHA (e.g. the submodule gitlink): no ref resolution
  # needed; the depth-1 `want <sha>` fetch below serves it directly.
  SHA="$BRANCH"
  log "pinned SHA: ${SHA}"
else
  SHA=$(retry 3 git ls-remote "$URL" "refs/heads/${BRANCH}" | awk '{print $1}')
  if [ -z "$SHA" ]; then
    log "could not resolve refs/heads/${BRANCH} at ${URL}"
    exit 1
  fi
  log "${BRANCH} HEAD: ${SHA}"
fi

NEED_FETCH=1
if [ -d "${DEST}/.git" ]; then
  CURRENT=$(git -C "$DEST" rev-parse HEAD 2>/dev/null || echo "")
  if [ "$CURRENT" = "$SHA" ]; then
    log "${DEST} already at ${SHA}"
    NEED_FETCH=0
  elif [ -n "$(git -C "$DEST" status --porcelain 2>/dev/null)" ]; then
    # Local-development guard: never silently wipe uncommitted work. CI
    # contexts never hit this (the destination is either absent or a clean
    # checkout of a previous stable HEAD).
    log "ERROR: ${DEST} is at ${CURRENT} with local modifications;" \
        "refusing to overwrite with ${BRANCH} HEAD ${SHA}"
    exit 1
  fi
fi
if [ "$NEED_FETCH" -eq 1 ]; then
  log "fetching ${DEST} @ ${SHA}"
  fetch_at "$DEST" "$SHA" "$URL"
fi

if [ "$WITH_NESTED" -eq 1 ]; then
  # Same as the old npuir-build.yml step: read the nested gitlink SHAs from
  # the fetched tree and pull each one directly (depth-1, by SHA).
  git -C "$DEST" ls-tree HEAD third-party/ | while read -r _ _ sha path; do
    if [ -f "${DEST}/${path}/CMakeLists.txt" ]; then
      log "${path} already present, skipping"
      continue
    fi
    url=$(git -C "$DEST" config -f .gitmodules --get "submodule.${path}.url")
    log "fetching ${path} @ ${sha} from ${url}"
    fetch_at "${DEST}/${path}" "$sha" "$url"
  done
fi

echo "$SHA"
