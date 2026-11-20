#!/usr/bin/env bash
# Stage a matching cached or downloaded BishengIR payload for the wheel build.
set -euo pipefail

source_dir=$1
expected_sha=$2
if [ -d "$source_dir" ]; then
  bin=$(find "$source_dir" -path '*/bin/bishengir-compile' -type f -print -quit)
  if [ -n "$bin" ]; then
    payload=${bin%/bin/bishengir-compile}
    actual_sha=$(cat "$payload/npuir-sha.txt" 2>/dev/null || true)
    if [ "$actual_sha" != "$expected_sha" ]; then
      echo "Skipping $payload (NPUIR ${actual_sha:-unknown}, expected $expected_sha)"
    elif [ -f "$payload/bin/bishengir-opt" ] && [ -d "$payload/lib" ]; then
      mkdir -p .npuir-payload
      cp -a "$payload/bin" "$payload/lib" .npuir-payload/
      # Downloaded artifact ZIPs lose the executable bits.
      chmod -R a+x .npuir-payload/bin
      echo "Staged BishengIR from $payload (NPUIR $expected_sha)"
      echo "staged=true" >> "$GITHUB_OUTPUT"
      exit 0
    else
      echo "Skipping incomplete BishengIR payload at $payload"
    fi
  fi
fi
echo "staged=false" >> "$GITHUB_OUTPUT"
