#!/usr/bin/env bash
set -euo pipefail

if [[ $# -ne 1 || ! -f "$1" ]]; then
  echo "::error::Expected one LLVM SDK tarball in the downloaded artifact" >&2
  exit 1
fi

install_dir=$(mktemp -d "$RUNNER_TEMP/llvm.XXXXXX")
tar -xzf "$1" -C "$install_dir" --strip-components=1
if [[ ! -x "$install_dir/bin/llvm-config" ]]; then
  echo "::error::LLVM artifact is missing executable bin/llvm-config" >&2
  exit 1
fi

# Keep the downloaded tarball out of the wheel's Docker build context.
rm -- "$1"
echo "LLVM_SYSPATH=$install_dir" >> "$GITHUB_ENV"
echo "path=$install_dir" >> "$GITHUB_OUTPUT"
