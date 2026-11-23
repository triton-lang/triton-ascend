#!/usr/bin/env bash
# Install the Bisheng package pinned for the CI container's architecture.
set -euo pipefail

manifest=cmake/bisheng-version.txt
if [[ ! -f "$manifest" && ${1:-} != --require-checksum ]]; then
  echo "$manifest not found, skipping"
  exit 0
fi

arch=$(uname -m)
filename=$(sed -n "s/^${arch}: //p" "$manifest" | xargs)
checksum=$(sed -n "s/^${arch}_sha256: //p" "$manifest" | xargs)
if [[ -z "$filename" ]]; then
  echo "Error: $manifest has no entry for arch $arch" >&2
  exit 1
fi
if [[ -z "$checksum" && ${1:-} == --require-checksum ]]; then
  echo "Error: $manifest has no SHA256 for arch $arch" >&2
  exit 1
fi

base_url=https://triton-ascend-artifacts.obs.cn-southwest-2.myhuaweicloud.com/cann
package=$(mktemp ./bisheng.XXXXXXXX)
trap 'rm -f "$package"' EXIT
echo "Downloading $filename for $arch"
curl -fL --connect-timeout 30 --retry 5 --retry-all-errors --retry-delay 10 -C - \
  "$base_url/$filename" -o "$package"
if [[ -n "$checksum" ]]; then
  printf '%s  %s\n' "$checksum" "$package" | sha256sum -c -
else
  echo "Warning: No SHA256 checksum provided, skipping verification"
fi

install_root=/usr/local/bisheng
payload_dir="$install_root/tools/bishengir"
case "$filename" in
  *.run)
    chmod +x "$package"
    "$package" --quiet --install --install-path="$install_root"
    ;;
  *.tar|*.tar.gz|*.tgz)
    # Published payloads have bin/ and lib/ at the archive root.
    mkdir -p "$payload_dir"
    tar -xf "$package" -C "$payload_dir"
    if [[ -d "$payload_dir/bin" ]]; then
      chmod -R a+x "$payload_dir/bin"
    fi
    ;;
  *)
    echo "Error: unsupported Bisheng package format: $filename (expected .run, .tar, .tar.gz or .tgz)" >&2
    exit 1
    ;;
esac

compiler="$payload_dir/bin/bishengir-compile"
if [[ ! -f "$compiler" || ! -x "$compiler" ]]; then
  echo "Error: expected an executable at $compiler after install" >&2
  exit 1
fi
echo "$payload_dir/bin" >> "$GITHUB_PATH"
