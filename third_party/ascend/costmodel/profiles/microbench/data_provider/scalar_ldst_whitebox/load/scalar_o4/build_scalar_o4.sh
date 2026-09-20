#!/usr/bin/env bash
set -eo pipefail
# Build the 4-op scalar-load CAModel probes.
cd "$(dirname "$0")"
source ~/env_ascend.sh
INC="${INC:-$HOME/AscendNPU-IR-triton/bishengir/lib/Template/include}"

ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
  -I"$INC" load_scalar_o4.cce -o load_scalar_o4.o

g++ -O2 load_scalar_o4_host.cpp -o load_scalar_o4_host \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" -lruntime -lascendcl

echo "BUILD DONE: $(pwd)/load_scalar_o4.o $(pwd)/load_scalar_o4_host"
