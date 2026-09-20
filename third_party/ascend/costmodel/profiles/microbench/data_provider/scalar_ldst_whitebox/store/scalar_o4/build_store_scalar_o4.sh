#!/usr/bin/env bash
set -eo pipefail
# Build the 4-op scalar-store CAModel probes.
cd "$(dirname "$0")"
source ~/env_ascend.sh
INC="${INC:-$HOME/AscendNPU-IR-triton/bishengir/lib/Template/include}"

ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
  -I"$INC" store_scalar_o4.cce -o store_scalar_o4.o

g++ -O2 store_scalar_o4_host.cpp -o store_scalar_o4_host \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" -lruntime -lascendcl

echo "BUILD DONE: $(pwd)/store_scalar_o4.o $(pwd)/store_scalar_o4_host"
