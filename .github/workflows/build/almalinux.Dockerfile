# https://github.com/AlmaLinux/container-images/blob/9f9b3c8c8cf4a57fd42f362570ff47c75788031f/default/amd64/Dockerfile
ARG BASE_IMAGE=almalinux:8.10-20250411
FROM ${BASE_IMAGE}
ARG llvm_dir=llvm-project
# Add the cache artifacts and the LLVM source tree to the container
ADD sccache /sccache
ADD "${llvm_dir}" /source/llvm-project
ENV SCCACHE_DIR="/sccache"
ENV SCCACHE_CACHE_SIZE="2G"

RUN dnf install --assumeyes llvm-toolset
RUN dnf install --assumeyes python39-pip python39-devel git
RUN alternatives --set python3 /usr/bin/python3.9

ENV PIP_INDEX_URL=http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple \
    PIP_TRUSTED_HOST=cache-service.nginx-pypi-cache.svc.cluster.local

RUN python3 -m pip install --upgrade pip
RUN python3 -m pip install --upgrade cmake ninja sccache lit nanobind

# Install MLIR's Python Dependencies
RUN python3 -m pip install -r /source/llvm-project/mlir/python/requirements.txt

# Configure, Build, Test, and Install LLVM
RUN cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_ASM_COMPILER=clang \
  -DCMAKE_CXX_FLAGS="-Wno-everything" \
  -DCMAKE_LINKER=lld \
  -DCMAKE_INSTALL_PREFIX="/install" \
  -Dnanobind_DIR="/usr/local/lib/python3.9/site-packages/nanobind/cmake" \
  -DPython3_EXECUTABLE=/usr/bin/python3.9 \
  -DPython_EXECUTABLE=/usr/bin/python3.9 \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_ENABLE_TERMINFO=OFF \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ZSTD=OFF \
  /source/llvm-project/llvm

RUN ninja -C build install

# ---------------------------------------------------------------------------
# Package build outputs for extraction via --output type=local (no need for
# docker cp / Docker daemon on the runner).
# ---------------------------------------------------------------------------
RUN tar czf /llvm.tar.gz -C / install \
    && tar czf /sccache.tar.gz -C / sccache

FROM scratch AS output
COPY --from=0 /llvm.tar.gz /
COPY --from=0 /sccache.tar.gz /
