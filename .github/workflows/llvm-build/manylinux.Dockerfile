# manylinux_2_28 is based on AlmaLinux 8, compatible with glibc >= 2.28
# Produces LLVM artifacts that are widely compatible across Linux distributions.
# ARG IMAGE is set by the caller: quay.io/pypa/manylinux_2_28_x86_64 or _aarch64
ARG IMAGE=quay.io/pypa/manylinux_2_28_x86_64
FROM ${IMAGE}

ARG llvm_dir=llvm-project

# Add the cache artifacts and the LLVM source tree to the container
ADD sccache /sccache
ADD "${llvm_dir}" /source/llvm-project

ENV SCCACHE_DIR="/sccache"
ENV SCCACHE_CACHE_SIZE="2G"

# Python 3.10 is pre-installed; use it for build tooling
ENV PYTHON=/opt/python/cp310-cp310/bin/python3.10

RUN dnf install --assumeyes llvm-toolset && \
    dnf clean all

RUN ${PYTHON} -m pip install --upgrade pip && \
    ${PYTHON} -m pip install --upgrade cmake ninja sccache lit

# Install MLIR's Python dependencies
RUN ${PYTHON} -m pip install -r /source/llvm-project/mlir/python/requirements.txt

# Symlink python3, ninja and sccache so cmake can find them
RUN ln -sf ${PYTHON} /usr/local/bin/python3 && \
    ln -sf /opt/python/cp310-cp310/bin/ninja /usr/local/bin/ninja && \
    ln -sf /opt/python/cp310-cp310/bin/sccache /usr/local/bin/sccache

# Configure, Build, and Install LLVM
RUN cmake -GNinja -Bbuild \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=clang \
  -DCMAKE_CXX_COMPILER=clang++ \
  -DCMAKE_ASM_COMPILER=clang \
  -DCMAKE_C_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_COMPILER_LAUNCHER=sccache \
  -DCMAKE_CXX_FLAGS="-Wno-everything" \
  -DCMAKE_LINKER=lld \
  -DCMAKE_INSTALL_PREFIX="/install" \
  -DPython3_EXECUTABLE="${PYTHON}" \
  -DPython_EXECUTABLE="${PYTHON}" \
  -DLLVM_BUILD_UTILS=ON \
  -DLLVM_BUILD_TOOLS=ON \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_ENABLE_BINDINGS_PYTHON=OFF \
  -DLLVM_ENABLE_PROJECTS="mlir;lld" \
  -DLLVM_INSTALL_UTILS=ON \
  -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
  -DLLVM_ENABLE_ZSTD=OFF \
  /source/llvm-project/llvm

RUN ninja -C build install
