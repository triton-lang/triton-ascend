ARG CANN_BASE_IMAGE=quay.io/ascend/cann:9.0.0-a3-ubuntu22.04-py3.11
ARG PIP_SOURCE="https://mirrors.ustc.edu.cn/pypi/simple"
FROM ${CANN_BASE_IMAGE}
ARG PIP_SOURCE
SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/shanghai"

RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.huaweicloud.com|g' /etc/apt/sources.list 2>/dev/null || true && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.huaweicloud.com|g' /etc/apt/sources.list 2>/dev/null || true && \
    echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries && \
    apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    bash \
    ca-certificates \
    clang-15 \
    clang-format \
    cmake \
    ccache \
    curl \
    git \
    gnupg \
    lld-15 \
    make \
    ninja-build \
    sudo \
    unzip \
    vim \
    wget \
    libzstd-dev \
    zlib1g-dev && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 100 && \
    update-ca-certificates

RUN mkdir -p /root/.config/pip && \
    echo -e "[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple\nextra-index-url = https://download.pytorch.org/whl/cpu" > /root/.config/pip/pip.conf

# Stage 2: install llvm
RUN pip3 install --no-cache-dir \
    ninja \
    cmake \
    wheel \
    pybind11 \
    --index-url ${PIP_SOURCE}

RUN \
    git config --global http.postBuffer 524288000 && \
    git config --global http.lowSpeedLimit 0 && \
    git config --global http.lowSpeedTime 1800 && \
    git clone --no-checkout https://github.com/llvm/llvm-project.git && \
    cd llvm-project && \
    git checkout fad3272286528b8a491085183434c5ad4b59ab92 && \
    wget https://raw.githubusercontent.com/triton-lang/triton-ascend/6765b03c81c4e9ecb277e4ef1dde61dea0d044f0/third_party/ascend/llvm_patch/fad3272.patch && \
    git apply fad3272.patch && \
    mkdir build && \
    cd build && \
    export LLVM_INSTALL_PREFIX=/usr/local/llvm-install && \
    cmake ../llvm \
      -G Ninja \
      -DCMAKE_C_COMPILER=/usr/bin/clang-15 \
      -DCMAKE_CXX_COMPILER=/usr/bin/clang++-15 \
      -DCMAKE_LINKER=/usr/bin/lld-15 \
      -DCMAKE_BUILD_TYPE=Release \
      -DLLVM_ENABLE_ASSERTIONS=ON \
      -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
      -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
      -DLLVM_ENABLE_LLD=ON \
      -DCMAKE_INSTALL_PREFIX=${LLVM_INSTALL_PREFIX} && \
    ninja install && \
    cp bin/FileCheck ${LLVM_INSTALL_PREFIX}/bin/FileCheck

COPY requirements.txt requirements_dev.txt ./

RUN \
    python3 -m pip install --no-cache-dir -r requirements_dev.txt && \
    python3 -m pip install --no-cache-dir -r requirements.txt

RUN \
   mkdir -p /home/triton && \
   cd /home/triton && \
   git clone https://gitcode.com/Ascend/triton-ascend.git && \
   cd triton-ascend && \
   export LLVM_INSTALL_PREFIX=/usr/local/llvm-install && \
   LLVM_SYSPATH=${LLVM_INSTALL_PREFIX} \
   TRITON_BUILD_WITH_CCACHE=true \
   TRITON_BUILD_WITH_CLANG_LLD=true \
   TRITON_BUILD_PROTON=OFF \
   TRITON_WHEEL_NAME="triton-ascend" \
   TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
   python3 setup.py install

RUN \
    echo "export LLVM_INSTALL_PREFIX=/usr/local/llvm-install" >> /etc/profile && \
    echo "export LLVM_INSTALL_PREFIX=/usr/local/llvm-install" >> ~/.bashrc

WORKDIR /home/triton
