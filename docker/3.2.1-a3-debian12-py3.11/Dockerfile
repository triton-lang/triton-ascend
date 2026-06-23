# Stage 0: init source
ARG PIP_SOURCE="https://mirrors.ustc.edu.cn/pypi/simple"

# Stage 1: Install Python
FROM library/debian:12 AS python-installer

# Python Environment variables
ENV PATH=/usr/local/python3.11.15/bin:${PATH}

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        apt-transport-https \
        ca-certificates \
        bash \
        curl \
        build-essential \
        libssl-dev \
        zlib1g-dev \
        libncurses5-dev \
        libbz2-dev \
        libreadline-dev \
        libsqlite3-dev \
        libffi-dev \
        libnss3-dev \
        libgdbm-dev \
        liblzma-dev \
        libev-dev \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/tmp/* \
    && rm -rf /tmp/*

# Install Python
RUN curl -fsSL https://repo.huaweicloud.com/python/3.11.15/Python-3.11.15.tgz -o /tmp/Python-3.11.15.tgz && \
    tar -xf /tmp/Python-3.11.15.tgz -C /tmp && \
    cd /tmp/Python-3.11.15 && \
    mkdir -p /usr/local/python3.11.15/lib && \
    ./configure --enable-shared --enable-shared LDFLAGS="-Wl,-rpath /usr/local/python3.11.15/lib" --prefix=/usr/local/python3.11.15 && \
    make -j $(nproc) && \
    make altinstall && \
    ln -sf /usr/local/python3.11.15/bin/python3.11 /usr/local/python3.11.15/bin/python3 && \
    ln -sf /usr/local/python3.11.15/bin/pip3.11 /usr/local/python3.11.15/bin/pip3 && \
    ln -sf /usr/local/python3.11.15/bin/python3 /usr/local/python3.11.15/bin/python && \
    ln -sf /usr/local/python3.11.15/bin/pip3 /usr/local/python3.11.15/bin/pip && \
    rm -rf /tmp/*

# Stage 2: Install CANN
FROM python-installer AS cann-installer

ARG PIP_SOURCE
ARG TARGETPLATFORM

RUN apt-get update && apt-get install --no-install-recommends -y \
        git \
        wget \
        gcc \
        g++ \
        make \
        cmake \
        zlib1g \
        openssl \
        unzip \
        pciutils \
        net-tools \
        libblas-dev \
        gfortran \
        patchelf \
        libblas3 \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Note: Install CANN runtime dependencies
RUN pip install --no-cache-dir --upgrade pip --index-url ${PIP_SOURCE}

# Note: Get the download link according to ARCH and download the installation package
RUN ARCH=$(case "${TARGETPLATFORM}" in \
        "linux/amd64") echo "x86_64" ;; \
        "linux/arm64") echo "aarch64" ;; \
        *) echo "Unsupported TARGETPLATFORM: ${TARGETPLATFORM}" && exit 1 ;; \
    esac) && \
    CANN_TOOLKIT_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-toolkit_9.0.0_linux-${ARCH}.run && \
    CANN_OPS_URL=https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.0/Ascend-cann-A3-ops_9.0.0_linux-${ARCH}.run && \
    wget --quiet --header="Referer: https://www.hiascend.com/" ${CANN_TOOLKIT_URL} -O ~/Ascend-cann-toolkit.run && \
    wget --quiet --header="Referer: https://www.hiascend.com/" ${CANN_OPS_URL} -O ~/Ascend-cann-ops.run

# Note: Install CANN Toolkit Development Kit Package
RUN chmod +x ~/Ascend-cann-toolkit.run && \
    ~/Ascend-cann-toolkit.run --quiet --install --install-for-all && \
    rm -f ~/Ascend-cann-toolkit.run

# Note: Install CANN Ops Operator Package
RUN chmod +x ~/Ascend-cann-ops.run && \
    ~/Ascend-cann-ops.run --quiet --install --install-for-all && \
    rm -f ~/Ascend-cann-ops.run

# Stage 3: Install torch_npu and triton-ascend
# Note: Install triton runtime dependencies
RUN pip3 install --no-cache-dir \
    pyyaml \
    setuptools \
    pybind11 \
    ninja \
    cmake \
    wheel \
    tabulate \
    --index-url ${PIP_SOURCE}

# Note: Install torch_npu and triton-ascend
RUN case "${TARGETPLATFORM}" in \
        "linux/amd64")  \
            echo "install torch-npu for the x86 architecture"; \
            pip3 install --no-cache-dir torch==2.7.1+cpu --index-url https://download.pytorch.org/whl/cpu;; \
        "linux/arm64") \
            echo "install torch-npu for the arm architecture"; \
            pip3 install --no-cache-dir torch==2.7.1 --index-url ${PIP_SOURCE};; \
        *) echo "Unsupported TARGETPLATFORM: ${TARGETPLATFORM}" && exit 1 ;; \
    esac && \
    pip3 install --no-cache-dir torch-npu==2.7.1.post4 --index-url ${PIP_SOURCE} && \
    pip3 install --no-cache-dir triton-ascend==3.2.1 --extra-index-url=https://triton-ascend.osinfra.cn/pypi/simple

# Stage 4: install llvm
FROM python-installer AS llvm-installer
ARG PIP_SOURCE
RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        gcc \
        g++ \
        make \
        cmake \
        zlib1g-dev \
        libzstd-dev \
        clang-15  \
        ccache \
        lld-15 \
        git \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/tmp/* \
    && rm -rf /tmp/* \
    && ln -s /usr/bin/clang-15 /usr/bin/clang \
    && ln -s /usr/bin/clang++-15 /usr/bin/clang++

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
    git checkout b5cc222d7429fe6f18c787f633d5262fac2e676f && \
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
    ninja install

# Stage 5: Copy results from previous stages
FROM library/debian:12 AS official-debian

ARG PIP_SOURCE
ARG TARGETPLATFORM
# Python Environment variables
ENV PATH=/usr/local/python3.11.15/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/python3.11.15/lib:${LD_LIBRARY_PATH}

# Note: Toolkit Environment variables, obtained from /usr/local/Ascend/ascend-toolkit/set_env.sh
ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-9.0.0
ENV ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_LATEST_HOME}/lib64:${ASCEND_TOOLKIT_LATEST_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_LATEST_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_LATEST_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_LATEST_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_LATEST_HOME}/tools/aml/lib64/plugin:$LD_LIBRARY_PATH
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/devlib:$LD_LIBRARY_PATH
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
ENV PYTHONPATH=${ASCEND_TOOLKIT_LATEST_HOME}/python/site-packages:${ASCEND_TOOLKIT_LATEST_HOME}/opp/built-in/op_impl/ai_core/tbe:$PYTHONPATH
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/profiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ascend_system_advisor/asys:$PATH
ENV PATH=${ASCEND_TOOLKIT_HOME}/tools/show_kernel_debug_data:${ASCEND_TOOLKIT_HOME}/tools/msobjdump:$PATH
ENV PATH=${ASCEND_TOOLKIT_LATEST_HOME}/bin:${ASCEND_TOOLKIT_LATEST_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_LATEST_HOME}/tools/ccec_compiler/bin:$PATH
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
ENV CMAKE_PREFIX_PATH=${TOOLCHAIN_HOME}/tools/tikicpulib/lib/cmake:${ASCEND_TOOLKIT_HOME}/lib64/cmake

SHELL [ "/bin/bash", "-c" ]

RUN apt-get update \
    && DEBIAN_FRONTEND=noninteractive apt-get install --no-install-recommends -y \
        apt-transport-https \
        ca-certificates \
        bash \
        libc6 \
        libsqlite3-dev \
        gcc \
        g++ \
        make \
        cmake \
        git \
        vim \
        wget \
        jq \
        curl \
        build-essential \
        libnuma-dev \
        sudo \
        procps \
        sysstat \
        systemd \
        net-tools \
        iproute2 \
        openssl \
        grep \
        tree \
        rsync \
        tar \
        zip \
        unzip \
        findutils \
        python3-dev \
        zlib1g-dev \
        libzstd-dev \
        clang-15  \
        ccache \
        lld-15 \
        openssh-server \
        openssh-client \
        libssl-dev \
        dos2unix \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/* \
    && rm -rf /var/tmp/* \
    && rm -rf /tmp/* \
    && ln -s /usr/bin/clang-15 /usr/bin/clang \
    && ln -s /usr/bin/clang++-15 /usr/bin/clang++

COPY --from=cann-installer /usr/local/python3.11.15 /usr/local/python3.11.15
COPY --from=cann-installer /usr/local/Ascend /usr/local/Ascend
COPY --from=cann-installer /etc/Ascend /etc/Ascend
COPY --from=llvm-installer /usr/local/llvm-install /usr/local/llvm-install
COPY --chown=root:root docker/copyright.sh /etc/profile.d/copyright.sh

# Note: Set environment variables
RUN \
    CANN_TOOLKIT_ENV_FILE="/usr/local/Ascend/ascend-toolkit/set_env.sh" && \
    echo "source ${CANN_TOOLKIT_ENV_FILE}" >> /etc/profile && \
    echo "source ${CANN_TOOLKIT_ENV_FILE}" >> ~/.bashrc && \
    echo "alias ll='ls -l'" >> ~/.bashrc && \
    echo "bash /etc/profile.d/copyright.sh" >> ~/.bashrc && \
    echo "export LLVM_INSTALL_PREFIX=/usr/local/llvm-install" >> /etc/profile && \
    echo "export LLVM_INSTALL_PREFIX=/usr/local/llvm-install" >> ~/.bashrc && \
    chmod 555 /etc/profile.d/copyright.sh

ENTRYPOINT ["/bin/bash", "-c", "\
    source /usr/local/Ascend/ascend-toolkit/set_env.sh && \
    bash /etc/profile.d/copyright.sh && \
    exec \"$@\"", "--"]
