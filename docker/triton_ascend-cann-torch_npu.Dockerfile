ARG BASE_IMAGE=debian:12
ARG DEVICE_TYPE=910b

ARG CANN_VERSION=9.2.0
ARG PYTHON_VERSION=3.11.15
ARG PYTHON_SERIES=3.11
ARG PYTORCH_VERSION=2.10.0
ARG TORCH_NPU_VERSION=2.10.0.post7
ARG TRITON_ASCEND_VERSION=3.6.0
ARG LLVM_COMMIT=b5cc222d7429fe6f18c787f633d5262fac2e676f
ARG PIP_SOURCE=https://mirrors.ustc.edu.cn/pypi/simple


# Stage 1: Install Python
FROM ${BASE_IMAGE} AS python-installer

ARG BASE_IMAGE
ARG PYTHON_VERSION
ARG PYTHON_SERIES

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/python${PYTHON_VERSION}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/python${PYTHON_VERSION}/lib:${LD_LIBRARY_PATH}

RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
        if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
            sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources; \
        fi; \
        apt-get update; \
        apt-get install --no-install-recommends -y \
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
            libev-dev; \
        apt-get clean; \
        rm -rf /var/lib/apt/lists/* /var/tmp/*; \
    elif command -v yum >/dev/null 2>&1; then \
        yum -y update; \
        yum install -y \
            gcc \
            gcc-c++ \
            make \
            cmake \
            curl \
            zlib-devel \
            bzip2-devel \
            openssl-devel \
            ncurses-devel \
            sqlite-devel \
            readline-devel \
            tk-devel \
            gdbm-devel \
            libpcap-devel \
            xz-devel \
            libev-devel \
            expat-devel \
            libffi-devel \
            systemtap-sdt-devel; \
        yum clean all; \
        rm -rf /var/cache/yum /var/cache/dnf; \
    else \
        echo "Unsupported package manager in BASE_IMAGE=${BASE_IMAGE}" >&2; \
        exit 1; \
    fi; \
    rm -rf /tmp/*

RUN set -eux; \
    curl -fsSL \
        "https://repo.huaweicloud.com/python/${PYTHON_VERSION}/Python-${PYTHON_VERSION}.tgz" \
        -o /tmp/Python-${PYTHON_VERSION}.tgz; \
    tar -xf /tmp/Python-${PYTHON_VERSION}.tgz -C /tmp; \
    cd /tmp/Python-${PYTHON_VERSION}; \
    mkdir -p /usr/local/python${PYTHON_VERSION}/lib; \
    ./configure \
        --enable-shared \
        --prefix=/usr/local/python${PYTHON_VERSION} \
        LDFLAGS="-Wl,-rpath /usr/local/python${PYTHON_VERSION}/lib"; \
    make -j"$(nproc)"; \
    make altinstall; \
    ln -sf /usr/local/python${PYTHON_VERSION}/bin/python${PYTHON_SERIES} \
        /usr/local/python${PYTHON_VERSION}/bin/python3; \
    ln -sf /usr/local/python${PYTHON_VERSION}/bin/pip${PYTHON_SERIES} \
        /usr/local/python${PYTHON_VERSION}/bin/pip3; \
    ln -sf /usr/local/python${PYTHON_VERSION}/bin/python3 \
        /usr/local/python${PYTHON_VERSION}/bin/python; \
    ln -sf /usr/local/python${PYTHON_VERSION}/bin/pip3 \
        /usr/local/python${PYTHON_VERSION}/bin/pip; \
    rm -rf /tmp/*


# Stage 2: install CANN, PyTorch, torch_npu, and Triton-Ascend.
FROM python-installer AS cann-installer

ARG BASE_IMAGE
ARG CANN_VERSION
ARG DEVICE_TYPE
ARG PYTORCH_VERSION
ARG TORCH_NPU_VERSION
ARG TRITON_ASCEND_VERSION
ARG PIP_SOURCE
ARG TARGETARCH

RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
        apt-get update; \
        apt-get install --no-install-recommends -y \
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
            libblas3; \
        apt-get clean; \
        rm -rf /var/lib/apt/lists/* /var/tmp/*; \
    elif command -v yum >/dev/null 2>&1; then \
        unset LD_LIBRARY_PATH PYTHONHOME PYTHONPATH; \
        yum -y update; \
        yum install -y \
            unzip \
            pciutils \
            net-tools \
            lapack-devel \
            gcc-gfortran \
            util-linux \
            findutils \
            wget; \
        yum clean all; \
        rm -rf /var/cache/yum /var/cache/dnf; \
    else \
        echo "Unsupported package manager in BASE_IMAGE=${BASE_IMAGE}" >&2; \
        exit 1; \
    fi; \
    rm -rf /tmp/*

RUN pip install --no-cache-dir --upgrade pip --index-url "${PIP_SOURCE}"

RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) ARCH="x86_64" ;; \
        arm64) ARCH="aarch64" ;; \
        *) echo "Unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
    esac; \
    case "${DEVICE_TYPE}" in \
        910b) OPS_NAME="910b" ;; \
        950)  OPS_NAME="950" ;; \
        a3)   OPS_NAME="A3" ;; \
        *) echo "Unsupported DEVICE_TYPE=${DEVICE_TYPE}" >&2; exit 1 ;; \
    esac; \
    CANN_TOOLKIT_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}/Ascend-cann-toolkit_${CANN_VERSION}_linux-${ARCH}.run"; \
    CANN_OPS_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%20${CANN_VERSION}/Ascend-cann-${OPS_NAME}-ops_${CANN_VERSION}_linux-${ARCH}.run"; \
    wget --quiet --header="Referer: https://www.hiascend.com/" \
        "${CANN_TOOLKIT_URL}" -O /root/Ascend-cann-toolkit.run; \
    wget --quiet --header="Referer: https://www.hiascend.com/" \
        "${CANN_OPS_URL}" -O /root/Ascend-cann-ops.run; \
    chmod +x /root/Ascend-cann-toolkit.run /root/Ascend-cann-ops.run; \
    /root/Ascend-cann-toolkit.run --quiet --install --install-for-all; \
    /root/Ascend-cann-ops.run --quiet --install --install-for-all; \
    rm -f /root/Ascend-cann-toolkit.run /root/Ascend-cann-ops.run

RUN pip3 install --no-cache-dir \
        pyyaml \
        setuptools \
        pybind11 \
        ninja \
        cmake \
        wheel \
        tabulate \
        --index-url "${PIP_SOURCE}"

RUN set -eux; \
    case "${TARGETARCH}" in \
        amd64) \
            pip3 install --no-cache-dir "torch==${PYTORCH_VERSION}+cpu" \
                --index-url https://mirrors.cloud.tencent.com/pytorch-wheels/whl/cpu/ ;; \
        arm64) \
            pip3 install --no-cache-dir "torch==${PYTORCH_VERSION}" \
                --index-url "${PIP_SOURCE}" ;; \
        *) \
            echo "Unsupported TARGETARCH=${TARGETARCH}" >&2; \
            exit 1 ;; \
    esac; \
    pip3 install --no-cache-dir "torch-npu==${TORCH_NPU_VERSION}" \
        --index-url "${PIP_SOURCE}"; \
    pip3 install --no-cache-dir "triton-ascend==${TRITON_ASCEND_VERSION}" \
        --index-url "${PIP_SOURCE}" \
        --extra-index-url=https://mirrors.huaweicloud.com/ascend/repos/pypi


# Stage 3: build LLVM/MLIR with the same configuration as the original images.
FROM python-installer AS llvm-installer

ARG BASE_IMAGE
ARG LLVM_COMMIT
ARG PIP_SOURCE

RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
        apt-get update; \
        apt-get install --no-install-recommends -y \
            gcc \
            g++ \
            make \
            cmake \
            zlib1g-dev \
            libzstd-dev \
            clang-15 \
            ccache \
            lld-15 \
            git; \
        apt-get clean; \
        rm -rf /var/lib/apt/lists/* /var/tmp/*; \
        ln -sf /usr/bin/clang-15 /usr/bin/clang; \
        ln -sf /usr/bin/clang++-15 /usr/bin/clang++; \
    elif command -v yum >/dev/null 2>&1; then \
        unset LD_LIBRARY_PATH PYTHONHOME PYTHONPATH; \
        yum -y update; \
        yum install -y \
            gcc \
            g++ \
            make \
            cmake \
            zlib-devel \
            libzstd \
            libzstd-devel \
            ccache \
            clang \
            lld \
            git; \
        yum clean all; \
        rm -rf /var/cache/yum /var/cache/dnf; \
    else \
        echo "Unsupported package manager in BASE_IMAGE=${BASE_IMAGE}" >&2; \
        exit 1; \
    fi; \
    rm -rf /tmp/*

RUN pip3 install --no-cache-dir \
        ninja \
        cmake \
        wheel \
        pybind11 \
        --index-url "${PIP_SOURCE}"

RUN set -eux; \
    git config --global http.postBuffer 524288000; \
    git config --global http.lowSpeedLimit 0; \
    git config --global http.lowSpeedTime 1800; \
    git clone --no-checkout --filter=blob:none https://github.com/llvm/llvm-project.git; \
    cd llvm-project; \
    git checkout "${LLVM_COMMIT}"; \
    mkdir build; \
    cd build; \
    C_COMPILER="$(command -v clang-15 || command -v clang)"; \
    CXX_COMPILER="$(command -v clang++-15 || command -v clang++)"; \
    LLD_LINKER="$(command -v lld-15 || command -v lld || command -v ld.lld)"; \
    cmake ../llvm \
        -G Ninja \
        -DCMAKE_C_COMPILER="${C_COMPILER}" \
        -DCMAKE_CXX_COMPILER="${CXX_COMPILER}" \
        -DCMAKE_LINKER="${LLD_LINKER}" \
        -DCMAKE_BUILD_TYPE=Release \
        -DLLVM_ENABLE_ASSERTIONS=ON \
        -DLLVM_ENABLE_PROJECTS="mlir;llvm;lld" \
        -DLLVM_TARGETS_TO_BUILD="host;NVPTX;AMDGPU" \
        -DLLVM_ENABLE_LLD=ON \
        -DCMAKE_INSTALL_PREFIX=/usr/local/llvm-install; \
    ninja install


# Stage 4: runtime image.
FROM ${BASE_IMAGE} AS final

ARG BASE_IMAGE
ARG CANN_VERSION
ARG DEVICE_TYPE
ARG PYTHON_VERSION
ARG PYTHON_SERIES
ARG TRITON_ASCEND_VERSION
ARG PYTORCH_VERSION
ARG TORCH_NPU_VERSION
ARG TARGETARCH

SHELL ["/bin/bash", "-c"]

ENV DEBIAN_FRONTEND=noninteractive
ENV PATH=/usr/local/python${PYTHON_VERSION}/bin:${PATH}
ENV LD_LIBRARY_PATH=/usr/local/python${PYTHON_VERSION}/lib:${LD_LIBRARY_PATH}

ENV ASCEND_TOOLKIT_HOME=/usr/local/Ascend/cann-${CANN_VERSION}
ENV ASCEND_TOOLKIT_LATEST_HOME=/usr/local/Ascend/ascend-toolkit/latest
ENV LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64:/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/lib64:${ASCEND_TOOLKIT_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_HOME}/tools/aml/lib64/plugin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_LATEST_HOME}/lib64:${ASCEND_TOOLKIT_LATEST_HOME}/lib64/plugin/opskernel:${ASCEND_TOOLKIT_LATEST_HOME}/lib64/plugin/nnengine:${ASCEND_TOOLKIT_LATEST_HOME}/opp/built-in/op_impl/ai_core/tbe/op_tiling:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_LATEST_HOME}/tools/aml/lib64:${ASCEND_TOOLKIT_LATEST_HOME}/tools/aml/lib64/plugin:${LD_LIBRARY_PATH}
ENV LD_LIBRARY_PATH=${ASCEND_TOOLKIT_HOME}/devlib:${LD_LIBRARY_PATH}
ENV PYTHONPATH=${ASCEND_TOOLKIT_HOME}/python/site-packages:${ASCEND_TOOLKIT_HOME}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}
ENV PYTHONPATH=${ASCEND_TOOLKIT_LATEST_HOME}/python/site-packages:${ASCEND_TOOLKIT_LATEST_HOME}/opp/built-in/op_impl/ai_core/tbe:${PYTHONPATH}
ENV PATH=${ASCEND_TOOLKIT_HOME}/bin:${ASCEND_TOOLKIT_HOME}/tools/ccec_compiler/bin:${ASCEND_TOOLKIT_HOME}/tools/profiler/bin:${ASCEND_TOOLKIT_HOME}/tools/ascend_system_advisor/asys:${PATH}
ENV PATH=${ASCEND_TOOLKIT_HOME}/tools/show_kernel_debug_data:${ASCEND_TOOLKIT_HOME}/tools/msobjdump:${PATH}
ENV PATH=${ASCEND_TOOLKIT_LATEST_HOME}/bin:${ASCEND_TOOLKIT_LATEST_HOME}/compiler/ccec_compiler/bin:${ASCEND_TOOLKIT_LATEST_HOME}/tools/ccec_compiler/bin:${PATH}
ENV ASCEND_AICPU_PATH=${ASCEND_TOOLKIT_HOME}
ENV ASCEND_OPP_PATH=${ASCEND_TOOLKIT_HOME}/opp
ENV TOOLCHAIN_HOME=${ASCEND_TOOLKIT_HOME}/toolkit
ENV ASCEND_HOME_PATH=${ASCEND_TOOLKIT_HOME}
ENV CMAKE_PREFIX_PATH=${TOOLCHAIN_HOME}/tools/tikicpulib/lib/cmake:${ASCEND_TOOLKIT_HOME}/lib64/cmake
ENV LLVM_INSTALL_PREFIX=/usr/local/llvm-install

LABEL org.opencontainers.image.title="Triton-Ascend"
LABEL org.opencontainers.image.description="Triton-Ascend ${TRITON_ASCEND_VERSION} with CANN ${CANN_VERSION} for ${DEVICE_TYPE}"
LABEL triton-ascend.version="${TRITON_ASCEND_VERSION}"
LABEL cann.version="${CANN_VERSION}"
LABEL torch.version="${PYTORCH_VERSION}"
LABEL torch-npu.version="${TORCH_NPU_VERSION}"

RUN set -eux; \
    if command -v apt-get >/dev/null 2>&1; then \
        if [ -f /etc/apt/sources.list.d/debian.sources ]; then \
            sed -i 's|deb.debian.org|mirrors.ustc.edu.cn|g' /etc/apt/sources.list.d/debian.sources; \
        fi; \
        apt-get update; \
        apt-get install --no-install-recommends -y \
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
            clang-15 \
            ccache \
            lld-15 \
            openssh-server \
            openssh-client \
            libssl-dev \
            dos2unix; \
        apt-get clean; \
        rm -rf /var/lib/apt/lists/* /var/tmp/*; \
        ln -sf /usr/bin/clang-15 /usr/bin/clang; \
        ln -sf /usr/bin/clang++-15 /usr/bin/clang++; \
    elif command -v yum >/dev/null 2>&1; then \
        yum -y update; \
        yum install -y \
            ca-certificates \
            bash \
            glibc \
            sqlite-devel \
            gcc \
            gcc-c++ \
            make \
            cmake \
            git \
            vim \
            wget \
            jq \
            curl \
            numactl-devel \
            sudo \
            procps-ng \
            sysstat \
            systemd \
            net-tools \
            iproute \
            openssl \
            grep \
            tree \
            rsync \
            tar \
            zip \
            unzip \
            findutils \
            python3-devel \
            zlib-devel \
            libzstd \
            libzstd-devel \
            ccache \
            clang \
            lld \
            openssl-devel \
            dos2unix; \
        yum clean all; \
        rm -rf /var/cache/yum /var/cache/dnf; \
    else \
        echo "Unsupported package manager in BASE_IMAGE=${BASE_IMAGE}" >&2; \
        exit 1; \
    fi; \
    rm -rf /tmp/*

COPY --from=cann-installer /usr/local/python${PYTHON_VERSION} /usr/local/python${PYTHON_VERSION}
COPY --from=cann-installer /usr/local/Ascend /usr/local/Ascend
COPY --from=cann-installer /etc/Ascend /etc/Ascend
COPY --from=llvm-installer /usr/local/llvm-install /usr/local/llvm-install
COPY --chown=root:root --chmod=555 docker/copyright.sh /etc/profile.d/copyright.sh

RUN set -eux; \
    if command -v yum >/dev/null 2>&1; then \
        case "${TARGETARCH}" in \
            amd64) ARCH="x86_64" ;; \
            arm64) ARCH="aarch64" ;; \
            *) echo "Unsupported TARGETARCH=${TARGETARCH}" >&2; exit 1 ;; \
        esac; \
        if [ -d "/usr/include/c++/12" ]; then \
            ln -sf "/usr/include/c++/12/${ARCH}-openEuler-linux" \
                "/usr/include/c++/12/${ARCH}-linux-gnu"; \
        fi; \
    fi; \
    CANN_TOOLKIT_ENV_FILE="/usr/local/Ascend/ascend-toolkit/set_env.sh"; \
    printf '%s\n' \
        "source ${CANN_TOOLKIT_ENV_FILE}" \
        "alias ll='ls -l'" \
        "bash /etc/profile.d/copyright.sh" \
        "export LLVM_INSTALL_PREFIX=/usr/local/llvm-install" \
        >> /etc/profile; \
    printf '%s\n' \
        "source ${CANN_TOOLKIT_ENV_FILE}" \
        "alias ll='ls -l'" \
        "bash /etc/profile.d/copyright.sh" \
        "export LLVM_INSTALL_PREFIX=/usr/local/llvm-install" \
        >> /root/.bashrc; \
    chmod 555 /etc/profile.d/copyright.sh

WORKDIR /workspace

ENTRYPOINT ["/bin/bash", "-c", "source /usr/local/Ascend/ascend-toolkit/set_env.sh && bash /etc/profile.d/copyright.sh && exec \"$@\"", "--"]
CMD ["/bin/bash"]
