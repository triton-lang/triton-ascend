FROM ubuntu:22.04 AS base

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/shanghai"

# Configure apt: use mirror, retry on failures, reduce timeouts
RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.huaweicloud.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.huaweicloud.com|g' /etc/apt/sources.list && \
    echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries

RUN apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    bash \
    ca-certificates \
    curl \
    git \
    gnupg \
    make \
    sudo \
    unzip \
    vim \
    wget && \
    apt clean && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates

RUN curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xBA6932366A755776" \
    | gpg --dearmor > /usr/share/keyrings/deadsnakes-ppa.gpg && \
    chmod 644 /usr/share/keyrings/deadsnakes-ppa.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/deadsnakes-ppa.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list && \
    apt-get update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    clang-15 \
    clang-format \
    cmake \
    lld-15 \
    ninja-build \
    ccache \
    python3-dev \
    python3-venv \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    python3.9 \
    python3.9-dev \
    python3.9-venv \
    python3.9-distutils \
    zlib1g-dev && \
    apt clean && \
    rm -rf /var/lib/apt/lists/*

RUN curl -sSLk https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.9 /tmp/get-pip.py && \
    python3.10 /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.9 1 \
      --slave /usr/bin/python3 python3 /usr/bin/python3.9 \
      --slave /usr/bin/pip pip /usr/local/bin/pip3.9 \
      --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.9 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 2 \
      --slave /usr/bin/python3 python3 /usr/bin/python3.10 \
      --slave /usr/bin/pip pip /usr/local/bin/pip3.10 \
      --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 3 \
      --slave /usr/bin/python3 python3 /usr/bin/python3.11 \
      --slave /usr/bin/pip pip /usr/local/bin/pip3.11 \
      --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.11 && \
    rm -f /usr/local/bin/pip /usr/local/bin/pip3 && \
    update-alternatives --set python /usr/bin/python3.10

# ── build stage: build tools only, no CANN ──────────────────────────────────
FROM base AS build

COPY requirements.txt requirements_dev.txt build/

RUN for PY in python3.9 python3.10 python3.11; do \
      ${PY} -m pip install -r build/requirements_dev.txt -qq && \
      ${PY} -m pip install cmake ninja wheel pybind11 -qq && \
      rm -rf ~/.cache/pip; \
    done && \
    rm -rf build/

WORKDIR /home/triton

# ── release stage: full runtime with CANN ───────────────────────────────────
FROM base AS release

ARG NPU_TYPE
ENV NPU_TYPE=${NPU_TYPE}

COPY requirements.txt requirements_dev.txt build/

RUN for PY in python3.9 python3.10 python3.11; do \
    ${PY} -m pip install torch==2.7.1 --index-url https://download.pytorch.org/whl/cpu -qq && \
    ${PY} -m pip install -r build/requirements_dev.txt && \
    ${PY} -m pip install -r build/requirements.txt -qq && \
    rm -rf ~/.cache/pip; \
  done

RUN ARCH=$(uname -i) && \
  TOOLKIT_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T2/Ascend-cann-toolkit_9.0.0-beta.1_linux-${ARCH}.run" && \
  KERNEL_URL="https://ascend-repo.obs.cn-east-2.myhuaweicloud.com/CANN/CANN%209.0.T2/Ascend-cann-${NPU_TYPE}-ops_9.0.0-beta.1_linux-${ARCH}.run" && \
  curl -sSL "${TOOLKIT_URL}" -o build/Ascend-cann-toolkit.run && \
  curl -sSL "${KERNEL_URL}" -o build/Ascend-cann-kernels.run && \
  chmod +x build/Ascend-cann-toolkit.run build/Ascend-cann-kernels.run && \
  build/Ascend-cann-toolkit.run --full --quiet && \
  build/Ascend-cann-kernels.run --install --quiet && \
  rm -rf build/

RUN echo "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:$LD_LIBRARY_PATH" >> ~/.bashrc&& \
    echo "source /usr/local/Ascend/cann/set_env.sh" >> ~/.bashrc