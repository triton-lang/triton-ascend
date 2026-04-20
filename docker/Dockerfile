FROM ubuntu:22.04

SHELL ["/bin/bash", "-c"]
ARG CHIP_TYPE=A3
ARG CANN_VERSION=8.5.0
RUN if [ "$CHIP_TYPE" = "A2" ]; then \
    CHIP_TYPE="910b";
ENV DEBIAN_FRONTEND=noninteractive \
    TZ="Asia/shanghai" \
    CHIP_TYPE=${CHIP_TYPE} \
    CANN_VERSION=${CANN_VERSION}

RUN \
    echo "===== 镜像构建参数 =====" && \
    echo "CHIP_TYPE: ${CHIP_TYPE}" && \
    echo "CANN_VERSION: ${CANN_VERSION}" && \
    echo "========================"

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
    | gpg --dearmor > /usr/share/keyrings/deadsnakes-ppa.gpg  && \
    chmod 644 /usr/share/keyrings/deadsnakes-ppa.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/deadsnakes-ppa.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list && \
    apt-get update || true

RUN apt install --yes --no-install-recommends --no-install-suggests \
    clang-15 \
    clang-format \
    cmake \
    lld-15 \
    ninja-build \
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

RUN update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 100

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

# Install triton requirments
COPY Makefile requirements.txt requirements_dev.txt  build/
COPY cmake/llvm-hash.txt build/cmake/

RUN cd build && \
    make install-deps && \
    make install-cann && \
    for PY in python3.9 python3.10 python3.11; do \
      make install-dev-reqs PYTHON=${PY} && \
      make install-rt-reqs PYTHON=${PY} && \
      rm -f .req_rt_installed .req_dev_installed && \
      rm -rf ~/.cache/pip; \
    done && \
    cd .. && \
    rm -rf build

# Install triton for 3.9/3.10/3.11 python dev
COPY . triton-ascend/
RUN cd triton-ascend && \
    cd python && \
    for PY in python3.9 python3.10 python3.11; do \
      ${PY} setup.py clean --all && \
      ${PY} -m pip install . ; \
    done && \
    cd ..

RUN echo "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "source /usr/local/Ascend/cann/set_env.sh" >> ~/.bashrc
