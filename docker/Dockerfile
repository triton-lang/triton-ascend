FROM ubuntu:22.04

ARG PY_VERSIONS="3.10 3.11"
ARG CHIP_TYPE=A3
ARG CANN_VERSION=8.5.0

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/shanghai"
ENV CHIP_TYPE=${CHIP_TYPE}
ENV CANN_VERSION=${CANN_VERSION}

RUN sed -i 's|http://archive.ubuntu.com|http://mirrors.huaweicloud.com|g' /etc/apt/sources.list && \
    sed -i 's|http://security.ubuntu.com|http://mirrors.huaweicloud.com|g' /etc/apt/sources.list && \
    echo 'Acquire::Retries "3";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "30";' >> /etc/apt/apt.conf.d/80-retries && \
    apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    ca-certificates \
    curl \
    gnupg && \
    curl -fsSL "https://keyserver.ubuntu.com/pks/lookup?op=get&search=0xBA6932366A755776" \
    | gpg --dearmor > /usr/share/keyrings/deadsnakes-ppa.gpg && \
    chmod 644 /usr/share/keyrings/deadsnakes-ppa.gpg && \
    echo "deb [signed-by=/usr/share/keyrings/deadsnakes-ppa.gpg] http://ppa.launchpad.net/deadsnakes/ppa/ubuntu jammy main" > /etc/apt/sources.list.d/deadsnakes-ppa.list && \
    apt update && \
    apt install --yes --no-install-recommends --no-install-suggests \
    bash \
    clang-15 \
    clang-format \
    cmake \
    ccache \
    git \
    lld-15 \
    make \
    ninja-build \
    python3-dev \
    python3-venv \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3.11-distutils \
    sudo \
    unzip \
    vim \
    wget \
    zlib1g-dev && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 100

RUN mkdir -p /root/.config/pip && \
    echo -e "[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple\nextra-index-url = https://download.pytorch.org/whl/cpu" > /root/.config/pip/pip.conf && \
    curl -sSLk https://bootstrap.pypa.io/get-pip.py -o /tmp/get-pip.py && \
    python3.10 /tmp/get-pip.py && \
    python3.11 /tmp/get-pip.py && \
    rm -f /tmp/get-pip.py && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.10 1 \
      --slave /usr/bin/python3 python3 /usr/bin/python3.10 \
      --slave /usr/bin/pip pip /usr/local/bin/pip3.10 \
      --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.10 && \
    update-alternatives --install /usr/bin/python python /usr/bin/python3.11 2 \
      --slave /usr/bin/python3 python3 /usr/bin/python3.11 \
      --slave /usr/bin/pip pip /usr/local/bin/pip3.11 \
      --slave /usr/bin/pip3 pip3 /usr/local/bin/pip3.11 && \
    rm -f /usr/local/bin/pip /usr/local/bin/pip3 && \
    update-alternatives --set python /usr/bin/python3.10

COPY requirements.txt requirements_dev.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    for VER in ${PY_VERSIONS}; do \
      PY="python${VER}"; \
      ${PY} -m pip install -r requirements_dev.txt && \
      ${PY} -m pip install -r requirements.txt; \
    done

RUN wget -q https://ascend.devcloud.huaweicloud.com/cann/debian/cann-keyring_1.0.0_all.deb && \
    dpkg -i cann-keyring_1.0.0_all.deb && \
    rm cann-keyring_1.0.0_all.deb && \
    apt update && \
    apt install --yes ascend-cann-toolkit=${CANN_VERSION} ascend-cann-${CHIP_TYPE}-ops=${CANN_VERSION} && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    echo "export LD_LIBRARY_PATH=/usr/local/Ascend/driver/lib64/common/:/usr/local/Ascend/driver/lib64/driver/:\$LD_LIBRARY_PATH" >> ~/.bashrc && \
    echo "source /usr/local/Ascend/cann/set_env.sh" >> ~/.bashrc

WORKDIR /home/triton