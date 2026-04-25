ARG CANN_BASE_IMAGE=quay.io/ascend/cann:8.5.0-a3-ubuntu22.04-py3.10
FROM ${CANN_BASE_IMAGE}

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
    update-ca-certificates

RUN mkdir -p /root/.config/pip && \
    echo -e "[global]\nindex-url = https://repo.huaweicloud.com/repository/pypi/simple\nextra-index-url = https://download.pytorch.org/whl/cpu" > /root/.config/pip/pip.conf

COPY requirements.txt requirements_dev.txt ./

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install -r requirements_dev.txt && \
    python3 -m pip install -r requirements.txt

WORKDIR /home/triton
