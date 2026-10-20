ARG CANN_BASE_IMAGE=swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/cann:8.5.0-a3-ubuntu22.04-py3.10
FROM ${CANN_BASE_IMAGE}

# torch-npu is pinned per CANN release: the CANN 8.5.0 line (a3/910b) uses
# 2.7.1.post8, while CANN 9.1.0 (a5/Ascend 950) uses 2.10.0.post4. requirements.txt
# is shared by every chip, so the version lives here instead — override at
# build time with --build-arg TORCH_NPU_VERSION=2.10.0.post4 for the a5 image.
ARG TORCH_NPU_VERSION=2.7.1.post8

ARG APTMIRROR=http://mirrors.huaweicloud.com
ARG PIP_INDEX_URL=https://repo.huaweicloud.com/repository/pypi/simple
ARG PIP_TRUSTED_HOST=repo.huaweicloud.com
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu

SHELL ["/bin/bash", "-c"]
ENV DEBIAN_FRONTEND=noninteractive
ENV TZ="Asia/shanghai"

RUN sed -i "s|http://archive.ubuntu.com|${APTMIRROR}|g" /etc/apt/sources.list 2>/dev/null || true && \
    sed -i "s|http://security.ubuntu.com|${APTMIRROR}|g" /etc/apt/sources.list 2>/dev/null || true && \
    echo 'Acquire::Retries "5";' > /etc/apt/apt.conf.d/80-retries && \
    echo 'Acquire::http::Timeout "60";' >> /etc/apt/apt.conf.d/80-retries && \
    for i in 1 2 3; do \
      apt update && apt install --yes --no-install-recommends --no-install-suggests \
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
      zlib1g-dev \
      && break \
      || { echo "apt attempt $i failed, retrying..."; sleep 10; [ $i -eq 3 ] && exit 1; }; \
    done && \
    apt clean && rm -rf /var/lib/apt/lists/* && \
    update-alternatives --install /usr/bin/clang clang /usr/bin/clang-15 100 && \
    update-alternatives --install /usr/bin/clang++ clang++ /usr/bin/clang++-15 100 && \
    update-alternatives --install /usr/bin/lld lld /usr/bin/lld-15 100 && \
    update-ca-certificates

RUN mkdir -p /root/.config/pip && \
    echo -e "[global]\nindex-url = ${PIP_INDEX_URL}\ntrusted-host = ${PIP_TRUSTED_HOST}" > /root/.config/pip/pip.conf

COPY requirements.txt requirements_dev.txt ./

# torch==2.7.1+cpu is a transitive dep of torch-npu but only available from
# the PyTorch CPU index, not the standard PyPI mirror.  Install it first so
# the subsequent pip install doesn't need to scan the extra index.
RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --timeout 600 \
      --find-links ${TORCH_INDEX_URL}/torch/ \
      torch==2.7.1+cpu

RUN --mount=type=cache,target=/root/.cache/pip \
    python3 -m pip install --timeout 300 -r requirements_dev.txt && \
    python3 -m pip install --timeout 300 -r requirements.txt && \
    python3 -m pip install torch-npu==${TORCH_NPU_VERSION}

WORKDIR /home/triton
