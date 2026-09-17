# Build triton-ascend wheels inside a manylinux container.
# Used by wheels.yml via docker/build-push-action@v7 — no Docker daemon needed
# on the runner; buildx talks to remote buildkitd.

ARG MANYLINUX_IMAGE=swr.cn-southwest-2.myhuaweicloud.com/modelfoundry/pypa/manylinux_2_28_arrch64:latest
FROM ${MANYLINUX_IMAGE} AS builder

# ---------------------------------------------------------------------------
# Build dependencies (matching CIBW_BEFORE_ALL)
# ---------------------------------------------------------------------------
RUN dnf install -y clang lld ccache cmake

# ---------------------------------------------------------------------------
# Build args — set by the workflow matrix
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=cp310
ARG MAX_JOBS=4
ARG TRITON_WHEEL_VERSION_SUFFIX=+dev
ARG BUILD_DATE=00000000

# ---------------------------------------------------------------------------
# Copy the full workspace.  Different branches keep setup.py in different
# places (root vs python/), so the build step detects it below.
# ---------------------------------------------------------------------------
COPY . /project/
WORKDIR /project

# ---------------------------------------------------------------------------
# Environment matching the original CIBW_ENVIRONMENT
# ---------------------------------------------------------------------------
ENV MAX_JOBS=${MAX_JOBS} \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_PROTON=OFF \
    TRITON_WHEEL_NAME=triton-ascend \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    TRITON_WHEEL_VERSION_SUFFIX=${TRITON_WHEEL_VERSION_SUFFIX}${BUILD_DATE} \
    IS_MANYLINUX=TRUE

# ---------------------------------------------------------------------------
# Install setuptools + wheel for the target Python (not pre-installed in
# minimal manylinux images).
# ---------------------------------------------------------------------------
RUN export PIP_INDEX_URL=http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple \
    && export PIP_TRUSTED_HOST=cache-service.nginx-pypi-cache.svc.cluster.local \
    && export PIP_TIMEOUT=120 \
    && /opt/python/${PYTHON_VERSION}-${PYTHON_VERSION}/bin/python3 -m ensurepip \
    && /opt/python/${PYTHON_VERSION}-${PYTHON_VERSION}/bin/python3 -m pip install --upgrade pip \
    && /opt/python/${PYTHON_VERSION}-${PYTHON_VERSION}/bin/python3 -m pip install setuptools wheel cmake ninja pybind11

# ---------------------------------------------------------------------------
# Build the wheel with the target Python from the manylinux toolchain.
# Put the pip-installed ninja/cmake ahead of the system ones on PATH.
# ---------------------------------------------------------------------------
RUN export PATH="/opt/python/${PYTHON_VERSION}-${PYTHON_VERSION}/bin:${PATH}" \
    && if [ -f setup_ascend.py ]; then \
         SETUP_DIR="./"; SETUP_PY="setup_ascend.py"; \
       elif [ -f setup.py ]; then \
         SETUP_DIR="./"; SETUP_PY="setup.py"; \
       elif [ -f python/setup.py ]; then \
         SETUP_DIR="python"; SETUP_PY="setup.py"; \
       else \
         echo "ERROR: setup.py or setup_ascend.py not found" >&2; \
         exit 1; \
       fi \
    && cd "$SETUP_DIR" \
    && python3 ${SETUP_PY} bdist_wheel \
    && mkdir -p /out \
    && cp dist/*.whl /out/

# ---------------------------------------------------------------------------
# Output stage — only the .whl files, extracted via --output type=local.
# ---------------------------------------------------------------------------
FROM scratch AS output
COPY --from=builder /out/*.whl /
