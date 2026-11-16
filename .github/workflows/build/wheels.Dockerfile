# Build triton-ascend wheels inside a manylinux container.
# Used by wheels.yml via docker/build-push-action@v7 — no Docker daemon needed
# on the runner; buildx talks to remote buildkitd.
#
# Layer order matters for the registry cache used by wheels.yml: everything
# that does not depend on the source tree (dnf, pip) must come BEFORE the
# COPY of the workspace, so those layers hit the cache across runs.

ARG MANYLINUX_IMAGE=swr.cn-southwest-2.myhuaweicloud.com/base_image/ascend-ci/manylinux:9.2.0-beta.2-a3-manylinux_2_34-py3.12
FROM ${MANYLINUX_IMAGE} AS builder

# ---------------------------------------------------------------------------
# Build args — set by the workflow matrix
# ---------------------------------------------------------------------------
ARG PYTHON_VERSION=cp310
ARG TRITON_WHEEL_VERSION_SUFFIX=+dev
ARG BUILD_DATE=00000000
ARG TRITON_BUILD_NPUIR=OFF
ARG ASCEND_HOME_PATH=/usr/local/Ascend/cann-9.2.0-beta.2

# ---------------------------------------------------------------------------
# Build dependencies (matching CIBW_BEFORE_ALL)
# ---------------------------------------------------------------------------
RUN dnf install -y clang lld ccache cmake

# ---------------------------------------------------------------------------
# Install setuptools + wheel for the target Python (not pre-installed in
# minimal manylinux images). Source-independent: cacheable layer.
# ---------------------------------------------------------------------------
RUN export PIP_INDEX_URL=http://cache-service.nginx-pypi-cache.svc.cluster.local/pypi/simple \
    && export PIP_TRUSTED_HOST=cache-service.nginx-pypi-cache.svc.cluster.local \
    && export PIP_TIMEOUT=120 \
    && /opt/python/${PYTHON_VERSION}-${PYTHON_VERSION}/bin/python3 -m ensurepip \
    && /opt/python/${PYTHON_VERSION}-${PYTHON_VERSION}/bin/python3 -m pip install --upgrade pip \
    && /opt/python/${PYTHON_VERSION}-${PYTHON_VERSION}/bin/python3 -m pip install setuptools wheel cmake ninja pybind11

# ---------------------------------------------------------------------------
# Environment matching the original CIBW_ENVIRONMENT
# ---------------------------------------------------------------------------
ENV TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_PROTON=OFF \
    TRITON_WHEEL_NAME=triton-ascend \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    TRITON_WHEEL_VERSION_SUFFIX=${TRITON_WHEEL_VERSION_SUFFIX}${BUILD_DATE} \
    TRITON_BUILD_NPUIR=${TRITON_BUILD_NPUIR} \
    ASCEND_HOME_PATH=${ASCEND_HOME_PATH} \
    IS_MANYLINUX=TRUE

# ---------------------------------------------------------------------------
# Copy the full workspace.  Different branches keep setup.py in different
# places (root vs python/), so the build step detects it below.
# ---------------------------------------------------------------------------
COPY . /project/
WORKDIR /project

# ---------------------------------------------------------------------------
# Build the wheel with the target Python from the manylinux toolchain.
# Put the pip-installed ninja/cmake ahead of the system ones on PATH.
# ---------------------------------------------------------------------------
# Parallelism is computed inside this RUN (the compile environment): the
# runner job pod's cgroup is tiny and unrelated to the shared buildkitd
# that actually runs the compile. Budget ~2.5 GiB per job and cap at 24:
# the build container's own cgroup reads 32 CPUs / 122 GiB, so 24 jobs fit
# the memory limit (~60 GiB at budget, ~96 GiB worst-case) and still leave
# headroom if the 32-core quota is shared with other matrix jobs. If the
# nightly numbers show the quota is shared (no speedup), drop back to 16.
# ccache (used by both the npuir build and the triton build) lives on a
# buildkit cache mount below and is exported via cache-to mode=max, so
# nightly rebuilds only recompile changed TUs.
RUN --mount=type=cache,target=/root/.cache/ccache \
    . /project/.github/workflows/build/cgroup-resources.sh \
    || { echo "FAILED to source cgroup-resources.sh; build dir contents:"; ls -la /project/.github/workflows/build/ 2>/dev/null || true; exit 1; } \
    && export MAX_JOBS=$(cg_cpus) \
    && MEM_JOBS=$(( $(cg_mem_bytes) / (2500 * 1024 * 1024) )) \
    && if [ "$MAX_JOBS" -gt "$MEM_JOBS" ]; then MAX_JOBS=$MEM_JOBS; fi \
    && if [ "$MAX_JOBS" -gt 32 ]; then MAX_JOBS=32; fi \
    && if [ "$MAX_JOBS" -lt 1 ]; then MAX_JOBS=1; fi \
    && export CCACHE_DIR=/root/.cache/ccache \
    && export CCACHE_MAXSIZE=20G \
    && echo "cgroup: $(cg_cpus) CPUs, mem $(( $(cg_mem_bytes) / (1024*1024*1024) )) GiB -> MAX_JOBS=$MAX_JOBS, ccache at ${CCACHE_DIR} (max ${CCACHE_MAXSIZE})" \
    && export PATH="/opt/python/${PYTHON_VERSION}-${PYTHON_VERSION}/bin:${PATH}" \
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
    && if [ -d /project/.npuir-payload/bin ] && [ -d /project/.npuir-payload/lib ]; then \
         export TRITON_ASCEND_BISHENGIR_PATH=/project/.npuir-payload; \
         echo "Bundling BishengIR payload from ${TRITON_ASCEND_BISHENGIR_PATH}"; \
       else \
         echo "No .npuir-payload in the build context; building without a bundled BishengIR"; \
       fi \
    && python3 ${SETUP_PY} bdist_wheel \
    && ccache -s \
    && mkdir -p /out \
    && cp dist/*.whl /out/

# ---------------------------------------------------------------------------
# ccache statistics, printed in a tiny layer so the build log's 2 MiB cap
# does not clip them (the compile RUN's warnings flood the log).
# ---------------------------------------------------------------------------
RUN --mount=type=cache,target=/root/.cache/ccache     CCACHE_DIR=/root/.cache/ccache ccache -s

# ---------------------------------------------------------------------------
# Output stage — only the .whl files, extracted via --output type=local.
# ---------------------------------------------------------------------------
FROM scratch AS output
COPY --from=builder /out/*.whl /
