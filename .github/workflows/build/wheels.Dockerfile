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
ARG MAX_JOBS=4
ARG TRITON_WHEEL_VERSION_SUFFIX=+dev
ARG BUILD_DATE=00000000

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
ENV MAX_JOBS=${MAX_JOBS} \
    TRITON_BUILD_WITH_CLANG_LLD=true \
    TRITON_BUILD_PROTON=OFF \
    TRITON_WHEEL_NAME=triton-ascend \
    TRITON_APPEND_CMAKE_ARGS="-DTRITON_BUILD_UT=OFF" \
    TRITON_WHEEL_VERSION_SUFFIX=${TRITON_WHEEL_VERSION_SUFFIX}${BUILD_DATE} \
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
    && if [ -d /project/.npuir-payload/bin ] && [ -d /project/.npuir-payload/lib ]; then \
         export TRITON_ASCEND_BISHENGIR_PATH=/project/.npuir-payload; \
         echo "Bundling BishengIR payload from ${TRITON_ASCEND_BISHENGIR_PATH}"; \
       else \
         echo "No .npuir-payload in the build context; building without a bundled BishengIR"; \
       fi \
    && echo "=== auditwheel pre-flight: version and known policies ===" \
    && (auditwheel --version 2>/dev/null || echo "WARN: auditwheel not on PATH") \
    && (AW_BIN=$(readlink -f "$(command -v auditwheel 2>/dev/null)"); \
        AW_PY=$(sed -n '1s/^#!//p' "$AW_BIN" 2>/dev/null); \
        FOUND=0; \
        for PY in "$AW_PY" /opt/_internal/pipx/venvs/auditwheel/bin/python3.14 /opt/_internal/pipx/venvs/auditwheel/bin/python3; do \
          [ -x "$PY" ] && "$PY" -c 'import auditwheel.policy as p; pols = getattr(p, "_POLICIES", None) or getattr(p, "policies", []); names = sorted(x["name"] for x in pols); print("known policies:", names); import os; target = "manylinux_2_34_" + os.uname().machine; print("target policy:", target, "->", "OK" if target in names else "MISSING")' && { FOUND=1; break; }; \
        done; \
        [ "$FOUND" = 1 ] || echo "WARN: could not introspect auditwheel policies") \
    && python3 ${SETUP_PY} bdist_wheel \
    && mkdir -p /out \
    && cp dist/*.whl /out/

# ---------------------------------------------------------------------------
# Output stage — only the .whl files, extracted via --output type=local.
# ---------------------------------------------------------------------------
FROM scratch AS output
COPY --from=builder /out/*.whl /
