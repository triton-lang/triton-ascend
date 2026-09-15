# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
#
# Ascend operator dtype support configuration and unified validation helpers.
#
# This directory is intentionally a standalone folder (it is not the backend
# package ``triton.backends.ascend.utils``). The backend loads it by file path
# via importlib, see ``third_party/ascend/backend/__init__.py``.

from .op_dtype_config import OP_DTYPE_RULES, FP8_DTYPES, FP8E4_DTYPES

__all__ = ["OP_DTYPE_RULES", "FP8_DTYPES", "FP8E4_DTYPES", "install_dtype_guard", "UnsupportedDtypeError"]


def __getattr__(name):
    # Lazy import: the guard imports triton only when ``install`` runs, so
    # merely importing this config package stays dependency-free.
    if name in ("install_dtype_guard", "UnsupportedDtypeError"):
        from .dtype_guard import install, UnsupportedDtypeError

        if name == "install_dtype_guard":
            return install
        return UnsupportedDtypeError
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
