# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Keep CPU model inputs in the candidate's ordinary compilation cache."""
from pathlib import Path

VERSION = 1


def _cache(key):
    from triton.runtime.cache import get_cache_manager
    return get_cache_manager(key)


def collect(temps, compiler, metadata, *, cache_factory=None):
    """Save evidence before compiler temporaries disappear; never launch code.

    Missing intermediate output disables selection without invalidating a
    successfully compiled baseline. Cache identity belongs to the normal JIT.
    """
    evidence = dict(version=VERSION, status="UNAVAILABLE", files={}, compiler=str(compiler))
    try:
        directory = Path(temps)
        sources = {"final_ir": directory / "module.hivm.opt.mlir"}
        for role in ("aic", "aiv"):
            path = directory / ("kernel_mix_" + role + ".ll")
            if path.is_file():
                sources["llvm_" + role] = path
        single = directory / "kernel.ll"
        if single.is_file():
            sources["llvm"] = single
        if not any(key.startswith("llvm") for key in sources):
            raise ValueError("compiler did not preserve LLVM modules")
        contents = {role: path.read_bytes() for role, path in sources.items()}
        key = metadata["hash"]
        cache = (cache_factory or _cache)(key)
        for role, data in contents.items():
            name = "block_auto_" + role + (".mlir" if role == "final_ir" else ".ll")
            cache.put(data, name, binary=True)
            evidence["files"][role] = dict(name=name)
        evidence.update(status="COLLECTED", cache_key=key)
    except (KeyError, OSError, ValueError) as error:
        evidence["reason"] = str(error)
    return evidence


def load_evidence(kernel, *, cache_factory=None):
    """Load the candidate's cached model inputs and runtime metadata."""
    metadata = kernel.metadata._asdict()
    if not isinstance(metadata.get("target"), dict) and hasattr(metadata.get("target"), "__dict__"):
        metadata["target"] = vars(metadata["target"])
    evidence = metadata.get("block_auto_evidence", {})
    if evidence.get("version") != VERSION or evidence.get("status") != "COLLECTED":
        raise ValueError("same-compilation model evidence unavailable")
    cache = (cache_factory or _cache)(evidence["cache_key"])
    result = dict(metadata=metadata, files={})
    for role, record in evidence["files"].items():
        name = record["name"]
        path = cache.get_file(name)
        if not path:
            raise ValueError("cached model input unavailable: " + role)
        result[role] = Path(path).read_text()
        result["files"][role] = dict(path=str(path))
    result["files"]["compiler"] = dict(path=evidence["compiler"])
    return result
