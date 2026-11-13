# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Persist and verify CPU model inputs from the candidate's own compilation."""
import hashlib
import json
from pathlib import Path

VERSION = 1


def digest(data):
    return hashlib.sha256(data).hexdigest()


def file_digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def _cache(key):
    from triton.runtime.cache import get_cache_manager
    return get_cache_manager(key)


def collect(temps, binary, compiler, compiler_sha256, metadata, command, *, cache_factory=None):
    """Save evidence before compiler temporaries disappear; never launch code.

    Missing intermediate output disables selection without invalidating a
    successfully compiled baseline. Every file is addressed by its contents.
    """
    evidence = dict(version=VERSION, status="UNAVAILABLE", files={}, binary_sha256=digest(binary),
                    compiler_sha256=compiler_sha256, ttir_sha256=metadata.get("block_auto_ttir_sha256"),
                    ttadapter_sha256=metadata.get("block_auto_ttadapter_sha256"), target=metadata.get("target_arch"),
                    command=list(command))
    try:
        if file_digest(compiler) != compiler_sha256:
            raise ValueError("compiler changed during candidate compilation")
        if not binary.startswith(b"\x7fELF"):
            raise ValueError("candidate binary is not ELF")
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
        identity = dict(version=VERSION, binary=evidence["binary_sha256"], compiler=compiler_sha256,
                        files={role: digest(data)
                               for role, data in contents.items()})
        key = digest(json.dumps(identity, sort_keys=True).encode())
        cache = (cache_factory or _cache)(key)
        for role, data in contents.items():
            name = "block_auto_" + role + (".mlir" if role == "final_ir" else ".ll")
            cache.put(data, name, binary=True)
            evidence["files"][role] = dict(name=name, sha256=digest(data))
        evidence.update(status="COLLECTED", cache_key=key)
    except (OSError, ValueError) as error:
        evidence["reason"] = str(error)
    return evidence


def load_evidence(kernel, *, cache_factory=None, compiler_path=None):
    """Resolve persisted names and verify exact binary/compiler/file identities."""
    metadata = kernel.metadata._asdict()
    if not isinstance(metadata.get("target"), dict) and hasattr(metadata.get("target"), "__dict__"):
        metadata["target"] = vars(metadata["target"])
    evidence = metadata.get("block_auto_evidence", {})
    if evidence.get("version") != VERSION or evidence.get("status") != "COLLECTED":
        raise ValueError("same-compilation model evidence unavailable")
    binary = kernel.kernel
    if not isinstance(binary, bytes) or digest(binary) != evidence["binary_sha256"]:
        raise ValueError("candidate binary changed after evidence collection")
    if compiler_path is None:
        from .utils import _get_npucompiler_path
        compiler_path, _ = _get_npucompiler_path()
    if file_digest(compiler_path) != evidence["compiler_sha256"]:
        raise ValueError("candidate compiler identity changed")
    cache = (cache_factory or _cache)(evidence["cache_key"])
    result = dict(metadata=metadata, binary_sha256=evidence["binary_sha256"],
                  compiler_sha256=evidence["compiler_sha256"], files={}, command=evidence.get("command",
                                                                                              []), evidence=evidence)
    for role, record in evidence["files"].items():
        name = record["name"]
        if Path(name).name != name:
            raise ValueError("invalid cached evidence name")
        path = cache.get_file(name)
        if not path or file_digest(path) != record["sha256"]:
            raise ValueError("cached model evidence changed: " + role)
        result[role] = Path(path).read_text()
        result["files"][role] = dict(path=str(path), sha256=record["sha256"])
    # These stages already belong to CompiledKernel's ordinary cache group.
    # Persist a model-local copy so downstream CPU LLVM preparation can consume
    # paths without knowing public compiler-cache implementation details.
    stages = {"ttir": kernel.asm.get("ttir"), "ttadapter": kernel.asm.get("ttadapter")}
    if not all(isinstance(text, str) and text for text in stages.values()):
        raise ValueError("candidate frontend IR unavailable")
    for role, text in stages.items():
        encoded = text.encode()
        if digest(encoded) != evidence.get(role + "_sha256"):
            raise ValueError("candidate frontend IR identity changed: " + role)
        path = cache.put(encoded, "block_auto_" + digest(encoded) + "." + role, binary=True)
        result[role] = text
        result["files"][role] = dict(path=str(path), sha256=digest(encoded))
    path = cache.put(binary, "block_auto_binary.o", binary=True)
    result["files"]["binary"] = dict(path=str(path), sha256=evidence["binary_sha256"])
    result["files"]["compiler"] = dict(path=str(compiler_path), sha256=evidence["compiler_sha256"])
    # Avoid a recursive metadata evidence structure; JSON includes the evidence
    # record itself, but that record does not refer to this derived copy.
    encoded = json.dumps(metadata, default=vars, sort_keys=True).encode()
    path = cache.put(encoded, "block_auto_" + digest(encoded) + ".json", binary=True)
    result["files"]["metadata"] = dict(path=str(path), sha256=digest(encoded))
    return result
