# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Collect compiler output, load resource inputs, then load performance inputs."""
import os
from pathlib import Path
import shutil
import subprocess
import tempfile

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
        target = metadata.get("target")
        arch = target.get("arch", "") if isinstance(target, dict) else getattr(target, "arch", "")
        # Published A3 consumers retain final allocation IR but remove LLVM.
        # Its conservative resource/occupancy policy does not estimate cycles.
        if not arch.startswith("Ascend910_93") and not any(key.startswith("llvm") for key in sources):
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


def load_resources(candidate, *, cache_factory=None):
    """Load allocation IR and metadata; keep LLVM paths without reading bodies."""
    result = {key: candidate[key] for key in ("factor", "axis", "status", "error") if key in candidate}
    if candidate.get("kernel") is None:
        return result
    try:
        metadata = candidate["kernel"].metadata._asdict()
        if not isinstance(metadata.get("target"), dict) and hasattr(metadata.get("target"), "__dict__"):
            metadata["target"] = vars(metadata["target"])
        evidence = metadata.get("block_auto_evidence", {})
        if evidence.get("version") != VERSION or evidence.get("status") != "COLLECTED":
            raise ValueError("same-compilation model evidence unavailable")
        cache = (cache_factory or _cache)(evidence["cache_key"])
        files = {}
        for role, record in evidence["files"].items():
            path = cache.get_file(record["name"])
            if not path:
                raise ValueError("cached model input unavailable: " + role)
            files[role] = dict(path=str(path))
        final_ir = Path(files["final_ir"]["path"]).read_text()
        files["compiler"] = dict(path=evidence["compiler"])
        result.update(metadata=metadata, files=files, final_ir=final_ir)
    except (ValueError, KeyError, OSError) as error:
        result.update(status="RESOURCE_UNCONFIRMED", error=str(error))
    return result


def load_analysis(resources, state):
    """Resolve model inputs after native resource checks, without device execution.

    CV uses preserved LLVM modules. A5 VV requests optimized LLVM; A3 VV uses
    resource facts alone. Cache/tool details stay here, outside the selector.
    """
    item = dict(resources, **state)
    try:
        for role, record in item.get("files", {}).items():
            if role.startswith("llvm") and not (role == "llvm" and item.get("vv")):
                item[role] = Path(record["path"]).read_text()
        if state["needs_llvm"]:
            item["optimized_llvm"] = _optimized_llvm(item)
    except (ValueError, KeyError, OSError, subprocess.SubprocessError) as error:
        item.update(status="STATIC_ANALYSIS_UNSUPPORTED", error=str(error))
    return {key: value for key, value in item.items() if key not in ("metadata", "files", "final_ir", "meta_library")}


def _optimized_llvm(item):
    """Reuse or generate the VV optimizer output under the candidate's JIT key."""
    from .utils import _get_bisheng_path
    cache = _cache(item['metadata']['hash'])
    name = 'block_auto_optimized.ll'
    cached = cache.get_file(name)
    if cached:
        return Path(cached).read_text()
    files = item['files']
    compiler = Path(_get_bisheng_path()).resolve()
    link = compiler.with_name('llvm-link')
    if not link.is_file():
        link = Path(shutil.which('llvm-link') or '')
    if not compiler.is_file() or not link.is_file():
        raise ValueError('installed optimizer/LLVM decoder is unavailable')
    meta = item.get('meta_library')
    if meta is None:
        raise ValueError('MIX needs one final-IR AIV bitcode reference')
    library = Path(files['compiler']['path']).parent.parent.parent / 'lib/libdevice.10.bc'
    directory = getattr(cache, 'cache_dir', None)
    if not directory:
        raise ValueError('optimized LLVM requires a local compiler cache')
    with tempfile.TemporaryDirectory(prefix='block-auto-', dir=directory) as temporary:
        output = Path(temporary)
        args = [
            str(compiler), '-c', '-emit-llvm', '-x', 'ir', files['llvm']['path'], '-o',
            str(output / 'kernel.bc'), '--cce-aicore-arch=dav-c310-vec', '--cce-aicore-only', '-O2',
            '-cce-bitcode-is-aicore', '-Wno-override-module', '-mllvm', '--cce-vf-auto-sync=global', '-mllvm',
            '-cce-aicore-dcci-insert-for-scalar=false', '-mllvm', '-enable-module-inliner', '-mllvm',
            '-replexitval=unusedindvarinloop', '-mllvm', '-cce-vf-aa-between-iters=true', '-mllvm',
            '-cce-vf-enable-loop-fusion=false', '--cce-disable-asc-reserved-ubuf', '-mllvm',
            '-cce-vf-stack-size=0x2000', '--cce-long-scbz=true', '-cce-link-aicore-ll-module',
            str(meta), '-cce-link-aicore-ll-module',
            str(library)
        ]
        env = dict(os.environ, ASCEND_RT_VISIBLE_DEVICES='', ASCEND_VISIBLE_DEVICES='', CUDA_VISIBLE_DEVICES='')
        for command in (args, [str(link), '-S', str(output / 'kernel.bc'), '-o', str(output / 'optimized.ll')]):
            subprocess.run(command, capture_output=True, env=env, check=True, timeout=30)
        text = (output / 'optimized.ll').read_text()
    cache.put(text, name)
    return text
