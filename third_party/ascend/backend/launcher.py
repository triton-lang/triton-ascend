# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.
"""Data plans and build/cache management for the fixed native Ascend launcher."""

from dataclasses import asdict, dataclass
from functools import lru_cache
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import re
import sysconfig
import tempfile

from triton.runtime.cache import get_cache_manager, get_dump_manager
from . import utils
from .program_grid import (ProgramGridContractError, get_persistent_transform,
                           normalize_program_grid_transforms)

# Keep these values in sync with launcher_abi.h. These are data tags, not C++
# spellings: a signature is decoded once when a native plan is constructed.
CONSTEXPR, POINTER, I8, I16, I32, I64, U8, U16, U32, U64, F32, F64 = range(12)
FFTS, PURE_SIMT, TASKQUEUE, AUTO_MAP, GRID_WARNING, COALESCE_CEIL, DYNAMIC_SHARED, DEVICE_PRINT, IAT, PTSM = (
    1 << i for i in range(10))

_KINDS = {
    "constexpr": CONSTEXPR, "i1": I32, "i8": I8, "i16": I16, "i32": I32, "i64": I64,
    "u1": U32, "u8": U8, "u16": U16, "u32": U32, "u64": U64,
    "fp16": F32, "bf16": F32, "fp32": F32, "f32": F32, "fp64": F64,
}


def ty_to_cpp(ty):
    if ty.startswith(("*", "tensordesc")):
        return "void*"
    return {
        I8: "int8_t", I16: "int16_t", I32: "int32_t", I64: "int64_t",
        U8: "uint8_t", U16: "uint16_t", U32: "uint32_t", U64: "uint64_t",
        F32: "float", F64: "double",
    }[_KINDS[ty]]


def _enabled(name, default=False):
    return os.getenv(name, str(default)).lower() in ("true", "1")


@dataclass(frozen=True)
class LaunchSpec:
    flags: int
    workspace_size: int
    ordered_locks: int
    unordered_locks: int
    lock_init_value: int
    participant_factor: int
    physical_blocks: int
    coalesce_factor: int
    coalesce_axis: int
    task_type: int
    mix_ratio: int
    shared_mem_dynamic_size: int

    def as_dict(self):
        return asdict(self)


def _program_grid_flags(metadata):
    # The compiler must explicitly report which rewriting and block-cap gates
    # applied. Missing fields must not silently select the legacy device ABI.
    for field in ("program_grid_mapping_applied", "row_coalescing_applied",
                  "auto_blockify_enabled", "ptsm_cap_authorized"):
        if not isinstance(getattr(metadata, field, None), bool):
            raise RuntimeError(f"compiler metadata missing {field}")
    raw = getattr(metadata, "program_grid_transforms", None)
    mapping = metadata.program_grid_mapping_applied
    if mapping and raw is None:
        raise RuntimeError("program_grid_mapping_applied requires program_grid_transforms")
    if not mapping and raw is not None:
        raise RuntimeError("program_grid_mapping_applied disagrees with program_grid_transforms")
    try:
        transforms = normalize_program_grid_transforms(raw) if mapping else None
    except ProgramGridContractError as error:
        raise RuntimeError(f"invalid program_grid_transforms launcher metadata: {error}") from error
    persistent = get_persistent_transform(transforms) if transforms is not None else None
    if transforms is not None and metadata.row_coalescing_applied:
        raise RuntimeError("program-grid transforms conflict with legacy RowCoalescing")
    if metadata.auto_blockify_enabled and (mapping or metadata.row_coalescing_applied):
        raise RuntimeError("auto_blockify_enabled conflicts with a rewritten program mapping")
    if metadata.auto_blockify_enabled and metadata.ptsm_cap_authorized:
        raise RuntimeError("auto_blockify_enabled and ptsm_cap_authorized cannot both be true")
    if persistent is None:
        if metadata.ptsm_cap_authorized:
            raise RuntimeError("ptsm_cap_authorized requires a persistent transform")
    elif not metadata.ptsm_cap_authorized:
        raise RuntimeError("persistent program-grid transform lacks PTSM cap authorization")
    elif metadata.mix_mode != "aiv":
        raise RuntimeError("persistent program-grid transform requires final mix_mode=aiv")

    factor = int(getattr(metadata, "coalesce_factor", 1) or 1)
    axis = int(getattr(metadata, "coalesce_axis", -1))
    ceil_div = bool(getattr(metadata, "coalesce_grid_ceil_div", False))
    if transforms is not None and factor > 1:
        raise RuntimeError("program-grid transforms conflict with legacy coalesce metadata")
    if metadata.row_coalescing_applied:
        if factor <= 1 or axis not in (0, 1, 2):
            raise RuntimeError("row_coalescing_applied requires valid coalesce metadata")
    elif factor != 1 or axis != -1 or ceil_div:
        raise RuntimeError("unmatched RowCoalescing must retain default coalesce metadata")
    if transforms is None:
        return 0
    # Encode exactly the three version-2 sequences supported by the compiler.
    # This extends the flag set without changing the version-1 C struct layout.
    # Reject a future schema/sequence until its native implementation is added.
    modes = {
        ((1, 16, False, False),): IAT,
        ((1, 16, False, False), (0, 4, True, True)): IAT | PTSM,
        ((0, 64, True, True),): PTSM,
    }
    sequence = tuple((t["axis"], t["factor"], t["persistent_coverage"],
                      t["grid_stride_abi_verified"]) for t in transforms["transforms"])
    if transforms["version"] != 2 or sequence not in modes:
        raise RuntimeError("unsupported native launcher program-grid transforms")
    return modes[sequence]


def make_launch_spec(metadata, npu_utils):
    """Translate compiler metadata without emitting native source code."""
    arch = metadata.target.arch
    flags = _program_grid_flags(metadata)
    if utils.is_ffts_supported(arch) and not utils.force_disable_ffts(arch):
        flags |= FFTS
    if metadata.is_pure_simt:
        flags |= PURE_SIMT
    if _enabled("TRITON_ENABLE_TASKQUEUE", True):
        flags |= TASKQUEUE
    if utils._is_auto_map_parallel_blocks_enabled() and not getattr(metadata, "has_auto_blockify_blacklist_op", False):
        flags |= AUTO_MAP
    if _enabled("TRITON_GRID_WARN_PRINT"):
        flags |= GRID_WARNING
    if _enabled("TRITON_DEVICE_PRINT"):
        flags |= DEVICE_PRINT
    if getattr(metadata, "coalesce_grid_ceil_div", False):
        flags |= COALESCE_CEIL
    enable_simt = "simt" in metadata.parallel_mode or metadata.is_pure_simt
    if metadata.compile_on_910_95 and enable_simt:
        flags |= DYNAMIC_SHARED
    locks = int(getattr(metadata, "sync_block_lock_layout", 0))
    ordered, unordered = locks & 0xFFFFFFFF, (locks >> 32) & 0xFFFFFFFF
    factor = 2 if unordered and metadata.mix_mode == "mix" and getattr(metadata, "auto_tile_and_bind_subblock", False) else 1
    task_type = 1 if metadata.mix_mode == "aiv" else 2
    encoded = int(getattr(metadata, "bs_task_type", 0))
    mix_ratio = 0
    if encoded:
        candidate, mix_ratio = divmod(encoded, 10)
        if candidate in (1, 2, 3, 4):
            task_type = candidate
    physical = npu_utils.get_aivector_core_num() if metadata.mix_mode == "aiv" else npu_utils.get_aicore_num()
    return LaunchSpec(
        flags=flags,
        workspace_size=max(int(getattr(metadata, "workspace_size", 0)), 0),
        ordered_locks=ordered, unordered_locks=unordered,
        lock_init_value=int(getattr(metadata, "lock_init_value", getattr(metadata, "lock_init_val", 0))),
        participant_factor=factor, physical_blocks=int(physical),
        coalesce_factor=int(getattr(metadata, "coalesce_factor", 1) or 1),
        coalesce_axis=int(getattr(metadata, "coalesce_axis", -1)),
        task_type=task_type, mix_ratio=mix_ratio,
        shared_mem_dynamic_size=int(getattr(metadata, "shared_mem_dynamic_size", 0)) if flags & DYNAMIC_SHARED else 0,
    )


def _descriptor_signature(sig):
    match = re.match(r"tensordesc<([^\[>]*)\[([^]]*)\]", sig)
    if not match:
        raise ValueError(f"Invalid tensor descriptor signature: {sig}")
    dtype, shape = match.groups()
    rank = shape.count(",") + 1
    return ["*" + dtype, *(["i64"] * (2 * rank)), "i1", *(["i32"] * rank), *(["i64"] * rank)]


def _expand_signature(sig):
    if isinstance(sig, tuple):
        return [ty for child in sig for ty in _expand_signature(child)]
    if sig.startswith("tensordesc"):
        return _descriptor_signature(sig)
    return [sig]


def argument_types(signature):
    result = []
    for sig in signature.values():
        for ty in _expand_signature(sig):
            if ty.startswith("*"):
                result.append((POINTER, utils.convert_sigtype_to_int(ty[1:])))
            else:
                result.append((_KINDS[ty], -1))
    return result


def make_tensordesc_arg(arg):
    return [arg.base, *arg.shape, *arg.strides, arg.padding == "nan", *arg.shape, *arg.strides]


def wrap_handle_tensordesc(launcher, signature):
    signatures = tuple(signature.values())
    if not any(isinstance(sig, tuple) or sig.startswith("tensordesc") for sig in signatures):
        return launcher

    def expand(sig, arg):
        if isinstance(sig, tuple):
            if not isinstance(arg, (tuple, list)) or len(arg) != len(sig):
                raise TypeError("Tuple argument does not match launcher signature")
            return [item for child, value in zip(sig, arg) for item in expand(child, value)]
        if sig.startswith("tensordesc"):
            return make_tensordesc_arg(arg)
        return [arg]

    def wrapped(*args):
        if len(args) != 9 + len(signatures):
            raise TypeError(f"launch expects {9 + len(signatures)} arguments, got {len(args)}")
        flattened = [item for sig, value in zip(signatures, args[9:]) for item in expand(sig, value)]
        return launcher(*args[:9], *flattened)

    return wrapped


def _runtime_source_identity():
    root = Path(__file__).with_name("launcher_src")
    # Content, rather than timestamps, also detects same-size edits within a
    # filesystem clock tick. The prepared mapping and digest are still reused.
    return tuple((name, (root / name).read_text()) for name in (
        "launcher_runtime.cpp", "launcher_abi.h", "launcher_args.h", "launcher_cann.h",
        "launcher_backend.h", "launcher_cache.h", "launcher_profiler.h",
    ))


@lru_cache(maxsize=8)
def _runtime_sources(identity):
    return dict(identity)


def _cache_relative(path):
    return "/".join(Path(path).parts[-2:])


@lru_cache(maxsize=32)
def _runtime_config(npu_utils_relative, backend, print_identity):
    config = f"#define TRITON_NPU_UTILS_RELATIVE {json.dumps(npu_utils_relative)}\n"
    if backend == "mindspore":
        config += "#define TRITON_LAUNCHER_MINDSPORE\n"
    sources = {"launcher_config.h": config}
    if print_identity is not None:
        sources["launcher_config.h"] += "#define TRITON_LAUNCHER_DEVICE_PRINT\n"
        sources["launcher_device_print.h"] = extract_device_print_code_from_cann()
    return sources


@lru_cache(maxsize=128)
def _shared_key(fingerprint, sources):
    inputs = [json.loads(fingerprint), dict(sources)]
    return hashlib.sha256(json.dumps(inputs, sort_keys=True).encode()).hexdigest()


def _build_shared(name, sources, debug=False):
    fingerprint = utils.npu_extension_fingerprint(name, extra_cflags=("-O3",))
    key = _shared_key(json.dumps(fingerprint, sort_keys=True), tuple(sorted(sources.items())))
    cache = get_cache_manager(key)
    filename = name + sysconfig.get_config_var("EXT_SUFFIX")
    if debug:
        dump = get_dump_manager(key)
        for source_name, content in sources.items():
            dump.put(content, source_name, binary=False)

    def build():
        with tempfile.TemporaryDirectory() as temp:
            for source_name, content in sources.items():
                Path(temp, source_name).write_text(content)
            source_path = Path(temp, "launcher_runtime.cpp" if name == "__triton_launcher_runtime" else "launcher_export.cpp")
            built = utils._build_npu_ext(name, str(source_path), extra_cflags=("-O3",))
            return Path(built).read_bytes()

    return utils._get_or_build_npu_artifact(cache, filename, build)


@lru_cache()
def _load_runtime(path):
    spec = importlib.util.spec_from_file_location("__triton_launcher_runtime", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def get_runtime(npu_utils_path, debug=False):
    print_identity = None
    if _enabled("TRITON_DEVICE_PRINT"):
        # CANN/loaded framework replacement requires a new process. Track the
        # compiler selection and version-file identity for derived print code.
        bisheng = utils._get_bisheng_path()
        version_file = utils._find_cann_version_file()
        print_identity = (bisheng, utils._file_identity(bisheng),
                          version_file, utils._file_identity(version_file) if version_file else None)
    sources = {**_runtime_sources(_runtime_source_identity()),
               **_runtime_config(_cache_relative(npu_utils_path), utils.backend_policy, print_identity)}
    path = _build_shared("__triton_launcher_runtime", sources, debug)
    return _load_runtime(path), path


def export_launcher(spec, types, runtime_path, debug=False):
    """Bind the legacy C ABI using constants and a fixed export adapter."""
    fields = ", ".join(str(value) for value in spec.as_dict().values())
    initializers = ", ".join(f"{{{kind}, {dtype}}}" for kind, dtype in types) or "{0, -1}"
    config = (
        f"#define TRITON_EXPORT_RUNTIME_RELATIVE {json.dumps(_cache_relative(runtime_path))}\n"
        f"static constexpr TritonNpuLaunchSpecV1 spec = {{1, sizeof(TritonNpuLaunchSpecV1), {fields}}};\n"
        f"static constexpr TritonNpuArgTypeV1 types[] = {{{initializers}}};\n"
        f"static constexpr size_t numTypes = {len(types)};\n"
    )
    native = _runtime_sources(_runtime_source_identity())
    sources = {name: native[name] for name in ("launcher_abi.h", "launcher_cache.h")}
    export_source = str(Path(__file__).with_name("launcher_src") / "launcher_export.cpp")
    sources["launcher_export.cpp"] = Path(export_source).read_text()
    sources["launcher_export_config.h"] = config
    return _build_shared("launcher_export", sources, debug)


def extract_device_print_code_from_cann():
    from triton.backends.ascend.utils import _get_bisheng_path
    ccec_compiler_bin_folder, _ = os.path.split(os.path.realpath(_get_bisheng_path()))
    ccec_compiler_folder, _ = os.path.split(ccec_compiler_bin_folder)
    clang_version = os.listdir(os.path.join(ccec_compiler_folder, "lib/clang/"))[0]
    ccelib_path = os.path.join(ccec_compiler_folder, f"lib/clang/{clang_version}/include/ccelib")

    def read_header(header_path):
        with open(os.path.join(ccelib_path, header_path), 'r') as f:
            code = f.read()

        # remove all #include "..."
        lines = code.splitlines()
        purged_lines = []
        for line in lines:
            normalized_line = ' '.join(line.split())
            if not normalized_line.startswith('#include "'):
                purged_lines.append(line)
        code = '\n'.join(purged_lines)

        # remove [aicore] functions
        aicore_positions = []
        for m in re.finditer(r'\[aicore\]', code):
            aicore_positions.append(m.start())

        def find_aicore_function_span(src, pos):
            for i in range(pos - 1, -1, -1):
                if src[i] == '}':  # this relies on that all [aicore] functions come after normal functions
                    left = i + 1
                    break
            n = len(src)
            brace_nest = 0
            for j in range(pos, n, 1):
                if src[j] == '{':
                    brace_nest += 1
                elif src[j] == '}':
                    brace_nest -= 1
                    if brace_nest == 0:
                        right = j
                        break
            return left, right

        new_code = ''
        segment_start = 0
        for pos in aicore_positions:
            left, right = find_aicore_function_span(code, pos)
            new_code += code[segment_start:left]
            segment_start = right + 1
        new_code += code[segment_start:]

        # remove __gm__ and rename macros
        new_code = new_code.replace('__gm__', ' ')
        new_code = new_code.replace('__CCELIB_RT_ERROR_NONE', 'RT_ERROR_NONE')
        new_code = new_code.replace('__CCELIB_RT_MEMORY_HBM', 'RT_MEMORY_HBM')
        new_code = new_code.replace('__CCELIB_RT_MEMCPY_HOST_TO_DEVICE', 'RT_MEMCPY_HOST_TO_DEVICE')
        new_code = new_code.replace('__CCELIB_RT_MEMCPY_DEVICE_TO_HOST', 'RT_MEMCPY_DEVICE_TO_HOST')
        return new_code

    # the following headers should be included in this order
    return '\n'.join([
        read_header('common/common_impl.h'),
        read_header('internal/debug_tunnel/payload.h'),
        read_header('internal/debug_tunnel/payload_impl.h'),
        read_header('internal/debug_tunnel/tunnel.h'),
        read_header('internal/debug_tunnel/tunnel_impl.h')
    ])
