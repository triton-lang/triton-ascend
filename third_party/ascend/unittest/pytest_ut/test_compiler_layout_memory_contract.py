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
"""Source-level contracts for the layout / memory-access compiler closure.

These tests intentionally load ``backend/compiler.py`` from this checkout.
The installed Triton package can point at another worktree, so importing
``triton.backends.ascend.compiler`` directly would not validate the source
being changed here.
"""

import importlib.util
import itertools
import sys
import types
import warnings
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.backend("none")

_UNSET = object()


def _stub_graph_ub_budget_bytes_for_arch(arch):
    """Mirror the documented architecture table for the compiler import shim.

    The table itself is covered by the source-loaded backend-utils test.  This
    local shim keeps this compiler-only contract independent of the installed
    Ascend package while retaining meaningful architecture-budget assertions.
    """
    if not isinstance(arch, str) or not arch:
        return 0
    if arch.startswith(("Ascend910_95", "Ascend950")):
        return 128 * 1024
    if arch.startswith((
            "Ascend910A",
            "Ascend910B",
            "Ascend910D",
            "Ascend910_93",
            "Ascend310B",
    )):
        return 96 * 1024
    return 0


def _stub_ub_size_in_kbytes_for_arch(arch):
    """Mirror the raw mapping-UB table used by the source compiler.

    StoreCoalescing retains the conservative half-UB budget above, whereas
    program mapping must receive the physical UB capacity.  Keeping both in
    this import shim makes the source-level call contract executable without
    importing whichever Ascend wheel happens to be installed.
    """
    return _stub_graph_ub_budget_bytes_for_arch(arch) * 2 // 1024


class _FakeModule:

    def __init__(self, events):
        self.context = object()
        self._events = events
        self._string_count = 0

    def __str__(self):
        self._events.append(f"str:{self._string_count}")
        self._string_count += 1
        return "module {}"


class _FakePassManager:

    def __init__(self, events):
        self._events = events

    def enable_debug(self):
        self._events.append("enable_debug")

    def run(self, _module, _pipeline_name):
        self._events.append("run_row")


@pytest.fixture(scope="module")
def compiler_module():
    """Load this checkout's compiler without depending on installed Ascend utils.

    The test invokes only ``ttir_to_npubin`` and replaces all external tool
    interactions below.  A tiny import-time shim keeps the source-level
    contract test runnable when the installed Triton wheel predates an import
    added by this checkout (for example ``_enable_msdebug``).
    """
    compiler_path = Path(__file__).resolve().parents[2] / "backend" / "compiler.py"
    module_name = "triton.backends.ascend.compiler_layout_memory_contract_under_test"
    utils_name = "triton.backends.ascend.utils"
    driver_name = "triton.backends.ascend.driver"
    cache_name = "triton.runtime.cache"
    program_grid_name = "triton.backends.ascend.program_grid"

    def return_false(*_args, **_kwargs):
        return False

    def remove_deprecated_npu_options(options, *, in_place=False):
        normalized = options if in_place else dict(options)
        normalized.pop("compile_on_910_95", None)
        return normalized

    utils_stub = types.ModuleType(utils_name)
    for name in (
            "_check_bishengir_api_change",
            "_check_bishengir_able_save_ir",
            "_check_bishengir_is_regbased",
            "_enable_print_ub_bits",
            "_enable_dump_memory_info",
            "_enable_msdebug",
            "_is_ascend_sanitizer_enabled",
            "_is_debug_line_info_disabled",
            "_is_auto_map_parallel_blocks_enabled",
            "force_disable_ffts",
    ):
        setattr(utils_stub, name, return_false)
    for name in (
            "_get_kernel_target",
            "_get_triton_adapter_opt_path",
            "_get_triton_mlir_opt_path",
            "_get_triton_opt_path",
            "_get_bishengir_opt_path",
    ):
        setattr(utils_stub, name, lambda *_args, **_kwargs: "")
    utils_stub._get_npucompiler_path = lambda *_args, **_kwargs: ("", {})
    utils_stub._get_auto_blockify_blacklist_reasons = lambda *_args, **_kwargs: []
    utils_stub._warn_auto_blockify_disabled = lambda *_args, **_kwargs: None
    utils_stub._remove_deprecated_npu_options = remove_deprecated_npu_options
    utils_stub._warn_deprecated_npu_option = lambda name: warnings.warn(
        f"Ascend compile option '{name}' is deprecated and ignored.", FutureWarning)
    utils_stub._warn_deprecated_ascend_env_vars = lambda: None
    utils_stub.downgrade_llir = lambda llir: llir
    utils_stub.get_cann_version_file_hash = lambda: ""
    utils_stub.graph_ub_budget_bytes_for_arch = _stub_graph_ub_budget_bytes_for_arch
    utils_stub.ub_size_in_kbytes_for_arch = _stub_ub_size_in_kbytes_for_arch

    class UnusedNPUUtils:
        pass

    driver_stub = types.ModuleType(driver_name)
    driver_stub.NPUUtils = UnusedNPUUtils

    cache_stub = types.ModuleType(cache_name)
    cache_stub._base32 = lambda value: str(value)
    cache_stub.get_dump_manager = lambda *_args, **_kwargs: SimpleNamespace(cache_dir="", put=lambda *_args, **_kwargs:
                                                                            None)

    # Initialize the installed Triton package before temporarily replacing its
    # cache module below.  Loading compiler.py starts from triton._C; doing it
    # in the reverse order would make unrelated backend imports see the tiny
    # test stub instead of the runtime cache API they require.
    importlib.import_module("triton")

    previous_utils = sys.modules.get(utils_name)
    previous_driver = sys.modules.get(driver_name)
    previous_cache = sys.modules.get(cache_name)
    previous_program_grid = sys.modules.get(program_grid_name)
    sys.modules[utils_name] = utils_stub
    sys.modules[driver_name] = driver_stub
    sys.modules[cache_name] = cache_stub
    sys.modules.pop(module_name, None)
    program_grid_spec = importlib.util.spec_from_file_location(program_grid_name,
                                                               compiler_path.with_name("program_grid.py"))
    program_grid = importlib.util.module_from_spec(program_grid_spec)
    assert program_grid_spec is not None and program_grid_spec.loader is not None
    sys.modules[program_grid_name] = program_grid
    program_grid_spec.loader.exec_module(program_grid)
    debug_line_rewriter_name = "triton.backends.ascend.debug_line_rewriter"
    previous_debug_line_rewriter = sys.modules.get(debug_line_rewriter_name)
    sys.modules.pop(debug_line_rewriter_name, None)
    try:
        debug_line_rewriter_path = compiler_path.with_name("debug_line_rewriter.py")
        debug_line_rewriter_spec = importlib.util.spec_from_file_location(debug_line_rewriter_name,
                                                                          debug_line_rewriter_path)
        debug_line_rewriter = importlib.util.module_from_spec(debug_line_rewriter_spec)
        assert debug_line_rewriter_spec is not None and debug_line_rewriter_spec.loader is not None
        sys.modules[debug_line_rewriter_name] = debug_line_rewriter
        debug_line_rewriter_spec.loader.exec_module(debug_line_rewriter)
        spec = importlib.util.spec_from_file_location(module_name, compiler_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
        if previous_debug_line_rewriter is None:
            sys.modules.pop(debug_line_rewriter_name, None)
        else:
            sys.modules[debug_line_rewriter_name] = previous_debug_line_rewriter
        if previous_utils is None:
            sys.modules.pop(utils_name, None)
        else:
            sys.modules[utils_name] = previous_utils
        if previous_driver is None:
            sys.modules.pop(driver_name, None)
        else:
            sys.modules[driver_name] = previous_driver
        if previous_cache is None:
            sys.modules.pop(cache_name, None)
        else:
            sys.modules[cache_name] = previous_cache
        if previous_program_grid is None:
            sys.modules.pop(program_grid_name, None)
        else:
            sys.modules[program_grid_name] = previous_program_grid
    return module


def _parse_options(compiler, arch, opts=None):
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch=arch))
    return backend.parse_options({} if opts is None else opts)


@pytest.mark.skip(reason="The case is not supported on A5, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("arch", "requested_capacity", "expected_capacity"),
    (
        ("Ascend910B1", _UNSET, 96 * 1024),
        ("Ascend910B1", None, 96 * 1024),
        ("Ascend910_9581", None, 128 * 1024),
        ("Ascend950A3", None, 128 * 1024),
        ("Ascend910B1", 0, 0),
        ("Ascend910B1", 4096, 4096),
        ("Ascend910B1", 96 * 1024 + 1, 96 * 1024),
        ("Ascend910_9581", 128 * 1024 + 1, 128 * 1024),
        ("unknown-arch", None, 0),
    ),
)
def test_npu_options_normalizes_graph_ub_budget(compiler_module, arch, requested_capacity, expected_capacity):
    """Direct NPUOptions users receive the same final integer as JIT users."""
    kwargs = {"arch": arch}
    if requested_capacity is not _UNSET:
        kwargs["graph_optimize_ub_capacity_bytes"] = requested_capacity

    options = compiler_module.NPUOptions(**kwargs)

    assert options.graph_optimize_ub_capacity_bytes == expected_capacity


@pytest.mark.skip(reason="The case is not supported on A5, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("arch", "requested_capacity", "expected_capacity"),
    (
        ("Ascend910B1", _UNSET, 96 * 1024),
        ("Ascend910B1", None, 96 * 1024),
        ("Ascend910_9581", None, 128 * 1024),
        ("Ascend950A3", None, 128 * 1024),
        ("Ascend910B1", 0, 0),
        ("Ascend910B1", 4096, 4096),
        ("Ascend910B1", 96 * 1024 + 1, 96 * 1024),
    ),
)
def test_parse_options_normalizes_graph_ub_budget(compiler_module, arch, requested_capacity, expected_capacity):
    opts = {}
    if requested_capacity is not _UNSET:
        opts["graph_optimize_ub_capacity_bytes"] = requested_capacity

    options = _parse_options(compiler_module, arch, opts)

    assert options.arch == arch
    assert not hasattr(options, "_arch")
    assert options.graph_optimize_ub_capacity_bytes == expected_capacity


@pytest.mark.skip(reason="The case is not supported on A5, skipping for now. Will be fixed in future.")
def test_normalized_graph_ub_budget_contributes_to_npu_hash(compiler_module):
    auto = compiler_module.NPUOptions(arch="Ascend910B1")
    explicit_none = compiler_module.NPUOptions(arch="Ascend910B1", graph_optimize_ub_capacity_bytes=None)
    disabled = compiler_module.NPUOptions(arch="Ascend910B1", graph_optimize_ub_capacity_bytes=0)
    small = compiler_module.NPUOptions(arch="Ascend910B1", graph_optimize_ub_capacity_bytes=4096)
    clamped = compiler_module.NPUOptions(arch="Ascend910B1", graph_optimize_ub_capacity_bytes=96 * 1024 + 1)

    assert auto.__dict__["graph_optimize_ub_capacity_bytes"] == 96 * 1024
    assert explicit_none.graph_optimize_ub_capacity_bytes == 96 * 1024
    assert clamped.graph_optimize_ub_capacity_bytes == 96 * 1024
    assert auto.hash() == explicit_none.hash() == clamped.hash()
    assert auto.hash() != disabled.hash()
    assert auto.hash() != small.hash()


@pytest.mark.skip(reason="The case is not supported on A5, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("requested_capacity", "error_type"),
    (
        (-1, ValueError),
        (True, TypeError),
        (1.5, TypeError),
    ),
)
def test_npu_options_rejects_invalid_graph_ub_budget_requests(compiler_module, requested_capacity, error_type):
    with pytest.raises(error_type):
        compiler_module.NPUOptions(
            arch="Ascend910B1",
            graph_optimize_ub_capacity_bytes=requested_capacity,
        )


def _make_opt(
    *,
    is_pure_simt,
    superblock_factor=0,
    enable_bishengir_simt_optimization=0,
    simt_stack_limit=None,
    shared_mem_dynamic_size=None,
    enable_simt_reorder_instruction=False,
    disable_fma=False,
):
    return SimpleNamespace(
        is_pure_simt=is_pure_simt,
        num_warps=4,
        warp_size=32,
        enable_bishengir_simt_optimization=enable_bishengir_simt_optimization,
        simt_stack_limit=simt_stack_limit,
        shared_mem_dynamic_size=shared_mem_dynamic_size,
        enable_simt_reorder_instruction=enable_simt_reorder_instruction,
        disable_fma=disable_fma,
        superblock_factor=superblock_factor,
    )


def _run_ttir_to_npubin(
    compiler,
    monkeypatch,
    *,
    is_pure_simt=True,
    auto_map_enabled=False,
    has_blacklist_op=False,
    row_coalescing_applied=False,
    superblock_factor=0,
    common_options=(),
    enable_bishengir_simt_optimization=0,
    simt_stack_limit=None,
    resolved_simt_stack_limit=1152,
    shared_mem_dynamic_size=None,
    enable_simt_reorder_instruction=False,
    disable_fma=False,
):
    events = []
    commands = []
    module = _FakeModule(events)
    pass_manager = _FakePassManager(events)

    def parse_ttir_metadata(_ttir, metadata):
        events.append("parse")
        return {
            **metadata,
            "has_auto_blockify_blacklist_op": has_blacklist_op,
            "row_coalescing_applied": row_coalescing_applied,
        }

    def export_program_grid_metadata(_mod, _metadata, *, require_row_contract=False):
        events.append(f"export:{require_row_contract}")

    def run_bisheng(command, **_kwargs):
        commands.append(list(command))
        output = Path(command[command.index("-o") + 1] + ".o")
        output.write_bytes(b"npubin")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(
        compiler,
        "ir",
        SimpleNamespace(pass_manager=lambda _context: (events.append("pass_manager") or pass_manager)),
    )
    monkeypatch.setattr(compiler, "_parse_ttir_metadata", parse_ttir_metadata)
    monkeypatch.setattr(compiler, "_export_program_grid_metadata", export_program_grid_metadata)
    monkeypatch.setattr(
        compiler,
        "get_common_bishengir_compile_options",
        lambda _metadata: list(common_options),
    )
    monkeypatch.setattr(compiler, "_get_npucompiler_path", lambda: ("/fake/bisheng", {}))
    monkeypatch.setattr(
        compiler,
        "_is_auto_map_parallel_blocks_enabled",
        lambda: auto_map_enabled,
    )

    # Keep this argv matrix independent of the host torch_npu configuration
    # while checking that Pure-SIMT passes the explicit option to the resolver.
    def get_simt_stack_limit(user_stack_limit):
        assert user_stack_limit == simt_stack_limit
        return resolved_simt_stack_limit if user_stack_limit is None else user_stack_limit

    monkeypatch.setattr(compiler, "get_simt_stack_limit", get_simt_stack_limit)
    monkeypatch.setattr(compiler.subprocess, "run", run_bisheng)

    result = compiler.ttir_to_npubin(
        module,
        {"bisheng_options": None},
        _make_opt(
            is_pure_simt=is_pure_simt,
            superblock_factor=superblock_factor,
            enable_bishengir_simt_optimization=enable_bishengir_simt_optimization,
            simt_stack_limit=simt_stack_limit,
            shared_mem_dynamic_size=shared_mem_dynamic_size,
            enable_simt_reorder_instruction=enable_simt_reorder_instruction,
            disable_fma=disable_fma,
        ),
    )
    assert result == b"npubin"
    assert len(commands) == 1
    return events, commands[0]


@pytest.mark.skip(reason="The case is not supported on A5, skipping for now. Will be fixed in future.")
def test_export_coalesce_metadata_removes_attrs_and_marks_row(compiler_module, monkeypatch):
    removed = []

    def get_int_attr(module, name):
        return module.attrs.get(name)

    def remove_attr(module, name):
        removed.append(name)
        module.attrs.pop(name, None)

    monkeypatch.setattr(
        compiler_module,
        "ascend",
        SimpleNamespace(ir=SimpleNamespace(
            get_int_attr=get_int_attr,
            remove_attr=remove_attr,
        )),
    )

    coalesced = SimpleNamespace(attrs={
        "hacc.coalesce_factor": 4,
        "hacc.coalesce_axis": 2,
        "hacc.coalesce_grid_ceil_div": 1,
    })
    metadata = {}
    compiler_module._export_coalesce_metadata(coalesced, metadata)

    assert metadata == {
        "coalesce_factor": 4,
        "coalesce_axis": 2,
        "coalesce_grid_ceil_div": True,
        "row_coalescing_applied": True,
    }
    assert coalesced.attrs == {}
    assert removed == [
        "hacc.coalesce_factor",
        "hacc.coalesce_axis",
        "hacc.coalesce_grid_ceil_div",
    ]

    uncoalesced = SimpleNamespace(attrs={})
    uncoalesced_metadata = {}
    compiler_module._export_coalesce_metadata(uncoalesced, uncoalesced_metadata)
    assert uncoalesced_metadata == {
        "coalesce_factor": 1,
        "coalesce_axis": -1,
        "coalesce_grid_ceil_div": False,
        "row_coalescing_applied": False,
    }


@pytest.mark.skip(reason="The case is not supported on A5, skipping for now. Will be fixed in future.")
def test_export_coalesce_metadata_rejects_partial_row_contract(compiler_module, monkeypatch):

    def get_int_attr(module, name):
        return module.attrs.get(name)

    def remove_attr(module, name):
        module.attrs.pop(name, None)

    monkeypatch.setattr(
        compiler_module,
        "ascend",
        SimpleNamespace(ir=SimpleNamespace(
            get_int_attr=get_int_attr,
            remove_attr=remove_attr,
        )),
    )

    with pytest.raises(RuntimeError, match="launch contract"):
        compiler_module._export_coalesce_metadata(
            SimpleNamespace(attrs={"hacc.coalesce_factor": 4}),
            {},
            require_row_contract=True,
        )

    with pytest.raises(RuntimeError, match="RowCoalescing"):
        compiler_module._export_coalesce_metadata(
            SimpleNamespace(attrs={
                "hacc.coalesce_factor": 4,
                "hacc.coalesce_axis": 0,
            }),
            {},
            require_row_contract=True,
        )


@pytest.mark.skip(reason="The case is not supported on A5, skipping for now. Will be fixed in future.")
def test_ttir_to_npubin_exports_make_ttir_row_contract_only_for_pure_simt(compiler_module, monkeypatch):
    events, _command = _run_ttir_to_npubin(
        compiler_module,
        monkeypatch,
        is_pure_simt=True,
    )
    assert events == [
        "str:0",
        "parse",
        "export:True",
        "str:1",
    ]

    with monkeypatch.context() as pure_simt_off:
        events, _command = _run_ttir_to_npubin(
            compiler_module,
            pure_simt_off,
            is_pure_simt=False,
        )
    assert events == ["str:0", "parse"]


def _run_make_ttir_with_recorded_graph_options(compiler, monkeypatch, options):
    events = []
    module = _FakeModule(events)
    pass_manager = _FakePassManager(events)
    graph_calls = []

    def record(name):
        return lambda _pm, *args, **kwargs: events.append((name, args, kwargs))

    monkeypatch.setattr(
        compiler,
        "ir",
        SimpleNamespace(pass_manager=lambda _context: pass_manager),
    )
    monkeypatch.setattr(
        compiler,
        "passes",
        SimpleNamespace(
            common=SimpleNamespace(
                add_inliner=record("inliner"),
                add_canonicalizer=record("canonicalizer"),
                add_cse=record("cse"),
                add_licm=record("licm"),
                add_symbol_dce=record("symbol_dce"),
            ),
            ttir=SimpleNamespace(
                add_rewrite_tensor_descriptor_to_pointer=record("rewrite_tensor_descriptor_to_pointer"),
                add_combine=record("combine"),
                add_reorder_broadcast=record("reorder_broadcast"),
                add_loop_unroll=record("loop_unroll"),
            ),
        ),
    )
    monkeypatch.setattr(
        compiler,
        "ascend",
        SimpleNamespace(passes=SimpleNamespace(ttir=SimpleNamespace(
            add_graph_optimize=lambda _pm, **kwargs: graph_calls.append(kwargs)))),
    )

    assert compiler.make_ttir(module, {}, options) is module
    return events, graph_calls


def test_make_ttir_passes_canonical_compile_mode_to_graph_optimize(compiler_module, monkeypatch):
    options = SimpleNamespace(
        enable_graph_optimize=True,
        target_arch="Ascend910B1",
        compile_mode="simt_only",
        debug=False,
    )

    events, graph_calls = _run_make_ttir_with_recorded_graph_options(compiler_module, monkeypatch, options)

    assert graph_calls == [{
        "ub_capacity_bytes": 96 * 1024,
        "compile_mode": "simt_only",
    }]
    assert events[-1] == "run_row"


def test_make_ttir_forwards_iat_rule_and_explicit_resource_snapshot(
    compiler_module,
    monkeypatch,
):
    """IAT must be opt-in and receive real target facts, never defaults."""
    options = SimpleNamespace(
        enable_graph_optimize=True,
        target_arch="Ascend910B1",
        compile_mode="simd_simt_template",
        debug=False,
        program_mapping_rule_mask=512,
    )
    monkeypatch.setattr(
        compiler_module,
        "NPUUtils",
        lambda: SimpleNamespace(get_aivector_core_num=lambda: 40),
    )

    events, graph_calls = _run_make_ttir_with_recorded_graph_options(
        compiler_module,
        monkeypatch,
        options,
    )

    assert graph_calls == [{
        "ub_capacity_bytes": 96 * 1024,
        "compile_mode": "simd_simt_template",
        "rule_mask": 512,
        "device_core_count": 40,
        "min_programs_per_core": 1,
        "ub_safety_percent": 80,
        "reserved_ub_bytes": 0,
        "mapping_ub_capacity_bytes": 192 * 1024,
        "store_coalescing_ub_budget_bytes": 96 * 1024,
    }]
    assert events[-1] == "run_row"


def test_make_ttir_forwards_static_axis_fusion_rule_and_resource_snapshot(
    compiler_module,
    monkeypatch,
):
    options = SimpleNamespace(
        enable_graph_optimize=True,
        target_arch="Ascend910B1",
        compile_mode="simd_simt_template",
        # An AOT-shaped caller can still request the rule. It deliberately
        # carries no specialization attr and therefore remains a C++ no-op.
        program_grid_specialization=None,
        program_mapping_rule_mask=1024,
        debug=False,
    )
    monkeypatch.setattr(
        compiler_module,
        "NPUUtils",
        lambda: SimpleNamespace(get_aivector_core_num=lambda: 40),
    )

    events, graph_calls = _run_make_ttir_with_recorded_graph_options(
        compiler_module,
        monkeypatch,
        options,
    )

    assert graph_calls == [{
        "ub_capacity_bytes": 96 * 1024,
        "compile_mode": "simd_simt_template",
        "rule_mask": 1024,
        "device_core_count": 40,
        "min_programs_per_core": 1,
        "ub_safety_percent": 80,
        "reserved_ub_bytes": 0,
        "mapping_ub_capacity_bytes": 192 * 1024,
        "store_coalescing_ub_budget_bytes": 96 * 1024,
    }]
    assert events[-1] == "run_row"


@pytest.mark.skip(reason="The case is not supported on A5, skipping for now. Will be fixed in future.")
def test_npu_options_do_not_expose_graph_remark_switch(compiler_module):
    """Graph rewrite logging is controlled by LLVM DEBUG, not an NPU option."""
    assert "graph_optimize_emit_remarks" not in compiler_module.NPUOptions.__dataclass_fields__


@pytest.mark.skip(reason="The case is not supported on A5, skipping for now. Will be fixed in future.")
@pytest.mark.parametrize(
    ("arch", "expected_capacity"),
    (
        ("Ascend910B1", 96 * 1024),
        ("Ascend910_9581", 128 * 1024),
        ("Ascend950A3", 128 * 1024),
        ("unknown-arch", 0),
    ),
)
def test_make_ttir_forwards_normalized_graph_ub_budget(compiler_module, monkeypatch, arch, expected_capacity):
    options = compiler_module.NPUOptions(arch=arch)

    events, graph_calls = _run_make_ttir_with_recorded_graph_options(compiler_module, monkeypatch, options)

    assert graph_calls[0]["ub_capacity_bytes"] == expected_capacity
    assert events[-1] == "run_row"


def test_ttir_to_npubin_auto_blockify_argv_matrix(compiler_module, monkeypatch):
    """Keep the internal-policy-and-safety pure-SIMT auto-blockify argv contract."""
    common_options = ["--common-before-pure-simt", "--common-after-pure-simt"]
    pure_simt_prefix = [
        "--enable-hivm-compile=false",
        "--enable-triton-ir-compile",
        "--pure-simt",
        "--num-warps=4",
        "--threads-per-warp=32",
        "--enable-bishengir-simt-optimization=17",
        "--simt-stack-limit=64",
        "--shared-mem-dynamic-size=4096",
        "--enable-simt-reorder-instruction=true",
        "--disable-fma",
    ]
    auto_blockify_flag = "--enable-auto-blockify-loop"

    for env_enabled, blacklisted, row_applied, superblock in itertools.product(
        (False, True),
        (False, True),
        (False, True),
        (0, 7),
    ):
        with monkeypatch.context() as case_monkeypatch:
            _events, command = _run_ttir_to_npubin(
                compiler_module,
                case_monkeypatch,
                auto_map_enabled=env_enabled,
                has_blacklist_op=blacklisted,
                row_coalescing_applied=row_applied,
                superblock_factor=superblock,
                common_options=common_options,
                enable_bishengir_simt_optimization=17,
                resolved_simt_stack_limit=64,
                shared_mem_dynamic_size=4096,
                enable_simt_reorder_instruction=True,
                disable_fma=True,
            )

        second_injection = env_enabled and not blacklisted and not row_applied
        case = f"E={env_enabled}, B={blacklisted}, R={row_applied}, superblock={superblock}"

        expected_options = [*common_options, *pure_simt_prefix]
        if second_injection:
            expected_options.append(auto_blockify_flag)
            if superblock > 0:
                expected_options.append(f"--super-block-factor={superblock}")

        # The compiler and launcher now agree on the single policy/safety gate.
        assert command[0] == "/fake/bisheng", case
        assert Path(command[1]).name == "kernel.ttir.mlir", case
        assert command[2:-2] == expected_options, case
        assert command[-2] == "-o", case
        assert Path(command[-1]).name == "kernel", case


def test_default_compile_mode_keeps_the_91095_layout_memory_gate_prepared(compiler_module):
    """The canonical default is portable and enables the A5 template gate."""

    a2_default = compiler_module.NPUOptions(arch="Ascend910B1")
    assert a2_default.compile_on_910_95 is False
    assert a2_default.compile_mode == "simd_simt_template"
    assert a2_default.is_pure_simt is False

    a5_default = compiler_module.NPUOptions(arch="Ascend910_9589")
    assert a5_default.compile_on_910_95 is True
    assert a5_default.compile_mode == "simd_simt_template"
    assert a5_default.is_pure_simt is False

    with pytest.warns(FutureWarning, match="compile_on_910_95"):
        ignored_legacy_value = compiler_module.NPUOptions(
            arch="Ascend910_9589",
            compile_on_910_95=False,
        )
    assert ignored_legacy_value.compile_on_910_95 is True

    canonical = compiler_module.NPUOptions(
        arch="Ascend910_9589",
        compile_mode="simd_simt_template",
    )
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        alias = compiler_module.NPUOptions(
            arch="Ascend910_9589",
            compile_mode="unstructured_in_simt",
        )
    assert not caught
    assert alias.compile_mode == canonical.compile_mode == "simd_simt_template"
    assert alias.hash() == canonical.hash()

    with pytest.raises(ValueError, match=r"invalid compile_mode='simt_template'"):
        compiler_module.NPUOptions(arch="Ascend910_9589", compile_mode="simt_template")

    explicit_simd = compiler_module.NPUOptions(arch="Ascend910_9589", compile_mode="simd")
    assert explicit_simd.compile_mode == "simd"
    assert explicit_simd.is_pure_simt is False

    explicit_template = compiler_module.NPUOptions(
        arch="Ascend910_9589",
        compile_mode="simd_simt_template",
    )
    assert explicit_template.compile_mode == "simd_simt_template"
    assert explicit_template.is_pure_simt is False

    explicit_only = compiler_module.NPUOptions(arch="Ascend910_9589", compile_mode="simt_only")
    assert explicit_only.compile_mode == "simt_only"
    assert explicit_only.is_pure_simt is True

    assert "force_simt_only" not in compiler_module.NPUOptions.__dataclass_fields__
    assert "force_simt_template" not in compiler_module.NPUOptions.__dataclass_fields__


def _program_grid_transform(order, axis, factor, logical_extent, *, persistent=False):
    return {
        "order": order,
        "kind": "ceil_div",
        "axis": axis,
        "factor": factor,
        "logical_extent": logical_extent,
        "persistent_coverage": persistent,
        "grid_stride_abi_verified": persistent,
    }


def _install_program_grid_attr_shim(monkeypatch, compiler_module):

    def get_int_attr(module, name):
        return module.attrs.get(name)

    def get_program_grid_transforms(module):
        return module.attrs.get("hacc.program_grid_transforms")

    def get_program_grid_specialization(module):
        return module.attrs.get("hacc.grid_specialization")

    def get_program_mapping_scalar_specialization(module):
        return module.attrs.get("hacc.program_mapping_scalar_specialization")

    def set_program_grid_specialization(module, version, grid_0, grid_1, grid_2, rule_mask):
        module.attrs["hacc.grid_specialization"] = {
            "version": version,
            "grid": [grid_0, grid_1, grid_2],
            "rule_mask": rule_mask,
        }

    def clear_program_grid_specialization(module):
        module.attrs.pop("hacc.grid_specialization", None)

    def set_program_mapping_scalar_specialization(module, version, arguments):
        items = []
        for argument in arguments:
            if len(argument) == 2:
                index, value = argument
                items.append({"index": index, "value": value})
            else:
                index, name, value = argument
                items.append({"index": index, "name": name, "value": value})
        module.attrs["hacc.program_mapping_scalar_specialization"] = {
            "version": version,
            "arguments": items,
        }

    def clear_program_mapping_scalar_specialization(module):
        module.attrs.pop("hacc.program_mapping_scalar_specialization", None)

    def remove_attr(module, name):
        module.attrs.pop(name, None)

    monkeypatch.setattr(
        compiler_module,
        "ascend",
        SimpleNamespace(ir=SimpleNamespace(
            get_int_attr=get_int_attr,
            get_program_grid_transforms=get_program_grid_transforms,
            get_program_grid_specialization=get_program_grid_specialization,
            get_program_mapping_scalar_specialization=(get_program_mapping_scalar_specialization),
            set_program_grid_specialization=set_program_grid_specialization,
            set_program_mapping_scalar_specialization=(set_program_mapping_scalar_specialization),
            clear_program_grid_specialization=clear_program_grid_specialization,
            clear_program_mapping_scalar_specialization=(clear_program_mapping_scalar_specialization),
            remove_attr=remove_attr,
        )),
    )


def test_export_program_grid_metadata_is_versioned_and_strips_module_attrs(
    compiler_module,
    monkeypatch,
):
    _install_program_grid_attr_shim(monkeypatch, compiler_module)
    module = SimpleNamespace(attrs={
        "hacc.program_grid_transforms": {
            "version": 1,
            "transforms": [_program_grid_transform(0, 1, 4, 33)],
        },
    })
    metadata = {}

    compiler_module._export_program_grid_metadata(module, metadata)

    assert module.attrs == {}
    assert metadata["program_grid_transform_schema_version"] == 1
    assert metadata["program_grid_transforms"] == {
        "version": 1,
        "transforms": [_program_grid_transform(0, 1, 4, 33)],
    }
    assert metadata["program_grid_transforms_cache_key"] == (
        '{"transforms":[{"axis":1,"factor":4,"grid_stride_abi_verified":false,'
        '"kind":"ceil_div","logical_extent":33,"order":0,"persistent_coverage":false}],'
        '"version":1}')
    assert metadata["coalesce_factor"] == 1
    assert metadata["coalesce_axis"] == -1


@pytest.mark.parametrize(
    "transforms",
    [
        [_program_grid_transform(0, 0, 2, 65)],
        [
            _program_grid_transform(0, 0, 2, 65),
            _program_grid_transform(1, 0, 3, 65),
        ],
        [
            _program_grid_transform(0, 0, 4, 17),
            _program_grid_transform(1, 1, 3, 10),
        ],
    ],
)
def test_export_program_grid_metadata_preserves_composable_transform_order(
    compiler_module,
    monkeypatch,
    transforms,
):
    _install_program_grid_attr_shim(monkeypatch, compiler_module)
    module = SimpleNamespace(attrs={
        "hacc.program_grid_transforms": {"version": 1, "transforms": transforms},
    })
    metadata = {}

    compiler_module._export_program_grid_metadata(module, metadata)

    assert module.attrs == {}
    assert metadata["program_grid_transforms"] == {
        "version": 1,
        "transforms": transforms,
    }
    assert metadata["program_grid_transforms_cache_key"] == (compiler_module.canonical_program_grid_transforms_json(
        metadata["program_grid_transforms"], ))


def test_program_grid_and_legacy_coalesce_metadata_are_mutually_exclusive(
    compiler_module,
    monkeypatch,
):
    _install_program_grid_attr_shim(monkeypatch, compiler_module)
    module = SimpleNamespace(
        attrs={
            "hacc.program_grid_transforms": {
                "version": 1,
                "transforms": [_program_grid_transform(0, 0, 2, 8)],
            },
            "hacc.coalesce_factor": 2,
            "hacc.coalesce_axis": 0,
            "hacc.coalesce_grid_ceil_div": 1,
        })

    with pytest.raises(RuntimeError, match="conflicts with legacy"):
        compiler_module._export_program_grid_metadata(module, {})
    # The rejection happens only after all hacc launch attrs were removed, so
    # an unrecognized attr cannot leak to a downstream compiler.
    assert module.attrs == {}


def test_legacy_metadata_defaults_remain_unchanged_without_new_schema(
    compiler_module,
    monkeypatch,
):
    _install_program_grid_attr_shim(monkeypatch, compiler_module)
    module = SimpleNamespace(attrs={})
    metadata = {}

    compiler_module._export_program_grid_metadata(module, metadata)

    assert metadata == {
        "program_grid_transforms": None,
        "program_grid_transform_schema_version": 0,
        "program_grid_transforms_cache_key": "legacy",
        "coalesce_factor": 1,
        "coalesce_axis": -1,
        "coalesce_grid_ceil_div": False,
        "row_coalescing_applied": False,
    }


def test_npu_options_hash_includes_program_grid_schema_and_precision_policy(
    compiler_module,
    monkeypatch,
):
    current = compiler_module.NPUOptions(arch="Ascend910B1")
    relaxed = compiler_module.NPUOptions(
        arch="Ascend910B1",
        precision_policy="relaxed",
    )
    assert current.hash() != relaxed.hash()

    # A future wheel that bumps the schema constant cannot collide with this
    # wheel's cache entries even if all other compile options are unchanged.
    with monkeypatch.context() as schema_v2:
        schema_v2.setattr(compiler_module, "PROGRAM_GRID_TRANSFORMS_VERSION", 2)
        next_schema = compiler_module.NPUOptions(
            arch="Ascend910B1",
            program_grid_transform_schema_version=2,
        )
    assert current.hash() != next_schema.hash()


def _program_grid_specialization(*, grid=(8, 65, 1), rule_mask=512):
    return {"version": 1, "grid": list(grid), "rule_mask": rule_mask}


def _program_mapping_scalar_specialization(*, arguments=((2, 64), (10, 4))):
    return {
        "version": 1,
        "arguments": [{"index": index, "value": value} for index, value in arguments],
    }


def test_program_grid_specialization_options_preserve_legacy_state_when_disabled(compiler_module, ):
    legacy = compiler_module.NPUOptions(arch="Ascend910B1")
    explicit_disabled = compiler_module.NPUOptions(arch="Ascend910B1", program_mapping_rule_mask=0)
    enabled = compiler_module.NPUOptions(
        arch="Ascend910B1",
        program_mapping_rule_mask=512,
        program_grid_specialization=_program_grid_specialization(),
        program_mapping_scalar_specialization=(_program_mapping_scalar_specialization()),
    )

    for options in (legacy, explicit_disabled):
        assert "program_mapping_rule_mask" not in options.__dict__
        assert "program_grid_specialization" not in options.__dict__
        assert "program_mapping_scalar_specialization" not in options.__dict__
    assert legacy.hash() == explicit_disabled.hash()
    assert enabled.__dict__["program_mapping_rule_mask"] == 512
    assert enabled.__dict__["program_mapping_resource_model_schema_version"] == 2
    assert enabled.__dict__["program_grid_specialization"] == _program_grid_specialization()
    assert enabled.__dict__["program_mapping_scalar_specialization"] == (_program_mapping_scalar_specialization())
    assert enabled.hash() != legacy.hash()


@pytest.mark.parametrize(
    "kwargs",
    [
        {"program_mapping_rule_mask": 1},
        {"program_grid_specialization": _program_grid_specialization()},
        {
            "program_mapping_rule_mask": 512,
            "program_grid_specialization": _program_grid_specialization(rule_mask=1024),
        },
        {
            "program_mapping_rule_mask": 512,
            "program_mapping_scalar_specialization": (_program_mapping_scalar_specialization()),
        },
    ],
)
def test_npu_options_reject_inconsistent_program_grid_specialization(
    compiler_module,
    kwargs,
):
    with pytest.raises(ValueError, match="program-grid|program_grid|program_mapping"):
        compiler_module.NPUOptions(arch="Ascend910B1", **kwargs)


def test_backend_prepares_reproducible_grid_before_cache_and_rejects_stale_input(compiler_module, ):
    backend = compiler_module.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend910B1"))
    calls = []

    def stable_grid(bound):
        calls.append(bound["extent"])
        return (bound["extent"], 16)

    prepared = backend.prepare_program_grid_specialization(stable_grid, {"extent": 65},
                                                           {"program_mapping_rule_mask": 512})
    assert prepared == (
        (65, 16, 1),
        {"program_grid_specialization": _program_grid_specialization(grid=(65, 16, 1))},
    )
    assert calls == [65, 65]

    unstable = iter(((8, 1), (9, 1)))
    with pytest.raises(RuntimeError, match="not reproducible"):
        backend.prepare_program_grid_specialization(lambda _bound: next(unstable), {},
                                                    {"program_mapping_rule_mask": 512})
    with pytest.raises(RuntimeError, match="JIT resolves"):
        backend.prepare_program_grid_specialization(
            (8, 1), {}, {
                "program_mapping_rule_mask": 512,
                "program_grid_specialization": _program_grid_specialization(grid=(8, 1, 1)),
            })


def test_backend_prepares_cache_keyed_runtime_scalar_specialization(compiler_module, ):
    backend = compiler_module.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend910B1"))
    params = [
        SimpleNamespace(name="ptr", is_constexpr=False),
        SimpleNamespace(name="stride", is_constexpr=False),
        SimpleNamespace(name="BLOCK", is_constexpr=True),
        SimpleNamespace(name="count", is_constexpr=False),
    ]
    prepared = backend.prepare_program_mapping_specialization(
        (8, 65),
        {"ptr": object(), "stride": 64, "BLOCK": 128, "count": 4},
        {"program_mapping_rule_mask": 512},
        params,
    )

    assert prepared == (
        (8, 65, 1),
        {
            "program_grid_specialization":
            _program_grid_specialization(),
            "program_mapping_scalar_specialization": ({
                "version":
                2,
                "arguments": [
                    {"index": 1, "name": "stride", "value": 64},
                    {"index": 2, "name": "count", "value": 4},
                ],
            }),
        },
    )


def test_backend_finalizes_python_launch_grid_once_and_checks_original_extent(
    compiler_module,
    monkeypatch,
):
    backend = compiler_module.AscendBackend(SimpleNamespace(backend="npu", arch="Ascend910B1"))
    monkeypatch.setattr(
        compiler_module,
        "NPUUtils",
        lambda: SimpleNamespace(
            get_aivector_core_num=lambda: 56,
            get_aicore_num=lambda: 28,
        ),
    )
    metadata = SimpleNamespace(
        program_grid_specialization=_program_grid_specialization(grid=(512, 16, 1), rule_mask=2560),
        program_grid_transforms={
            "version": 1,
            "transforms": [
                _program_grid_transform(0, 1, 8, 16),
                _program_grid_transform(1, 0, 2, 512, persistent=True),
            ],
        },
        mix_mode="aiv",
    )

    # IAT first gives [512, 2, 1], PTSM then gives [256, 2, 1], and its
    # verified axis-0 coverage permits the vector-core cap to [28, 2, 1].
    assert backend.finalize_program_mapping_launch_grid((512, 16, 1), metadata) == (28, 2, 1)

    with pytest.raises(RuntimeError, match="does not match"):
        backend.finalize_program_mapping_launch_grid((256, 16, 1), metadata)


def test_compiler_injects_and_exports_grid_specialization_without_attr_leak(
    compiler_module,
    monkeypatch,
):
    _install_program_grid_attr_shim(monkeypatch, compiler_module)
    module = SimpleNamespace(attrs={})
    metadata = {}
    option = SimpleNamespace(
        program_mapping_rule_mask=512,
        program_grid_specialization=_program_grid_specialization(),
        program_mapping_scalar_specialization=(_program_mapping_scalar_specialization()),
    )

    compiler_module._inject_program_grid_specialization(module, metadata, option)
    assert module.attrs == {
        "hacc.grid_specialization": _program_grid_specialization(),
        "hacc.program_mapping_scalar_specialization": (_program_mapping_scalar_specialization()),
    }
    assert metadata["program_grid_specialization"] == _program_grid_specialization()
    assert metadata["program_grid_specialization_cache_key"] == ('{"grid":[8,65,1],"rule_mask":512,"version":1}')
    assert metadata["program_mapping_scalar_specialization"] == (_program_mapping_scalar_specialization())

    compiler_module._export_program_grid_metadata(module, metadata)
    assert module.attrs == {}
    assert metadata["program_grid_specialization"] == _program_grid_specialization()
    assert metadata["program_grid_specialization_cache_key"] == ('{"grid":[8,65,1],"rule_mask":512,"version":1}')
    assert metadata["program_mapping_scalar_specialization"] == (_program_mapping_scalar_specialization())


def test_aot_program_mapping_without_fixed_grid_stays_attr_free(
    compiler_module,
    monkeypatch,
):
    _install_program_grid_attr_shim(monkeypatch, compiler_module)
    module = SimpleNamespace(attrs={})
    metadata = {}
    option = SimpleNamespace(program_mapping_rule_mask=512, program_grid_specialization=None)

    assert compiler_module._inject_program_grid_specialization(module, metadata, option) is None
    assert module.attrs == {}
    assert metadata == {}
