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
from pathlib import Path
from types import SimpleNamespace

import pytest

pytestmark = pytest.mark.backend("none")

_UNSET = object()


def _stub_graph_ub_budget_bytes_for_arch(arch):
    """Mirror the documented architecture table for the compiler import shim.

    The table itself is covered by the source-loaded backend-utils test.  This
    local shim keeps this compiler-only contract independent of the installed
    Ascend package while retaining meaningful option-normalization assertions.
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

    def return_false(*_args, **_kwargs):
        return False

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
            "_get_llvm_path",
            "_get_mlir_path",
            "_get_triton_adapter_opt_path",
            "_get_triton_mlir_opt_path",
            "_get_triton_opt_path",
            "_get_bishengir_opt_path",
    ):
        setattr(utils_stub, name, lambda *_args, **_kwargs: "")
    utils_stub._get_npucompiler_path = lambda *_args, **_kwargs: ("", {})
    utils_stub._get_auto_blockify_blacklist_reasons = lambda *_args, **_kwargs: []
    utils_stub._warn_auto_blockify_disabled = lambda *_args, **_kwargs: None
    utils_stub.downgrade_llir = lambda llir: llir
    utils_stub.get_cann_version_file_hash = lambda: ""
    utils_stub.graph_ub_budget_bytes_for_arch = _stub_graph_ub_budget_bytes_for_arch
    utils_stub.is_compile_on_910_95 = lambda: False

    class UnusedNPUUtils:
        pass

    driver_stub = types.ModuleType(driver_name)
    driver_stub.NPUUtils = UnusedNPUUtils

    cache_stub = types.ModuleType(cache_name)
    cache_stub._base32 = lambda value: str(value)
    cache_stub.get_dump_manager = lambda *_args, **_kwargs: SimpleNamespace(cache_dir="", put=lambda *_args, **_kwargs:
                                                                            None)

    previous_utils = sys.modules.get(utils_name)
    previous_driver = sys.modules.get(driver_name)
    previous_cache = sys.modules.get(cache_name)
    sys.modules[utils_name] = utils_stub
    sys.modules[driver_name] = driver_stub
    sys.modules[cache_name] = cache_stub
    sys.modules.pop(module_name, None)
    try:
        spec = importlib.util.spec_from_file_location(module_name, compiler_path)
        module = importlib.util.module_from_spec(spec)
        assert spec is not None and spec.loader is not None
        sys.modules[module_name] = module
        spec.loader.exec_module(module)
    finally:
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
    return module


def _parse_options(compiler, arch, opts=None):
    backend = compiler.AscendBackend(SimpleNamespace(backend="npu", arch=arch))
    return backend.parse_options({} if opts is None else opts)


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
    assert options.graph_optimize_ub_capacity_bytes == expected_capacity


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
    force_simt_only,
    enable_auto_blockify=None,
    superblock_factor=0,
    enable_bishengir_simt_optimization=0,
    simt_stack_limit=0,
    shared_mem_dynamic_size=None,
    enable_simt_reorder_instruction=False,
    disable_fma=False,
):
    return SimpleNamespace(
        force_simt_only=force_simt_only,
        num_warps=4,
        warp_size=32,
        enable_bishengir_simt_optimization=enable_bishengir_simt_optimization,
        simt_stack_limit=simt_stack_limit,
        shared_mem_dynamic_size=shared_mem_dynamic_size,
        enable_simt_reorder_instruction=enable_simt_reorder_instruction,
        disable_fma=disable_fma,
        enable_auto_blockify=enable_auto_blockify,
        superblock_factor=superblock_factor,
    )


def _run_ttir_to_npubin(
    compiler,
    monkeypatch,
    *,
    force_simt_only=True,
    auto_map_enabled=False,
    enable_auto_blockify=None,
    has_blacklist_op=False,
    row_coalescing_applied=False,
    superblock_factor=0,
    common_options=(),
    bisheng_options=None,
    enable_bishengir_simt_optimization=0,
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
            "bisheng_options": bisheng_options,
            "has_auto_blockify_blacklist_op": has_blacklist_op,
            "row_coalescing_applied": row_coalescing_applied,
        }

    def export_coalesce_metadata(_mod, _metadata, *, require_row_contract=False):
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
    monkeypatch.setattr(compiler, "_export_coalesce_metadata", export_coalesce_metadata)
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
    # StackSize precedence is covered by test_compiler.py.  Keep this argv
    # matrix independent of the host torch_npu configuration while verifying
    # that ttir_to_npubin uses the resolver rather than the legacy option.
    monkeypatch.setattr(
        compiler,
        "get_simt_stack_limit",
        lambda: resolved_simt_stack_limit,
    )
    monkeypatch.setattr(compiler.subprocess, "run", run_bisheng)

    result = compiler.ttir_to_npubin(
        module,
        {},
        _make_opt(
            force_simt_only=force_simt_only,
            enable_auto_blockify=enable_auto_blockify,
            superblock_factor=superblock_factor,
            enable_bishengir_simt_optimization=enable_bishengir_simt_optimization,
            shared_mem_dynamic_size=shared_mem_dynamic_size,
            enable_simt_reorder_instruction=enable_simt_reorder_instruction,
            disable_fma=disable_fma,
        ),
    )
    assert result == b"npubin"
    assert len(commands) == 1
    return events, commands[0]


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


def test_ttir_to_npubin_exports_make_ttir_row_contract_only_for_pure_simt(compiler_module, monkeypatch):
    events, _command = _run_ttir_to_npubin(
        compiler_module,
        monkeypatch,
        force_simt_only=True,
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
            force_simt_only=False,
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


def test_make_ttir_passes_force_simt_only_to_graph_optimize(compiler_module, monkeypatch):
    options = SimpleNamespace(
        enable_graph_optimize=True,
        graph_optimize_rule_mask=8,
        graph_optimize_max_rewrites_per_function=17,
        graph_optimize_ub_capacity_bytes=4096,
        graph_optimize_emit_remarks=True,
        force_simt_only=True,
        debug=False,
    )

    events, graph_calls = _run_make_ttir_with_recorded_graph_options(compiler_module, monkeypatch, options)

    assert graph_calls == [{
        "rule_mask": 8,
        "max_rewrites_per_function": 17,
        "ub_capacity_bytes": 4096,
        "emit_remarks": True,
        "force_simt_only": True,
    }]
    assert events[-1] == "run_row"


@pytest.mark.parametrize(
    ("requested_capacity", "expected_capacity"),
    (
        (None, 96 * 1024),
        (0, 0),
        (4096, 4096),
        (96 * 1024 + 1, 96 * 1024),
    ),
)
def test_make_ttir_forwards_normalized_graph_ub_budget(compiler_module, monkeypatch, requested_capacity,
                                                       expected_capacity):
    options = compiler_module.NPUOptions(
        arch="Ascend910B1",
        graph_optimize_rule_mask=8,
        graph_optimize_max_rewrites_per_function=17,
        graph_optimize_ub_capacity_bytes=requested_capacity,
        graph_optimize_emit_remarks=True,
        force_simt_only=True,
    )

    events, graph_calls = _run_make_ttir_with_recorded_graph_options(compiler_module, monkeypatch, options)

    assert graph_calls == [{
        "rule_mask": 8,
        "max_rewrites_per_function": 17,
        "ub_capacity_bytes": expected_capacity,
        "emit_remarks": True,
        "force_simt_only": True,
    }]
    assert events[-1] == "run_row"


def test_ttir_to_npubin_auto_blockify_argv_matrix(compiler_module, monkeypatch):
    """Keep the complete 895 pure-SIMT argv, including duplicate flag order.

    E: TRITON_ALL_BLOCKS_PARALLEL; O: user option; B: blacklist; R: Row
    coalescing result.  O is intentionally tri-state because ``None`` is the
    default contract rather than an explicit user choice.
    """
    common_options = ["--common-before-pure-simt", "--common-after-pure-simt"]
    bisheng_options = "--preserve-bisheng-option-order"
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

    for (
            env_enabled,
            user_option,
            blacklisted,
            row_applied,
            superblock,
            case_bisheng_options,
    ) in itertools.product(
        (False, True),
        (None, False, True),
        (False, True),
        (False, True),
        (0, 7),
        (None, bisheng_options),
    ):
        with monkeypatch.context() as case_monkeypatch:
            _events, command = _run_ttir_to_npubin(
                compiler_module,
                case_monkeypatch,
                auto_map_enabled=env_enabled,
                enable_auto_blockify=user_option,
                has_blacklist_op=blacklisted,
                row_coalescing_applied=row_applied,
                superblock_factor=superblock,
                common_options=common_options,
                bisheng_options=case_bisheng_options,
                enable_bishengir_simt_optimization=17,
                resolved_simt_stack_limit=64,
                shared_mem_dynamic_size=4096,
                enable_simt_reorder_instruction=True,
                disable_fma=True,
            )

        first_injection = (env_enabled and
                           (user_option is None or user_option)) or (not env_enabled and bool(user_option))
        second_injection = env_enabled and not blacklisted and not row_applied
        case = (f"E={env_enabled}, O={user_option}, B={blacklisted}, "
                f"R={row_applied}, superblock={superblock}, "
                f"bisheng_options={case_bisheng_options!r}")

        expected_options = [*common_options, *pure_simt_prefix]
        if first_injection:
            expected_options.append(auto_blockify_flag)
        if case_bisheng_options is not None:
            expected_options.append(f"--append-bisheng-options={case_bisheng_options}")
        if second_injection:
            expected_options.append(auto_blockify_flag)
            if superblock > 0:
                expected_options.append(f"--super-block-factor={superblock}")

        # Keep the source/output envelope as well as every option.  In
        # particular, two copies of the auto-blockify flag must remain in their
        # historical insertion slots: adjacent when no Bisheng option exists,
        # and on opposite sides of append-bisheng-options when it does.
        assert command[0] == "/fake/bisheng", case
        assert Path(command[1]).name == "kernel.ttir.mlir", case
        assert command[2:-2] == expected_options, case
        assert command[-2] == "-o", case
        assert Path(command[-1]).name == "kernel", case


def test_default_compile_mode_keeps_the_91095_layout_memory_gate_prepared(compiler_module, ):
    """The normal compiler default supplies the second half of the T2L gate.

    Axis/Chunk/SLS must remain controlled by the original
    ``compile_on_910_95 && force_simt_template`` predicate.  The first half
    comes only from real hardware detection; this source-level contract makes
    sure the normal 91095 path does not accidentally lose its historical
    ``unstructured_in_simt``/``force_simt_template`` default while tests run
    on a non-91095 host.
    """

    default_options = compiler_module.NPUOptions()
    assert default_options.compile_mode == "unstructured_in_simt"
    assert default_options.force_simt_template is True
    assert default_options.force_simt_only is False
    assert default_options.graph_optimize_ub_capacity_bytes == 0

    simd_options = compiler_module.NPUOptions(compile_mode="simd")
    assert simd_options.force_simt_template is False
    assert simd_options.force_simt_only is False
