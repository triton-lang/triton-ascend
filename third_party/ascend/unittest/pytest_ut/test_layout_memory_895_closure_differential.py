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
"""895 closure differential tests for layout / memory-access migration.

The migration deliberately keeps the original compiler, launcher, and
autotuner contracts.  This test obtains those three 895 sources with
``git show`` and executes only the relevant AST closure with local fakes.  It
does not import a baseline wheel or depend on a sibling checkout.  A normal
checkout which no longer contains the baseline object simply skips this
optional historical differential test; release validation sets
``TRITON_REQUIRE_895_DIFFERENTIAL=1`` so a missing baseline is a failure
rather than a misleading green skip.
"""

import ast
import copy
import itertools
import os
import subprocess
import tempfile
from pathlib import Path
from types import SimpleNamespace

import pytest

_BASELINE_COMMIT = "895c5fbe2b0e69349b76388e65fd8c3e79703bb9"
_REQUIRE_BASELINE_ENV = "TRITON_REQUIRE_895_DIFFERENTIAL"
_SOURCE_PATHS = {
    "compiler": "third_party/ascend/backend/compiler.py",
    "driver": "third_party/ascend/backend/driver.py",
    "autotuner": "third_party/ascend/backend/runtime/autotuner.py",
}


def _repo_root():
    return Path(__file__).resolve().parents[4]


def _source_text(path):
    # compiler.py starts with a UTF-8 BOM in some revisions.  Python's normal
    # source loader accepts it, whereas ast.parse receives the decoded marker.
    return path.read_text(encoding="utf-8-sig")


def _baseline_is_required():
    return os.environ.get(_REQUIRE_BASELINE_ENV, "").lower() in {
        "1",
        "true",
        "yes",
    }


@pytest.fixture(scope="module")
def source_pairs():
    """Return (895, target) source text for every closure under test.

    Keep the historical object optional for a normal shallow checkout.  The
    release-validation environment explicitly upgrades an unavailable object
    to failure, so the differential cannot silently disappear from the
    migration acceptance evidence.
    """
    root = _repo_root()
    pairs = {}
    for name, relative_path in _SOURCE_PATHS.items():
        try:
            result = subprocess.run(
                ["git", "show", f"{_BASELINE_COMMIT}:{relative_path}"],
                cwd=root,
                text=True,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                check=False,
            )
        except OSError as error:
            message = f"cannot invoke git for 895 closure differential: {error}"
            if _baseline_is_required():
                pytest.fail(message)
            pytest.skip(message)
        if result.returncode != 0:
            message = ("895 closure differential baseline is unavailable: "
                       f"{result.stderr.strip() or _BASELINE_COMMIT}")
            if _baseline_is_required():
                pytest.fail(message)
            pytest.skip(message)
        pairs[name] = (result.stdout.lstrip("\ufeff"), _source_text(root / relative_path))
    return pairs


def _top_level_functions(source, *names):
    tree = ast.parse(source)
    by_name = {node.name: node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))}
    missing = set(names) - set(by_name)
    assert not missing, f"source no longer defines closure functions: {sorted(missing)}"
    return [copy.deepcopy(by_name[name]) for name in names]


def _exec_functions(source, names, namespace):
    module = ast.Module(body=_top_level_functions(source, *names), type_ignores=[])
    ast.fix_missing_locations(module)
    exec(compile(module, "<layout-memory-closure>", "exec"), namespace)
    return namespace


def _normalised_function_ast(source, name):
    function = _top_level_functions(source, name)[0]
    return ast.dump(function, include_attributes=False)


def _load_compiler_closure(source):
    # The selected functions only need these imported names.  Keeping a tiny
    # namespace lets the test execute the exact source closure without loading
    # the installed compiler package or an NPU toolchain.
    subprocess_proxy = SimpleNamespace(CalledProcessError=subprocess.CalledProcessError, )
    namespace = {
        "os": os,
        "tempfile": tempfile,
        "Path": Path,
        "subprocess": subprocess_proxy,
    }
    _exec_functions(
        source,
        ("_get_then_remove_rc", "_export_coalesce_metadata", "ttir_to_npubin"),
        namespace,
    )
    return namespace


def test_895_compiler_closure_ast_is_identical_outside_row_migration(source_pairs):
    """Keep unrelated helpers stable and derive the block blacklist internally."""

    baseline_source, target_source = source_pairs["compiler"]
    for name in (
            "_get_then_remove_rc",
            "get_common_bishengir_compile_options",
    ):
        baseline = _normalised_function_ast(baseline_source, name)
        target = _normalised_function_ast(target_source, name)
        assert target == baseline, name

    baseline_metadata = _normalised_function_ast(baseline_source, "_parse_ttir_metadata")
    target_metadata = _normalised_function_ast(target_source, "_parse_ttir_metadata")
    assert target_metadata != baseline_metadata
    assert "_get_auto_blockify_blacklist_reasons" in target_metadata
    assert "attr='get'" in baseline_metadata
    assert "attr='get'" not in target_metadata


class _FakeIrModule:

    def __init__(self, attrs=None):
        self.context = object()
        self.attrs = dict(attrs or {})
        self.stringify_calls = 0

    def __str__(self):
        self.stringify_calls += 1
        return "module {}"


class _FakePassManager:

    def __init__(self):
        self.enable_debug_calls = 0
        self.run_calls = []

    def enable_debug(self):
        self.enable_debug_calls += 1

    def run(self, _module, *pipeline_name):
        self.run_calls.append(pipeline_name)


def _row_attrs(row_applied):
    if not row_applied:
        return {}
    return {
        "hacc.coalesce_factor": 4,
        "hacc.coalesce_axis": 2,
        "hacc.coalesce_grid_ceil_div": 1,
    }


def _make_opt(
    *,
    superblock_factor,
):
    return SimpleNamespace(
        is_pure_simt=True,
        num_warps=4,
        warp_size=32,
        enable_bishengir_simt_optimization=17,
        simt_stack_limit=64,
        shared_mem_dynamic_size=4096,
        enable_simt_reorder_instruction=True,
        disable_fma=True,
        superblock_factor=superblock_factor,
    )


def _run_ttir_to_npubin(
    closure,
    *,
    env_enabled,
    blacklisted,
    row_applied,
    superblock_factor,
):
    """Run the historical pure-SIMT tail with a fake pass manager/compiler."""
    pass_manager = _FakePassManager()
    commands = []
    metadata_after_parse = []

    def get_int_attr(module, name):
        return module.attrs.get(name)

    def remove_attr(module, name):
        module.attrs.pop(name, None)

    def parse_ttir_metadata(_ttir, metadata):
        parsed = dict(metadata)
        parsed.update({
            "has_auto_blockify_blacklist_op": blacklisted,
            # _export_coalesce_metadata below replaces this with the row
            # pass result from the mock module attrs, just like production.
            "row_coalescing_applied": False,
        })
        # Both the 895 baseline and the compatibility-restored target read
        # this option. Keep it neutral for the argv differential below.
        if "bisheng_options" in closure["ttir_to_npubin"].__code__.co_consts:
            parsed["bisheng_options"] = None
        metadata_after_parse.append(parsed)
        return parsed

    def run_bisheng(command, **_kwargs):
        commands.append(list(command))
        bin_file = Path(command[command.index("-o") + 1])
        bin_file.with_name(f"{bin_file.name}.o").write_bytes(b"npubin")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    closure["ir"] = SimpleNamespace(pass_manager=lambda _context: pass_manager)
    closure["ascend"] = SimpleNamespace(
        ir=SimpleNamespace(get_int_attr=get_int_attr, remove_attr=remove_attr),
        passes=SimpleNamespace(ttir=SimpleNamespace(add_row_coalescing=lambda _pm: None), ),
    )
    closure["_parse_ttir_metadata"] = parse_ttir_metadata
    closure["get_common_bishengir_compile_options"] = lambda _metadata: [
        "--common-before-pure-simt",
        "--common-after-pure-simt",
    ]
    closure["_get_npucompiler_path"] = lambda: ("bishengir-compile", {})
    closure["_is_auto_map_parallel_blocks_enabled"] = lambda: env_enabled
    closure["get_simt_stack_limit"] = lambda _user_stack_limit=None: 64
    closure["subprocess"].run = run_bisheng

    result = closure["ttir_to_npubin"](
        _FakeIrModule(_row_attrs(row_applied)),
        {},
        _make_opt(superblock_factor=superblock_factor, ),
    )
    assert result == b"npubin"
    assert len(commands) == 1
    assert len(metadata_after_parse) == 1
    return pass_manager, commands[0], metadata_after_parse[0]


def _normalise_command(command):
    """Remove only temporary-directory entropy from an argv comparison."""
    return [
        command[0],
        Path(command[1]).name,
        *command[2:-2],
        command[-2],
        Path(command[-1]).name,
    ]


def _export_coalesce_metadata(closure, attrs):
    removed = []
    module = _FakeIrModule(attrs)

    def get_int_attr(current_module, name):
        return current_module.attrs.get(name)

    def remove_attr(current_module, name):
        removed.append(name)
        current_module.attrs.pop(name, None)

    closure["ascend"] = SimpleNamespace(ir=SimpleNamespace(get_int_attr=get_int_attr, remove_attr=remove_attr), )
    metadata = {}
    closure["_export_coalesce_metadata"](module, metadata)
    return metadata, module.attrs, removed


def test_895_pure_simt_bisheng_argv_matrix_after_row_make_ttir_migration(source_pairs):
    """The fixed-policy pure-SIMT argv cases retain no user block switch."""
    _baseline_source, target_source = source_pairs["compiler"]
    common_prefix = [
        "--common-before-pure-simt",
        "--common-after-pure-simt",
        "--enable-hivm-compile=false",
        "--enable-triton-ir-compile",
        "--pure-simt",
        "--enable-global-scratch-allocation",
        "--num-warps=4",
        "--threads-per-warp=32",
        "--enable-bishengir-simt-optimization=17",
        "--simt-stack-limit=64",
        "--shared-mem-dynamic-size=4096",
        "--enable-simt-reorder-instruction=true",
        "--disable-fma",
    ]
    cases = itertools.product((False, True),  # E: TRITON_ALL_BLOCKS_PARALLEL
                              (False, True),  # B: blacklist result
                              (False, True),  # R: Row pass result
                              (0, 7),  # superblock factor
                              )

    count = 0
    for env_enabled, blacklisted, row_applied, superblock in cases:
        target_closure = _load_compiler_closure(target_source)
        target_pm, target_command, _target_metadata = _run_ttir_to_npubin(
            target_closure,
            env_enabled=env_enabled,
            blacklisted=blacklisted,
            row_applied=row_applied,
            superblock_factor=superblock,
        )
        case = f"E={env_enabled}, B={blacklisted}, R={row_applied}, superblock={superblock}"

        # Keep bisheng_options neutral in this matrix so it verifies only the
        # pure-SIMT envelope and automatic block policy.
        expected_options = list(common_prefix)
        auto_blockify = env_enabled and not blacklisted and not row_applied
        if auto_blockify:
            expected_options.append("--enable-auto-blockify-loop")
            if superblock > 0:
                expected_options.append(f"--super-block-factor={superblock}")
        assert _normalise_command(target_command) == [
            "bishengir-compile",
            "kernel.ttir.mlir",
            *expected_options,
            "-o",
            "kernel",
        ], case

        # Row is applied by make_ttir's graph pass.  npubin must preserve all
        # compile arguments while no longer creating a Row pass manager.
        assert target_pm.run_calls == [], case
        count += 1

    assert count == 16


@pytest.mark.parametrize(
    "name,attrs,expected",
    (
        (
            "axis",
            {
                "hacc.coalesce_factor": 2,
                "hacc.coalesce_axis": 0,
                "hacc.coalesce_grid_ceil_div": 0,
            },
            {
                "coalesce_factor": 2,
                "coalesce_axis": 0,
                "coalesce_grid_ceil_div": False,
                "row_coalescing_applied": True,
            },
        ),
        (
            "chunk",
            {
                "hacc.coalesce_factor": 16,
                "hacc.coalesce_axis": 1,
                "hacc.coalesce_grid_ceil_div": 0,
            },
            {
                "coalesce_factor": 16,
                "coalesce_axis": 1,
                "coalesce_grid_ceil_div": False,
                "row_coalescing_applied": True,
            },
        ),
        (
            "row",
            {
                "hacc.coalesce_factor": 4,
                "hacc.coalesce_axis": 2,
                "hacc.coalesce_grid_ceil_div": 1,
            },
            {
                "coalesce_factor": 4,
                "coalesce_axis": 2,
                "coalesce_grid_ceil_div": True,
                "row_coalescing_applied": True,
            },
        ),
    ),
)
def test_895_coalesce_attrs_export_identically(name, attrs, expected, source_pairs):
    """Axis, Chunk, and Row attrs retain the exact 895 metadata handoff."""
    baseline_closure = _load_compiler_closure(source_pairs["compiler"][0])
    target_closure = _load_compiler_closure(source_pairs["compiler"][1])
    baseline = _export_coalesce_metadata(baseline_closure, attrs)
    target = _export_coalesce_metadata(target_closure, attrs)
    expected_removed = [
        "hacc.coalesce_factor",
        "hacc.coalesce_axis",
        "hacc.coalesce_grid_ceil_div",
    ]
    assert baseline == target == (expected, {}, expected_removed), name


class _FakeNPUUtils:
    npu_utils_mod = SimpleNamespace(__file__="")

    def get_aivector_core_num(self):
        return 40

    def get_aicore_num(self):
        return 20


def _load_make_launcher(source):
    state = {"auto_map_enabled": False}
    namespace = {
        "NPUUtils": _FakeNPUUtils,
        "_BASE_ARGS_FORMAT": "iiiKKOOOO",
        "_is_auto_map_parallel_blocks_enabled": lambda: state["auto_map_enabled"],
        "force_disable_ffts": lambda *_args: False,
        "is_ffts_supported": lambda _arch: True,
        "get_ascend_arch_from_env": lambda: "Ascend910B",
        "get_backend_func": lambda name, *_args: f"/* {name} */",
        "convert_sigtype_to_int": lambda _ty: 0,
        "generate_npu_header_src": lambda: "",
        "extract_device_print_code_from_cann": lambda: "",
    }
    _exec_functions(source, ("ty_to_cpp", "make_launcher"), namespace)
    return namespace["make_launcher"], state


def _make_metadata(*, factor, axis, ceil_div, blacklisted, row_applied):
    return SimpleNamespace(
        target=SimpleNamespace(arch="Ascend910B"),
        workspace_size=0,
        lock_init_value=0,
        lock_num=0,
        bs_task_type=0,
        mix_mode="aiv",
        shared=0,
        compile_on_910_95=False,
        parallel_mode="",
        # The baseline closure still reads this retired field; the target
        # closure reads is_pure_simt.  Keep both in this historical test mock.
        force_simt_only=False,
        is_pure_simt=False,
        debug=False,
        shared_mem_dynamic_size=221184,
        coalesce_factor=factor,
        coalesce_axis=axis,
        coalesce_grid_ceil_div=ceil_div,
        has_auto_blockify_blacklist_op=blacklisted,
        row_coalescing_applied=row_applied,
    )


def _launcher_paths(source):
    # make_launcher produces both the stable ABI path and local C++ packing
    # path.  The coalescing and auto-blockify fragments must be present in both.
    return source.split("static void _launch(", maxsplit=1)


def _coalescing_fragment(path):
    start = path.index("// coalescing: each program covers")
    end = path.index("uint32_t blockNum4Workspace", start)
    return path[start:end]


@pytest.mark.parametrize(
    "name,factor,axis,ceil_div,assignment,guard",
    (
        (
            "axis",
            2,
            0,
            False,
            "gridX = gridX / 2;",
            "ChunkCoalescing: grid[0] not divisible by coalesce_factor 2",
        ),
        (
            "chunk",
            16,
            1,
            False,
            "gridY = gridY / 16;",
            "ChunkCoalescing: grid[1] not divisible by coalesce_factor 16",
        ),
        (
            "row",
            4,
            2,
            True,
            "gridZ = (gridZ + 4 - 1) / 4;",
            None,
        ),
    ),
)
def test_895_launcher_coalescing_and_block_cap_closure(
    name,
    factor,
    axis,
    ceil_div,
    assignment,
    guard,
    source_pairs,
):
    """Compare both generated launcher paths for all E x B x R cap cases."""
    baseline_make_launcher, baseline_state = _load_make_launcher(source_pairs["driver"][0])
    target_make_launcher, target_state = _load_make_launcher(source_pairs["driver"][1])
    cap = "blockNum = std::min(blockNum, (uint32_t)40);"

    for env_enabled, blacklisted, row_applied in itertools.product(
        (False, True),
        (False, True),
        (False, True),
    ):
        baseline_state["auto_map_enabled"] = env_enabled
        target_state["auto_map_enabled"] = env_enabled
        baseline_src = baseline_make_launcher(
            constants={},
            signature={0: "*fp32", 1: "*fp32"},
            metadata=_make_metadata(
                factor=factor,
                axis=axis,
                ceil_div=ceil_div,
                blacklisted=blacklisted,
                row_applied=row_applied,
            ),
        )
        target_src = target_make_launcher(
            constants={},
            signature={0: "*fp32", 1: "*fp32"},
            metadata=_make_metadata(
                factor=factor,
                axis=axis,
                ceil_div=ceil_div,
                blacklisted=blacklisted,
                row_applied=row_applied,
            ),
        )
        case = f"{name}: E={env_enabled}, B={blacklisted}, R={row_applied}"
        baseline_paths = _launcher_paths(baseline_src)
        target_paths = _launcher_paths(target_src)
        assert len(baseline_paths) == len(target_paths) == 2, case
        expected_cap_count = 1 if env_enabled and not blacklisted else 0
        for baseline_path, target_path in zip(baseline_paths, target_paths):
            assert _coalescing_fragment(baseline_path) == _coalescing_fragment(target_path), case
            assert baseline_path.count(assignment) == target_path.count(assignment) == 1, case
            if guard is None:
                assert "ChunkCoalescing: grid[2] not divisible" not in baseline_path, case
                assert "ChunkCoalescing: grid[2] not divisible" not in target_path, case
            else:
                assert baseline_path.count(guard) == target_path.count(guard) == 1, case
            assert baseline_path.count(cap) == target_path.count(cap) == expected_cap_count, case


def test_895_launcher_all_emittable_coalescing_metadata_cases(source_pairs):
    """Differentially cover every metadata form emitted by the four passes.

    Axis can preserve a non-power-of-two split factor, Chunk is bounded to
    2/4/8/16, and Row derives H=2/4/8.  Each family may target x/y/z.  This
    checks the complete corresponding launcher fragment on both generated
    launch paths for all AutoBlockify cap inputs, without pretending that a
    910B4 smoke run entered the 91095 T2L gate.
    """

    baseline_make_launcher, baseline_state = _load_make_launcher(source_pairs["driver"][0])
    target_make_launcher, target_state = _load_make_launcher(source_pairs["driver"][1])
    grid_names = ("gridX", "gridY", "gridZ")
    cap = "blockNum = std::min(blockNum, (uint32_t)40);"
    # The factor sets deliberately mirror what each legacy pass can emit.
    families = (
        ("axis", (2, 3, 4, 8, 16), False),
        ("chunk", (2, 4, 8, 16), False),
        ("row", (2, 4, 8), True),
    )

    for family, factors, ceil_div in families:
        for factor, axis, env_enabled, blacklisted, row_applied in itertools.product(
                factors,
            (0, 1, 2),
            (False, True),
            (False, True),
            (False, True),
        ):
            baseline_state["auto_map_enabled"] = env_enabled
            target_state["auto_map_enabled"] = env_enabled
            metadata = _make_metadata(
                factor=factor,
                axis=axis,
                ceil_div=ceil_div,
                blacklisted=blacklisted,
                row_applied=row_applied,
            )
            baseline_src = baseline_make_launcher(constants={}, signature={0: "*fp32", 1: "*fp32"}, metadata=metadata)
            target_src = target_make_launcher(constants={}, signature={0: "*fp32", 1: "*fp32"}, metadata=metadata)
            grid = grid_names[axis]
            case = (f"{family}: H={factor}, axis={axis}, ceil={ceil_div}, "
                    f"E={env_enabled}, B={blacklisted}, R={row_applied}")
            expected_assignment = (f"{grid} = ({grid} + {factor} - 1) / {factor};"
                                   if ceil_div else f"{grid} = {grid} / {factor};")
            expected_cap_count = 1 if env_enabled and not blacklisted else 0
            for baseline_path, target_path in zip(_launcher_paths(baseline_src), _launcher_paths(target_src)):
                assert _coalescing_fragment(baseline_path) == _coalescing_fragment(target_path), case
                assert baseline_path.count(expected_assignment) == 1, case
                assert target_path.count(expected_assignment) == 1, case
                if ceil_div:
                    assert f"grid[{axis}] not divisible by coalesce_factor" not in baseline_path, case
                    assert f"grid[{axis}] not divisible by coalesce_factor" not in target_path, case
                else:
                    guard = (f"ChunkCoalescing: grid[{axis}] not divisible by "
                             f"coalesce_factor {factor}")
                    assert baseline_path.count(guard) == 1, case
                    assert target_path.count(guard) == 1, case
                assert baseline_path.count(cap) == expected_cap_count, case
                assert target_path.count(cap) == expected_cap_count, case


def test_895_launcher_keeps_mixed_simt_sls_marker_in_both_paths(source_pairs):
    """SLS still selects the original 910_95 mixed-SIMT launch ABI."""
    baseline_make_launcher, _baseline_state = _load_make_launcher(source_pairs["driver"][0])
    target_make_launcher, _target_state = _load_make_launcher(source_pairs["driver"][1])

    metadata = _make_metadata(
        factor=1,
        axis=-1,
        ceil_div=False,
        blacklisted=False,
        row_applied=False,
    )
    metadata.compile_on_910_95 = True
    metadata.parallel_mode = "mix_simd_simt"
    baseline_src = baseline_make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )
    target_src = target_make_launcher(
        constants={},
        signature={0: "*fp32", 1: "*fp32"},
        metadata=metadata,
    )

    baseline_paths = _launcher_paths(baseline_src)
    target_paths = _launcher_paths(target_src)

    for baseline_path, target_path in zip(baseline_paths, target_paths):
        # Baseline keeps the inlined RT launch ABI in each launch path.
        assert baseline_path.count("rtKernelLaunchWithFlagV2") == 1
        assert baseline_path.count("rtArgsEx_t argsInfo") == 1
        assert "cfgInfo.localMemorySize = 221184;" in baseline_path
        # Target routes both paths through the shared shim entry points; the
        # 221184 literal is carried at the cfg acquisition call site.
        assert target_path.count("cann_get_launch_kernel_cfg(221184)") == 1
        assert "cann_launch_kernel(func, blockNum" in target_path

    # The shim lives once in the header (first path) and carries both the
    # ACL (9.1.0+) and RT (<9.1.0) launch implementations.
    assert target_src.count("aclrtLaunchKernelWithHostArgs") == 1
    assert target_src.count("rtKernelLaunchWithFlagV2") == 1
    assert target_src.count("aclrtLaunchKernelAttr attrInfo") == 1
    assert target_src.count("rtArgsEx_t argsInfo") == 1


def _load_inject_grid_num_tiles(source):
    tree = ast.parse(source)
    candidates = [
        node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef) and node.name == "_inject_grid_num_tiles"
    ]
    assert len(candidates) == 1
    function = copy.deepcopy(candidates[0])
    # Execute the method as a stand-alone function.  Supply a local equivalent
    # of the backend's int marker so this verifies the source behavior rather
    # than importing whichever Triton wheel happens to be installed.
    function.decorator_list = []
    module = ast.Module(body=[function], type_ignores=[])
    ast.fix_missing_locations(module)
    internal_npu_option_int = type("_InternalNPUOptionInt", (int, ), {})
    namespace = {"_InternalNPUOptionInt": internal_npu_option_int}
    exec(compile(module, "<grid-num-tiles-closure>", "exec"), namespace)
    return namespace["_inject_grid_num_tiles"]


def test_895_grid_num_tiles_ast_closure_differential(source_pairs):
    """Static, callable, and explicit-grid paths remain byte-for-byte semantic peers."""
    baseline_inject = _load_inject_grid_num_tiles(source_pairs["autotuner"][0])
    target_inject = _load_inject_grid_num_tiles(source_pairs["autotuner"][1])
    dynamic_grid = lambda _meta: (2, 16)
    cases = (
        ("static", {"grid": (2, 16)}, {"grid": (2, 16), "grid_num_tiles": 16}),
        ("callable", {"grid": dynamic_grid}, {"grid": dynamic_grid}),
        (
            "explicit",
            {"grid": (2, 16), "grid_num_tiles": 99},
            {"grid": (2, 16), "grid_num_tiles": 99},
        ),
    )

    for name, initial, expected in cases:
        baseline_kwargs = dict(initial)
        target_kwargs = dict(initial)
        baseline_inject(baseline_kwargs)
        target_inject(target_kwargs)
        assert baseline_kwargs == expected, name
        assert target_kwargs == expected, name
        assert baseline_kwargs == target_kwargs, name


def _baseline_source(root, relative_path):
    """Read one required 895 source after ``source_pairs`` checked its object."""
    result = subprocess.run(
        ["git", "show", f"{_BASELINE_COMMIT}:{relative_path}"],
        cwd=root,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        check=False,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout


def _expected_relocated_legacy_core(name, baseline):
    """Apply only the documented mechanical relocation edits to an 895 file.

    The conversion fixtures show representative behavior; this check is the
    complementary all-path guard.  If an old matcher, bailout, or rewrite is
    altered in the moved core source, this equality fails before a narrow IR
    fixture can accidentally hide it.  Do not normalize arbitrary whitespace,
    identifiers, or control flow here: every allowed difference is listed.
    """
    if name == "axis":
        return baseline.replace(
            '#include "TritonToLinalg/StridedAxisCoalescing.h"',
            '#include "TritonToGraph/LegacyMemoryAccess/StridedAxisCoalescing.h"',
        )
    if name == "chunk":
        return baseline.replace(
            '#include "TritonToLinalg/ChunkCoalescing.h"',
            '#include "TritonToGraph/LegacyMemoryAccess/ChunkCoalescing.h"',
        )
    if name == "sls":
        return (baseline.replace(
            '#include "TritonToLinalg/StridedLoadStoreRewrite.h"',
            '#include "TritonToGraph/LegacyMemoryAccess/StridedLoadStoreRewrite.h"',
        ).replace(
            '#include "TritonToLinalg/ImplicitPermute.h"',
            '#include "TritonMemoryAccess/MemoryAccessTags.h"',
        ).replace(
            '#include "TritonToLinalg/MaskAnalysis.h"',
            '#include "TritonMemoryAccess/LoadStoreMaskAnalysis.h"',
        ).replace(
            "ImplicitPermute::ImplicitPermuteHandledTAG",
            "mlir::triton::memory_access::ImplicitPermuteHandledTAG",
        ))
    if name == "row":
        old_wrapper_start = baseline.index("\nnamespace {\n\nstruct RowCoalescingPass")
        old_wrapper_end = baseline.index("\n}  // namespace RowCoalescing", old_wrapper_start)
        return (
            baseline.replace(
                '#include "TritonToLinalg/RowCoalescing.h"',
                '#include "TritonToGraph/LegacyMemoryAccess/RowCoalescing.h"',
            ).replace('#include "mlir/Pass/Pass.h"\n', "").replace(baseline[old_wrapper_start:old_wrapper_end],
                                                                   "").replace("}  // namespace RowCoalescing",
                                                                               "} // namespace RowCoalescing")
            # The moved file keeps one explicit visual separator where the old
            # pass wrapper was removed; permit that one formatting-only delta.
            .replace(
                "\n\n} // namespace RowCoalescing",
                "\n\n\n} // namespace RowCoalescing",
            ))
    raise AssertionError(f"unknown legacy core: {name}")


def test_895_legacy_memory_access_core_sources_are_mechanical_relocations(source_pairs):
    """All four migrated core bodies remain 895-equivalent by construction.

    This intentionally compares full implementation sources rather than only
    selected positive examples.  The only permitted differences are include
    ownership, the shared tag namespace, and removal of Row's old pass wrapper;
    scheduling itself is separately exercised by compatibility-pass tests.
    """
    del source_pairs  # Fixture makes a missing 895 object fail in strict mode.
    root = _repo_root()
    sources = {
        "axis": (
            "third_party/ascend/lib/TritonToLinalg/StridedAxisCoalescing.cpp",
            "third_party/ascend/lib/TritonToGraph/LegacyMemoryAccess/StridedAxisCoalescing.cpp",
        ),
        "chunk": (
            "third_party/ascend/lib/TritonToLinalg/ChunkCoalescing.cpp",
            "third_party/ascend/lib/TritonToGraph/LegacyMemoryAccess/ChunkCoalescing.cpp",
        ),
        "sls": (
            "third_party/ascend/lib/TritonToLinalg/StridedLoadStoreRewrite.cpp",
            "third_party/ascend/lib/TritonToGraph/LegacyMemoryAccess/StridedLoadStoreRewrite.cpp",
        ),
        "row": (
            "third_party/ascend/lib/TritonToLinalg/RowCoalescing.cpp",
            "third_party/ascend/lib/TritonToGraph/LegacyMemoryAccess/RowCoalescing.cpp",
        ),
    }
    for name, (baseline_path, target_path) in sources.items():
        baseline = _baseline_source(root, baseline_path)
        target = _source_text(root / target_path)
        assert target == _expected_relocated_legacy_core(name, baseline), name
