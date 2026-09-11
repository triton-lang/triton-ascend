"""Device-independent unit tests for Ascend specialization cache keys.

These tests exercise binder/native-specializer policy without CANN or an NPU.
The NPU integration suite separately verifies compilation counts and outputs.
"""

import importlib.util
import inspect
import sys
from pathlib import Path

import pytest
import torch

from triton.backends.compiler import GPUTarget
from triton.runtime.jit import KernelParam, compute_cache_key, create_function_from_signature


def _load_source_specialization_backend():
    """Load the changed backend files instead of an installed wheel copy."""
    backend_root = Path(__file__).resolve().parents[2] / "backend"
    program_grid_name = "triton.backends.ascend.program_grid"
    program_grid_spec = importlib.util.spec_from_file_location(
        program_grid_name, backend_root / "program_grid.py")
    program_grid = importlib.util.module_from_spec(program_grid_spec)
    assert program_grid_spec.loader is not None
    sys.modules[program_grid_name] = program_grid
    program_grid_spec.loader.exec_module(program_grid)

    compiler_spec = importlib.util.spec_from_file_location(
        "triton.backends.ascend.compiler_specialization_cache_under_test",
        backend_root / "compiler.py")
    compiler = importlib.util.module_from_spec(compiler_spec)
    assert compiler_spec.loader is not None
    sys.modules[compiler_spec.name] = compiler
    compiler_spec.loader.exec_module(compiler)
    return compiler, program_grid


_source_compiler, _source_program_grid = _load_source_specialization_backend()
AscendBackend = _source_compiler.AscendBackend
DEFAULT_PROGRAM_MAPPING_RULE_MASK = _source_program_grid.DEFAULT_PROGRAM_MAPPING_RULE_MASK
INDEPENDENT_AXIS_TENSORIZE_RULE_BIT = _source_program_grid.INDEPENDENT_AXIS_TENSORIZE_RULE_BIT
PERSISTENT_TASK_STRIP_MINING_RULE_BIT = _source_program_grid.PERSISTENT_TASK_STRIP_MINING_RULE_BIT
ProgramGridContractError = _source_program_grid.ProgramGridContractError
make_program_mapping_compile_options = _source_program_grid.make_program_mapping_compile_options

pytestmark = pytest.mark.backend("native")


class PointerArg:

    dtype = torch.float32

    def __init__(self, address):
        self.address = address

    def data_ptr(self):
        return self.address


class CountingAscendBackend(AscendBackend):

    def __init__(self):
        super().__init__(GPUTarget("npu", "Ascend910B", 32))
        self.alignment_specialization_calls = 0

    def use_alignment_specialization(self, options):
        self.alignment_specialization_calls += 1
        return super().use_alignment_specialization(options)


def make_binder(backend=None, do_not_specialize_on_alignment=False):

    def kernel(value, pointer):
        pass

    signature = inspect.signature(kernel)
    params = [
        KernelParam(i, param, False, do_not_specialize_on_alignment)
        for i, param in enumerate(signature.parameters.values())
    ]
    if backend is None:
        backend = AscendBackend(GPUTarget("npu", "Ascend910B", 32))
    return create_function_from_signature(signature, params, backend)


def bind_and_key(binder, cache, value, address, options):
    _, specialization, raw_options = binder(value, PointerArg(address), **options)
    return specialization, compute_cache_key(cache, specialization, raw_options)


SIMD_OPTIONS = [
    pytest.param({"compile_mode": "simd"}, id="simd"),
    pytest.param({"compile_mode": "unstructured_in_simt"}, id="unstructured-in-simt"),
    pytest.param({"force_simt_template": True}, id="force-simt-template"),
    pytest.param({}, id="default-simd-simt-template"),
]

SIMT_OPTIONS = [
    pytest.param({"compile_mode": "simt_only"}, id="simt-only"),
    pytest.param({"force_simt_only": True}, id="force-simt-only"),
    pytest.param({"compile_mode": "simd", "force_simt_only": True}, id="force-simt-overrides-mode"),
]


def test_alignment_policy_is_evaluated_once_per_binder_call():
    backend = CountingAscendBackend()
    binder = make_binder(backend)

    binder(16, PointerArg(0x1000), compile_mode="simd")
    assert backend.alignment_specialization_calls == 1

    binder(17, PointerArg(0x1004), compile_mode="simt_only")
    assert backend.alignment_specialization_calls == 2


def test_alignment_policy_is_not_evaluated_when_unused():
    backend = CountingAscendBackend()
    binder = make_binder(backend, do_not_specialize_on_alignment=True)

    binder(16, PointerArg(0x1000), compile_mode="simd")

    assert backend.alignment_specialization_calls == 0


def test_alignment_policy_is_not_evaluated_for_non_specialized_annotation():

    def kernel(value):
        pass

    kernel.__annotations__["value"] = "fp32"
    signature = inspect.signature(kernel)
    params = [KernelParam(0, next(iter(signature.parameters.values())), False, False)]
    backend = CountingAscendBackend()
    binder = create_function_from_signature(signature, params, backend)

    _, specialization, _ = binder(1.25, compile_mode="simd")

    assert specialization == [("fp32", None)]
    assert backend.alignment_specialization_calls == 0


def test_alignment_policy_is_not_evaluated_when_value_specialization_is_disabled():

    def kernel(value):
        pass

    signature = inspect.signature(kernel)
    params = [KernelParam(0, next(iter(signature.parameters.values())), True, False)]
    backend = CountingAscendBackend()
    binder = create_function_from_signature(signature, params, backend)

    _, specialization, _ = binder(16, compile_mode="simd")

    assert specialization == [("i32", None)]
    assert backend.alignment_specialization_calls == 0


@pytest.mark.parametrize("options", SIMD_OPTIONS)
@pytest.mark.parametrize("values", [(16, 17), (17, 16)], ids=["aligned-first", "unaligned-first"])
def test_simd_integer_alignment_reuses_cache_key(options, values):
    binder = make_binder()
    cache = {}
    first, first_key = bind_and_key(binder, cache, values[0], 0x1004, options)
    second, second_key = bind_and_key(binder, cache, values[1], 0x1004, options)

    assert first_key == second_key
    assert first[0][1] == second[0][1] == ""


@pytest.mark.parametrize("options", [
    pytest.param({"compile_mode": "simd"}, id="simd"),
    pytest.param({"compile_mode": "simd_simt_template"}, id="simd-simt-template"),
])
@pytest.mark.parametrize(
    "values, expected_type",
    [
        pytest.param((-16, -15), "i32", id="negative-i32"),
        pytest.param((1 << 32, (1 << 32) + 1), "i64", id="i64"),
        pytest.param((1 << 63, (1 << 63) + 1), "u64", id="u64"),
    ],
)
def test_simd_integer_alignment_reuses_cache_key_across_integer_types(options, values, expected_type):
    binder = make_binder()
    cache = {}
    first, first_key = bind_and_key(binder, cache, values[0], 0x1004, options)
    second, second_key = bind_and_key(binder, cache, values[1], 0x1004, options)

    assert first_key == second_key
    assert first[0] == second[0] == (expected_type, "")


@pytest.mark.parametrize("options", SIMD_OPTIONS)
@pytest.mark.parametrize("addresses", [(0x1000, 0x1004), (0x1004, 0x1000)], ids=["aligned-first", "unaligned-first"])
def test_simd_pointer_alignment_reuses_cache_key(options, addresses):
    binder = make_binder()
    cache = {}
    first, first_key = bind_and_key(binder, cache, 17, addresses[0], options)
    second, second_key = bind_and_key(binder, cache, 17, addresses[1], options)

    assert first_key == second_key
    assert first[1][1] == second[1][1] == ""


@pytest.mark.parametrize("options", SIMT_OPTIONS)
@pytest.mark.parametrize("values", [(16, 17), (17, 16)], ids=["aligned-first", "unaligned-first"])
def test_simt_integer_alignment_keeps_distinct_cache_keys(options, values):
    binder = make_binder()
    cache = {}
    first, first_key = bind_and_key(binder, cache, values[0], 0x1004, options)
    second, second_key = bind_and_key(binder, cache, values[1], 0x1004, options)

    assert first_key != second_key
    assert {first[0][1], second[0][1]} == {"", "D"}


@pytest.mark.parametrize("options", SIMT_OPTIONS)
@pytest.mark.parametrize("addresses", [(0x1000, 0x1004), (0x1004, 0x1000)], ids=["aligned-first", "unaligned-first"])
def test_simt_pointer_alignment_keeps_distinct_cache_keys(options, addresses):
    binder = make_binder()
    cache = {}
    first, first_key = bind_and_key(binder, cache, 17, addresses[0], options)
    second, second_key = bind_and_key(binder, cache, 17, addresses[1], options)

    assert first_key != second_key
    assert {first[1][1], second[1][1]} == {"", "D"}


@pytest.mark.parametrize("options", SIMD_OPTIONS + SIMT_OPTIONS)
@pytest.mark.parametrize("values", [(1, 2), (2, 1)], ids=["one-first", "one-second"])
def test_integer_one_keeps_distinct_cache_key(options, values):
    binder = make_binder()
    cache = {}
    first, first_key = bind_and_key(binder, cache, values[0], 0x1004, options)
    second, second_key = bind_and_key(binder, cache, values[1], 0x1004, options)

    assert first_key != second_key
    assert {first[0][0], second[0][0]} == {"constexpr", "i32"}


@pytest.mark.parametrize("options", SIMD_OPTIONS + SIMT_OPTIONS)
def test_annotated_integer_one_keeps_constexpr_specialization(options):

    def kernel(value):
        pass

    annotation = "i32"
    kernel.__annotations__["value"] = annotation
    signature = inspect.signature(kernel)
    params = [KernelParam(0, next(iter(signature.parameters.values())), False, False)]
    backend = AscendBackend(GPUTarget("npu", "Ascend910B", 32))
    binder = create_function_from_signature(signature, params, backend)
    cache = {}

    _, one_specialization, one_options = binder(1, **options)
    _, two_specialization, two_options = binder(2, **options)

    assert one_specialization == [("constexpr", 1)]
    assert two_specialization == [(annotation, "")]
    assert compute_cache_key(cache, one_specialization,
                             one_options) != compute_cache_key(cache, two_specialization, two_options)


@pytest.mark.parametrize("options", [SIMD_OPTIONS[0], *SIMT_OPTIONS])
@pytest.mark.parametrize("values", [(1, 2), (16, 17)], ids=["value-specialization", "alignment-specialization"])
def test_annotated_integer_do_not_specialize_keeps_legacy_cache_key(options, values):

    def kernel(value):
        pass

    kernel.__annotations__["value"] = "i32"
    signature = inspect.signature(kernel)
    params = [KernelParam(0, next(iter(signature.parameters.values())), True, False)]
    backend = AscendBackend(GPUTarget("npu", "Ascend910B", 32))
    binder = create_function_from_signature(signature, params, backend)
    cache = {}

    _, first_specialization, first_options = binder(values[0], **options)
    first_key = compute_cache_key(cache, first_specialization, first_options)
    _, second_specialization, second_options = binder(values[1], **options)
    second_key = compute_cache_key(cache, second_specialization, second_options)

    assert first_specialization == second_specialization == [("i32", None)]
    assert first_key == second_key


def test_program_mapping_factory_canonicalizes_raw_jit_options_and_backend_hash():
    rule_mask = INDEPENDENT_AXIS_TENSORIZE_RULE_BIT | PERSISTENT_TASK_STRIP_MINING_RULE_BIT
    first = make_program_mapping_compile_options(
        (65, 1),
        rule_mask,
        scalar_arguments=[(3, "tokens", 19)],
        compile_mode="unstructured_in_simt",
    )
    equivalent = make_program_mapping_compile_options(
        [65, 1, 1],
        rule_mask,
        scalar_arguments=((3, "tokens", 19), ),
        compile_mode="simd_simt_template",
    )

    assert first == equivalent
    assert repr(first) == repr(equivalent)
    assert list(first) == [
        "program_mapping_rule_mask",
        "program_grid_specialization",
        "program_grid_transform_schema_version",
        "program_mapping_scalar_specialization",
        "compile_mode",
    ]

    backend = AscendBackend(GPUTarget("npu", "Ascend910B", 32))
    first_options = backend.parse_options(dict(first))
    equivalent_options = backend.parse_options(dict(equivalent))
    for option_name in (
            "program_mapping_rule_mask",
            "program_grid_specialization",
            "program_mapping_scalar_specialization",
    ):
        assert option_name in first_options.__dict__
    assert first_options.hash() == equivalent_options.hash()

    specialization = [("i32", "")]
    assert compute_cache_key({}, specialization, first) == compute_cache_key({}, specialization, equivalent)

    changed = make_program_mapping_compile_options(
        (66, 1, 1),
        rule_mask,
        scalar_arguments=[(3, "tokens", 19)],
    )
    changed_options = backend.parse_options(changed)
    assert compute_cache_key({}, specialization, first) != compute_cache_key({}, specialization, changed)
    assert first_options.hash() != changed_options.hash()


def test_program_mapping_default_stays_explicit_and_no_jit_hook_exists():
    backend = AscendBackend(GPUTarget("npu", "Ascend910B", 32))
    default_options = backend.parse_options({})
    assert default_options.program_mapping_rule_mask == DEFAULT_PROGRAM_MAPPING_RULE_MASK
    assert "program_grid_specialization" not in default_options.__dict__

    with pytest.raises(ProgramGridContractError, match="enabled rule mask"):
        make_program_mapping_compile_options((1, ), 0)
    with pytest.raises(ProgramGridContractError, match="must not be empty"):
        make_program_mapping_compile_options(
            (1, ),
            INDEPENDENT_AXIS_TENSORIZE_RULE_BIT,
            scalar_arguments=[],
        )

    assert not hasattr(AscendBackend, "prepare_program_grid_specialization")
    assert not hasattr(AscendBackend, "prepare_program_mapping_specialization")
    assert not hasattr(AscendBackend, "finalize_program_mapping_launch_grid")
