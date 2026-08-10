"""NPU-side regression checks for Ascend specialization cache behavior.

Run this file only with a configured CANN/TorchNPU environment, for example::

    ASCEND_RT_VISIBLE_DEVICES=0 pytest -q test_specialization_cache.py

The SIMT-only path is intentionally covered by the CPU/native binder tests in
``python/test/unit/runtime/test_ascend_specialize.py`` because current 910B
hardware does not advertise the SIMT execution mode.
"""

import pytest
import torch
import torch_npu  # noqa: F401  # registers the NPU backend with PyTorch

import triton
import triton.language as tl


@triton.jit
def _add_value(src_ptr, dst_ptr, n, value, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    src = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, src + value, mask=mask)


@triton.jit
def _add_annotated_value(src_ptr, dst_ptr, n, value: tl.int32, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < n
    src = tl.load(src_ptr + offsets, mask=mask, other=0)
    tl.store(dst_ptr + offsets, src + value, mask=mask)


@triton.jit
def _add_value_tuple(args, BLOCK: tl.constexpr):
    offsets = tl.arange(0, BLOCK)
    mask = offsets < args[2]
    src = tl.load(args[0] + offsets, mask=mask, other=0)
    tl.store(args[1] + offsets, src + args[3], mask=mask)


def _clear_kernel_cache(kernel=_add_value):
    device = torch.npu.current_device()
    kernel.device_caches[device][0].clear()


def _run_case(n, src, dst, mode):
    _add_value[(1,)](src, dst, n, 3, BLOCK=32, compile_mode=mode)
    torch.npu.synchronize()
    expected = src[:n] + 3
    torch.testing.assert_close(dst[:n], expected)


def _run_annotated_case(n, value, src, dst, mode):
    _add_annotated_value[(1,)](src, dst, n, value, BLOCK=32, compile_mode=mode)
    torch.npu.synchronize()
    expected = src[:n] + value
    torch.testing.assert_close(dst[:n], expected)


def _run_tuple_case(n, src, dst, mode):
    _add_value_tuple[(1,)]((src, dst, n, 3), BLOCK=32, compile_mode=mode)
    torch.npu.synchronize()
    expected = src[:n] + 3
    torch.testing.assert_close(dst[:n], expected)


def _assert_aligned_and_unaligned_pair(first, second):
    assert {first.data_ptr() % 16, second.data_ptr() % 16} == {0, first.element_size()}


@pytest.fixture(autouse=True)
def _reset_cache_hook():
    _clear_kernel_cache()
    _clear_kernel_cache(_add_annotated_value)
    _clear_kernel_cache(_add_value_tuple)
    old_hook = triton.knobs.runtime.jit_cache_hook
    try:
        yield
    finally:
        triton.knobs.runtime.jit_cache_hook = old_hook
        _clear_kernel_cache()
        _clear_kernel_cache(_add_annotated_value)
        _clear_kernel_cache(_add_value_tuple)


@pytest.mark.parametrize("mode", ["simd", "unstructured_in_simt"])
@pytest.mark.parametrize("values", [(16, 17), (17, 16)])
def test_simd_integer_alignment_reuses_compilation(mode, values):
    src = torch.arange(32, dtype=torch.int32, device="npu")
    dst = torch.empty_like(src)
    compile_count = 0

    def count_compile(**_kwargs):
        nonlocal compile_count
        compile_count += 1

    triton.knobs.runtime.jit_cache_hook = count_compile
    _run_case(values[0], src, dst, mode)
    _run_case(values[1], src, dst, mode)

    assert compile_count == 1


@pytest.mark.parametrize("mode", ["simd", "unstructured_in_simt"])
@pytest.mark.parametrize("offsets", [(0, 1), (1, 0)])
def test_simd_pointer_alignment_reuses_compilation(mode, offsets):
    src_base = torch.arange(17, dtype=torch.int32, device="npu")
    dst_base = torch.empty_like(src_base)
    compile_count = 0

    def count_compile(**_kwargs):
        nonlocal compile_count
        compile_count += 1

    triton.knobs.runtime.jit_cache_hook = count_compile
    src0 = src_base[offsets[0]: offsets[0] + 16]
    dst0 = dst_base[offsets[0]: offsets[0] + 16]
    src1 = src_base[offsets[1]: offsets[1] + 16]
    dst1 = dst_base[offsets[1]: offsets[1] + 16]
    _assert_aligned_and_unaligned_pair(src0, src1)
    _assert_aligned_and_unaligned_pair(dst0, dst1)
    _run_case(16, src0, dst0, mode)
    _run_case(16, src1, dst1, mode)

    assert compile_count == 1


@pytest.mark.parametrize("mode", ["simd", "unstructured_in_simt"])
@pytest.mark.parametrize("values", [(1, 2), (2, 1)])
def test_simd_integer_one_still_recompiles(mode, values):
    src = torch.arange(32, dtype=torch.int32, device="npu")
    dst = torch.empty_like(src)
    compile_count = 0

    def count_compile(**_kwargs):
        nonlocal compile_count
        compile_count += 1

    triton.knobs.runtime.jit_cache_hook = count_compile
    _run_case(values[0], src, dst, mode)
    _run_case(values[1], src, dst, mode)

    assert compile_count == 2


@pytest.mark.parametrize("mode", ["simd", "unstructured_in_simt"])
@pytest.mark.parametrize("values", [(1, 2), (2, 1)])
def test_simd_annotated_integer_one_is_constexpr_and_recompiles(mode, values):
    src = torch.arange(32, dtype=torch.int32, device="npu")
    dst = torch.empty_like(src)
    compile_records = []

    def record_compile(**kwargs):
        compile_info = kwargs["compile"]
        compile_records.append((compile_info["signature"], compile_info["constants"]))

    triton.knobs.runtime.jit_cache_hook = record_compile
    _run_annotated_case(32, values[0], src, dst, mode)
    _run_annotated_case(32, values[1], src, dst, mode)

    assert len(compile_records) == 2
    for value, (signature, constants) in zip(values, compile_records):
        if value == 1:
            assert signature["value"] == "constexpr"
            assert constants[(3, )] == 1
        else:
            assert signature["value"] == "i32"
            assert (3, ) not in constants


@pytest.mark.parametrize("mode", ["simd", "unstructured_in_simt"])
@pytest.mark.parametrize("values", [(16, 17), (17, 16)])
def test_simd_tuple_integer_alignment_reuses_compilation(mode, values):
    src = torch.arange(32, dtype=torch.int32, device="npu")
    dst = torch.empty_like(src)
    compile_count = 0

    def count_compile(**_kwargs):
        nonlocal compile_count
        compile_count += 1

    triton.knobs.runtime.jit_cache_hook = count_compile
    _run_tuple_case(values[0], src, dst, mode)
    _run_tuple_case(values[1], src, dst, mode)

    assert compile_count == 1


@pytest.mark.parametrize("mode", ["simd", "unstructured_in_simt"])
@pytest.mark.parametrize("offsets", [(0, 1), (1, 0)])
def test_simd_tuple_pointer_alignment_reuses_compilation(mode, offsets):
    src_base = torch.arange(17, dtype=torch.int32, device="npu")
    dst_base = torch.empty_like(src_base)
    compile_count = 0

    def count_compile(**_kwargs):
        nonlocal compile_count
        compile_count += 1

    triton.knobs.runtime.jit_cache_hook = count_compile
    src0 = src_base[offsets[0]: offsets[0] + 16]
    dst0 = dst_base[offsets[0]: offsets[0] + 16]
    src1 = src_base[offsets[1]: offsets[1] + 16]
    dst1 = dst_base[offsets[1]: offsets[1] + 16]
    _assert_aligned_and_unaligned_pair(src0, src1)
    _assert_aligned_and_unaligned_pair(dst0, dst1)
    _run_tuple_case(16, src0, dst0, mode)
    _run_tuple_case(16, src1, dst1, mode)

    assert compile_count == 1


@pytest.mark.parametrize("mode", ["simd", "unstructured_in_simt"])
@pytest.mark.parametrize("values", [(1, 2), (2, 1)])
def test_simd_tuple_integer_one_still_recompiles(mode, values):
    src = torch.arange(32, dtype=torch.int32, device="npu")
    dst = torch.empty_like(src)
    compile_count = 0

    def count_compile(**_kwargs):
        nonlocal compile_count
        compile_count += 1

    triton.knobs.runtime.jit_cache_hook = count_compile
    _run_tuple_case(values[0], src, dst, mode)
    _run_tuple_case(values[1], src, dst, mode)

    assert compile_count == 2
