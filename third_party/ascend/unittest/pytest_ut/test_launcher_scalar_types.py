import ctypes
import importlib.util
import os
from pathlib import Path
import subprocess

import numpy as np
import pytest
import torch
import triton
import triton.language as tl


@pytest.fixture(scope="module")
def scalar_encoder(tmp_path_factory):
    path = Path(__file__).resolve().parents[2] / "backend" / "driver.py"
    spec = importlib.util.spec_from_file_location("ascend_scalar_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    directory = tmp_path_factory.mktemp("scalar_encoder")
    source = directory / "encode.cpp"
    source.write_text("#include <cstdint>\n#include <cstring>\n" + driver._CPP_LOW_PRECISION_SCALARS + r'''
extern "C" void encode(const float *values, uint16_t *output, int size, bool bf16) {
  for (int i = 0; i < size; ++i)
    output[i] = bf16 ? float_to_bf16(values[i]) : float_to_fp16(values[i]);
}
''')
    library = directory / "encode.so"
    subprocess.run(
        [os.environ.get("CXX", "c++"), "-shared", "-fPIC", "-O2",
         str(source), "-o", str(library)], check=True, capture_output=True)
    function = ctypes.CDLL(str(library)).encode
    function.argtypes = [ctypes.c_void_p, ctypes.c_void_p, ctypes.c_int, ctypes.c_bool]
    return function


@pytest.mark.parametrize("dtype", [torch.float16, torch.bfloat16])
def test_scalar_encoding_matches_torch(scalar_encoder, dtype):
    # Include halfway rounding, overflow, subnormals, signed zero, Inf and NaN.
    boundaries = np.array([
        0., -0., 42., -42., 1 + 2**-11, 1 + 3 * 2**-11, 1 + 2**-8, 1 + 3 * 2**-8, 65504., 65520., -65520., 2**-24, 2**
        -25, 3 * 2**-25, 2**-133, 2**-134,
        float("inf"), -float("inf"),
        float("nan")
    ], dtype=np.float32)
    bits = np.random.default_rng(123).integers(0, 2**32, size=100000, dtype=np.uint32)
    values = np.concatenate([boundaries, bits.view(np.float32)])
    result = np.empty(values.size, dtype=np.uint16)
    scalar_encoder(values.ctypes.data, result.ctypes.data, values.size, dtype == torch.bfloat16)
    expected = torch.from_numpy(values).to(dtype).view(torch.uint16).numpy()
    finite_or_inf = ~np.isnan(values)
    np.testing.assert_array_equal(result[finite_or_inf], expected[finite_or_inf])
    exponent, mantissa = (0x7c00, 0x3ff) if dtype == torch.float16 else (0x7f80, 0x7f)
    assert np.all((result[~finite_or_inf] & exponent) == exponent)
    assert np.all((result[~finite_or_inf] & mantissa) != 0)


@pytest.mark.parametrize("dtype,torch_dtype", [(tl.float16, torch.float16), (tl.bfloat16, torch.bfloat16)])
@pytest.mark.parametrize("value", [42., -0., float("inf"), float("nan"), 1.00390625, 2**-24])
def test_half_scalar_between_other_arguments(dtype, torch_dtype, value):

    def kernel(output, before, first, middle, second, after):
        tl.store(output + 0, before)
        tl.store(output + 1, first)
        tl.store(output + 2, middle)
        tl.store(output + 3, second)
        tl.store(output + 4, after)

    kernel.__annotations__ = {
        "before": tl.int32, "first": dtype, "middle": tl.float32, "second": dtype, "after": tl.int64
    }
    compiled = triton.jit(kernel)
    output = torch.empty(5, device="npu", dtype=torch.float32)
    compiled[(1, )](output, 7, value, 2.5, -42., 19)
    rounded = torch.tensor([value, -42.], dtype=torch_dtype).float()
    expected = torch.tensor([7., rounded[0], 2.5, rounded[1], 19.])
    torch.testing.assert_close(output.cpu(), expected, rtol=0, atol=0, equal_nan=True)
    assert torch.signbit(output.cpu()[1]) == torch.signbit(expected[1]) or torch.isnan(expected[1])
