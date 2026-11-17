import os
import re
import subprocess
import sys
import uuid

import pytest
import torch

import torch_npu  # noqa: F401
from torch.testing import assert_close

import triton
import triton.language as tl
from triton._internal_testing import is_interpreter


def get_current_target_warp_size():
    return triton.runtime.driver.active.get_current_target().warp_size or 32


def is_a5_target_arch():
    arch = triton.runtime.driver.active.get_current_target().arch
    return isinstance(arch, str) and arch.startswith(("Ascend910_95", "Ascend950"))


@triton.jit
def kernel_device_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_print("x: ", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_device_print_cast(BLOCK: tl.constexpr):
    x = tl.arange(0, BLOCK) + 128
    tl.device_print("x: ", x.to(tl.uint8))


@triton.jit
def kernel_device_print_hex(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.device_print("x: ", x, hex=True)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_print(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    # Triton should add a space after this prefix.
    print("x:", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_device_print_scalar(SCALAR):
    x = tl.load(SCALAR)
    # Triton should add a space after this prefix.
    print("x:", x)


@triton.jit
def kernel_device_print_large(
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
):
    x = tl.full([BLOCK_M, BLOCK_N], 1, tl.int32)
    # Triton should change this prefix to "x: ".
    tl.device_print("x ", x)


@triton.jit
def kernel_print_multiple_args(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.full((BLOCK, ), 1, tl.int32)
    print("", x, y)


@triton.jit
def kernel_device_print_multiple_args(X, Y, BLOCK: tl.constexpr):
    x = tl.load(X + tl.arange(0, BLOCK))
    y = tl.full((BLOCK, ), 1, tl.int32)
    tl.device_print("", x, y)
    tl.store(Y + tl.arange(0, BLOCK), y)


@triton.jit
def kernel_static_print(X, Y, BLOCK: tl.constexpr, PLACEHOLDER: tl.constexpr):
    # This function takes an extra value as a tl.constexpr so this kernel is not
    # cached.  This way the static print is run every time.
    x = tl.load(X + tl.arange(0, BLOCK))
    tl.static_print("", x)
    tl.store(Y + tl.arange(0, BLOCK), x)


@triton.jit
def kernel_no_arg_print():
    print("", tl.program_id(0))


@triton.jit
def kernel_print_no_arg():
    print("no arg")


@triton.jit
def kernel_print_pointer(X, Y, BLOCK: tl.constexpr):
    tl.device_print("ptr ", X + tl.arange(0, BLOCK))


@triton.jit
def kernel_print_2d_tensor(X, Y, BLOCK_SIZE_X: tl.constexpr, BLOCK_SIZE_Y: tl.constexpr):
    off_x = tl.arange(0, BLOCK_SIZE_X)
    off_y = tl.arange(0, BLOCK_SIZE_Y)
    x = tl.load(X + off_x[:, None] * BLOCK_SIZE_Y + off_y[None, :])
    tl.device_print("", x)


def run_print_case(func: str, data_type: str, device: str):
    N = 128  # This value should match with test_print below.
    # TODO(antiagainst): Currently the warp count is chosen to make sure we don't have multiple
    # threads printing duplicated messages due to broadcasting. Improve print op lowering logic
    # to filter out duplicated data range.
    num_warps = N // get_current_target_warp_size()

    x = torch.arange(0, N, dtype=torch.int32, device=device).to(getattr(torch, data_type))
    y = torch.zeros((N, ), dtype=x.dtype, device=device)
    if func == "device_print":
        kernel_device_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_scalar":
        scalar = torch.tensor(42, dtype=x.dtype, device=device)
        kernel_device_print_scalar[(1, )](scalar, num_warps=num_warps)
    elif func == "device_print_negative":
        x = -x
        kernel_device_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_uint":
        x = torch.arange((1 << 31), (1 << 31) + N, device=device).to(getattr(torch, data_type))
        kernel_device_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_uint_cast":
        kernel_device_print_cast[(1, )](num_warps=num_warps, BLOCK=N)
    elif func == "print":
        kernel_print[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_large":
        kernel_device_print_large[(1, 2)](BLOCK_M=64 // 8, num_warps=num_warps, BLOCK_N=N // 8)
    elif func == "print_multiple_args":
        kernel_print_multiple_args[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_multiple_args":
        kernel_device_print_multiple_args[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "static_print":
        kernel_static_print[(1, )](x, y, num_warps=num_warps, BLOCK=N, PLACEHOLDER=uuid.uuid4())
    elif func == "no_arg_print":
        kernel_no_arg_print[(1, )](num_warps=num_warps)
    elif func == "print_no_arg":
        kernel_print_no_arg[(1, )](num_warps=num_warps)
    elif func == "device_print_hex":
        kernel_device_print_hex[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_pointer":
        kernel_print_pointer[(1, )](x, y, num_warps=num_warps, BLOCK=N)
    elif func == "device_print_2d_tensor":
        BLOCK_SIZE_X = num_warps
        BLOCK_SIZE_Y = get_current_target_warp_size()
        x_2d_tensor = x.reshape((BLOCK_SIZE_X, BLOCK_SIZE_Y))
        kernel_print_2d_tensor[(1, )](x_2d_tensor, y, num_warps=num_warps, BLOCK_SIZE_X=BLOCK_SIZE_X,
                                      BLOCK_SIZE_Y=BLOCK_SIZE_Y)
    else:
        assert f"Unknown kernel: {func}"

    excluded_funcs = {
        "print_no_arg", "no_arg_print", "device_print_large", "print_multiple_args", "device_print_multiple_args",
        "device_print_pointer", "device_print_scalar", "device_print_2d_tensor", "device_print_uint_cast"
    }
    if func not in excluded_funcs:
        assert_close(y, x)

    # Wait until driver complete all the jobs for the device_print, especially test_subprocess
    # require this which captures stdout when child exits.
    getattr(torch, device).synchronize()


print_path = os.path.abspath(__file__)
torch_types = ["int8", "uint8", "int16", "int32", "long", "float16", "float32", "float64"]

os.environ.setdefault("TRITON_DEVICE_PRINT", "1")

# The runtime frames each dump with a core marker line: "==> Logical Block N"
# on 910B and "=> Vec N" on 950.
_ASCEND_FRAMING_PREFIXES = ("==> ", "=> ")


def _ascend_payload(stdout: str):
    payload = []
    for line in stdout.splitlines():
        if not line or set(line) == {"-"} or "HiIPU Print" in line:
            continue
        if line.startswith(_ASCEND_FRAMING_PREFIXES):
            continue
        payload.append(line)
    return payload


def _ascend_tensor_dump(values, data_type: str, hex_values: bool = False) -> str:
    if hex_values:
        body = ",".join(f"{v:X}" for v in values)
    elif data_type.startswith("float"):
        # The runtime formats floats with printf's %f.
        body = ",".join(f"{v}.000000" for v in values)
    else:
        body = ",".join(str(v) for v in values)
    return f"[{body}]"


def _expected_ascend_payload(func_type: str, data_type: str, n: int, warp_size: int):
    if func_type in ("print", "device_print"):
        return [" x: ", _ascend_tensor_dump(range(n), data_type)]
    if func_type == "device_print_scalar":
        value = "42.000000" if data_type.startswith("float") else "42"
        return [" x: ", value]
    if func_type == "device_print_negative":
        return [" x: ", _ascend_tensor_dump(range(0, -n, -1), data_type)]
    if func_type == "device_print_uint":
        # uint32 values above INT32_MAX must print as non-negative numbers.
        start = 1 << 31
        return [" x: ", _ascend_tensor_dump(range(start, start + n), data_type)]
    if func_type == "device_print_uint_cast":
        # (arange + 128).to(uint8) wraps to 128..255 and must print unsigned.
        start = 1 << 7
        return [" x: ", _ascend_tensor_dump(range(start, start + n), data_type)]
    if func_type == "device_print_hex":
        return [" x: ", _ascend_tensor_dump(range(n), data_type, hex_values=True)]
    if func_type in ("print_multiple_args", "device_print_multiple_args"):
        # Each operand becomes its own dump; the prefix is " " (normalized to
        # ": " by the frontend) and is repeated for every operand.
        return [": ", _ascend_tensor_dump(range(n), data_type), ": ", _ascend_tensor_dump([1] * n, data_type)]
    if func_type == "static_print":
        # Compile-time print: the dump happens on the host, once.
        return [f" int32[constexpr[{n}]]"]
    if func_type == "no_arg_print":
        # Scalar program_id() prints once, unlike CUDA's per-thread output.
        return [": ", "0"]
    if func_type == "print_no_arg":
        # A print with no operands produces no device dump on Ascend.
        return []
    if func_type == "device_print_2d_tensor":
        if is_a5_target_arch():
            return [": ", _ascend_tensor_dump(range(n), data_type)]
        y_dim = warp_size
        x_dim = n // y_dim
        rows = [",".join(str(x * y_dim + y) for y in range(y_dim)) for x in range(x_dim)]
        return [": "] + ("[[" + "],\n[".join(rows) + "]]").splitlines()
    if func_type == "device_print_large":
        elements_per_block = (64 // 8) * (n // 8)
        return [" x: ", "1"] * (2 * elements_per_block)
    raise AssertionError(f"no Ascend expectation for func_type={func_type}")


def _check_ascend_print(func_type: str, data_type: str, stdout: str, n: int, warp_size: int):
    payload = _ascend_payload(stdout)
    if func_type == "device_print_pointer":
        assert payload and len(payload) % 2 == 0, f"Unexpected Ascend pointer print payload in:\n{stdout}"
        for prefix, value in zip(payload[0::2], payload[1::2]):
            assert prefix == " ptr: ", f"Unexpected Ascend pointer print prefix in:\n{stdout}"
            assert re.fullmatch(r"-?\d+", value), f"Unexpected Ascend pointer dump in:\n{stdout}"
        return

    expected = _expected_ascend_payload(func_type, data_type, n, warp_size)

    block = len(expected)
    repeated = block > 0 and len(payload) % block == 0 and payload == expected * (len(payload) // block)
    if payload != expected and not repeated:
        expected_text = "\n".join(expected)
        actual_text = "\n".join(payload)
        raise AssertionError(f"Unexpected Ascend print payload for {func_type}/{data_type}\n"
                             f"expected:\n{expected_text}\nactual:\n{actual_text}\nfull stdout:\n{stdout}")


@pytest.mark.interpreter
@pytest.mark.parametrize("func_type, data_type", [(fn, data_type)
                                                  for fn in ["device_print", "device_print_scalar"]
                                                  for data_type in torch_types] + [
                                                      ("print", "int32"),
                                                      ("static_print", "int32"),
                                                      ("no_arg_print", "int32"),
                                                      ("print_no_arg", "int32"),
                                                      ("device_print_large", "int32"),
                                                      ("print_multiple_args", "int32"),
                                                      ("device_print_multiple_args", "int32"),
                                                      ("device_print_hex", "int16"),
                                                      ("device_print_hex", "int32"),
                                                      ("device_print_hex", "int64"),
                                                      ("device_print_pointer", "int32"),
                                                      ("device_print_negative", "int32"),
                                                      ("device_print_uint_cast", "uint8"),
                                                      ("device_print_2d_tensor", "int32"),
                                                  ])
def test_print(func_type: str, data_type: str):
    proc = subprocess.run(
        [sys.executable, print_path, "run_print_case", func_type, data_type, "npu"],
        capture_output=True,
    )
    assert proc.returncode == 0, proc.stderr.decode("UTF-8", errors="replace")

    if is_interpreter() and func_type != "static_assert":
        # Interpreter uses a different format for device_print
        # Only check if there's no error
        assert proc.stderr == b''
        return

    N = 128

    _check_ascend_print(func_type, data_type, proc.stdout.decode("UTF-8"), N, get_current_target_warp_size())


if __name__ == "__main__":
    fn = globals()[sys.argv[1]]
    fn(*sys.argv[2:])
