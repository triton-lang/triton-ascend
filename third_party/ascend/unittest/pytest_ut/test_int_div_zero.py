# Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
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

# Regression tests for integer division / remainder by a zero divisor on
# Ascend (issue #2518).
#
# Compiled-kernel contract (verified baseline: CANN 9.2.0, Ascend 910B,
# default SIMD(hfusion) pipeline):
#   * x % 0 returns the all-ones value of the result type (-1 for signed,
#     2**n - 1 for unsigned). Divisor forms pinned by tests: loaded tensor
#     with zero lanes for every observed width, plus in-kernel splat zero
#     and literal 0 for int32/int64. Non-zero lanes follow
#     C-truncation (np.fmod) remainder semantics. The guard is inserted by
#     the BiShengIR HFusion NormalizeModOp pass: (y == 0) ? -1 : x % y.
#   * x // 0 saturates to {0, +MAX, -MIN} for i32 (all three divisor forms)
#     and for i64 with a loaded divisor. i64 // splat/literal zero currently
#     returns the dividend (pinned below as a known gap, xfail).
#   * x // 0 for i8/i16 returns {-1, 0} instead of saturating (known gap).
#   * Unsigned remainder wider than 8 bits fails to compile (vmodui / vcast
#     legalization gaps).
# The Ascend interpreter (TRITON_INTERPRET=1) must produce the same values
# as the compiled kernels for the remainder promise; parity tests below run
# each kernel a second time in a child process with TRITON_INTERPRET=1 over
# CPU tensors (the flag is read at @triton.jit decoration time, so it cannot
# be flipped mid-process) and compare element-wise against the compiled
# outputs. For x // 0 the interpreter saturates in every configuration,
# including the compiled gaps below, so those combinations are excluded
# from the parity set rather than asserted false.
#
# SIMT-mode lowering and other SoCs (e.g. the Ascend 950 generation) are
# outside the verified baseline, so every case that launches a compiled
# kernel on device is gated on the verified arch; CPU-only interpreter
# cases run everywhere.
#
# Each compiled device launch runs in a short-lived child process: AIV
# kernels that execute a raw divide-by-zero can raise a delayed
# "Vector Core kernel execution failed" exception whose report poisons the
# stream of whoever syncs next. The results are correct, so running each
# case in its own process keeps the suite deterministic on shared CI cards.

import importlib.util
import json
import os
import subprocess
import sys
import tempfile

# Child-process mode: TRITON_INTERPRET must be set before @triton.jit
# decoration below, so honour the --interpret flag before importing triton.
_CHILD_INTERPRET = "--case" in sys.argv and "--interpret" in sys.argv
if _CHILD_INTERPRET:
    os.environ["TRITON_INTERPRET"] = "1"

import numpy as np
import pytest

import triton
import triton.language as tl

# conftest.py's module-scoped assign_npu fixture depends on pytest-xdist's
# worker_id. Mirror its single-process value ("master" -> device 0, i.e. the
# first card in ASCEND_RT_VISIBLE_DEVICES) when xdist is not installed.
if importlib.util.find_spec("xdist") is None:

    @pytest.fixture(scope="session")
    def worker_id():
        return "master"


# ---------------------------------------------------------------------------
# verified-baseline architecture gate
# ---------------------------------------------------------------------------
# The compiled zero-divisor contract is pinned on the Ascend 910B family
# with CANN 9.2.0 and the default SIMD(hfusion) pipeline. Other SoCs and
# SIMT mode are outside that baseline, so device-launched cases must not
# assert it there. Detection follows the in-repo precedent
# (test_load_strided.py, test_pointer_reshape.py): read the arch string
# from triton.runtime.driver.active.get_current_target().arch.
_VERIFIED_BASELINE_ARCH = "Ascend910B"


def _is_verified_baseline_arch(arch):
    """Return whether arch is the SoC family the contract was verified on."""
    return isinstance(arch, str) and arch.startswith(_VERIFIED_BASELINE_ARCH)


try:
    _TARGET_ARCH = triton.runtime.driver.active.get_current_target().arch
except Exception:
    # No resolvable NPU target: treat as unverified so the device cases
    # skip instead of erroring at collection.
    _TARGET_ARCH = ""

requires_verified_baseline_arch = pytest.mark.skipif(
    not _is_verified_baseline_arch(_TARGET_ARCH),
    reason="verified baseline is Ascend 910B / CANN 9.2.0 SIMD(hfusion); "
    "other SoCs unverified for the compiled zero-divisor contract",
)

BLOCK = 1024
SMALL = 16
SEED = 2518
HEAD = [0, 1, -1, 5, 7, 123, -3, -12345]
HEAD8 = [0, 1, -1, 5, 7, 100, -3, -100]
HEADU8 = [0, 1, 2, 5, 7, 123, 3, 100]
NONZERO_DIVISORS = (3, 2, 7, -5)
NONZERO_DIVISORS_U = (3, 2, 7, 5)

# ---------------------------------------------------------------------------
# kernels
# ---------------------------------------------------------------------------


@triton.jit
def k_rem_vec(XP, YP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(XP + offs)
    y = tl.load(YP + offs)
    tl.store(OP + offs, x % y)


@triton.jit
def k_div_vec(XP, YP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(XP + offs)
    y = tl.load(YP + offs)
    tl.store(OP + offs, x // y)


@triton.jit
def k_rem_splat_i32(XP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(XP + offs)
    y = tl.zeros([BLOCK], dtype=tl.int32)
    tl.store(OP + offs, x % y)


@triton.jit
def k_rem_splat_i64(XP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(XP + offs)
    y = tl.zeros([BLOCK], dtype=tl.int64)
    tl.store(OP + offs, x % y)


@triton.jit
def k_div_splat_i32(XP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(XP + offs)
    y = tl.zeros([BLOCK], dtype=tl.int32)
    tl.store(OP + offs, x // y)


@triton.jit
def k_div_splat_i64(XP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(XP + offs)
    y = tl.zeros([BLOCK], dtype=tl.int64)
    tl.store(OP + offs, x // y)


@triton.jit
def k_rem_literal_i32(XP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(XP + offs)
    tl.store(OP + offs, x % 0)


@triton.jit
def k_rem_literal_i64(XP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(XP + offs)
    tl.store(OP + offs, x % 0)


@triton.jit
def k_div_literal_i32(XP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(XP + offs)
    tl.store(OP + offs, x // 0)


@triton.jit
def k_div_literal_i64(XP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    x = tl.load(XP + offs)
    tl.store(OP + offs, x // 0)


# Unsigned remainder for widths torch cannot express as tensors: load the
# signed torch dtype and cast in-kernel.
@triton.jit
def k_urem_vec_u16(XP, YP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    xu = tl.load(XP + offs).to(tl.uint16)
    yu = tl.load(YP + offs).to(tl.uint16)
    tl.store(OP + offs, (xu % yu).to(tl.int16))


@triton.jit
def k_urem_vec_u32(XP, YP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    xu = tl.load(XP + offs).to(tl.uint32)
    yu = tl.load(YP + offs).to(tl.uint32)
    tl.store(OP + offs, (xu % yu).to(tl.int32))


@triton.jit
def k_urem_vec_u64(XP, YP, OP, BLOCK: tl.constexpr):
    offs = tl.arange(0, BLOCK)
    xu = tl.load(XP + offs).to(tl.uint64)
    yu = tl.load(YP + offs).to(tl.uint64)
    tl.store(OP + offs, (xu % yu).to(tl.int64))


# ---------------------------------------------------------------------------
# data and reference helpers
# ---------------------------------------------------------------------------


def _make_array(dtype, n, head=None, lo=None, hi=None):
    dt = np.dtype(dtype)
    info = np.iinfo(dt)
    lo = int(info.min) if lo is None else max(int(lo), int(info.min))
    hi = int(info.max) if hi is None else min(int(hi), int(info.max))
    rng = np.random.default_rng(SEED)
    arr = rng.integers(lo, hi, size=n).astype(dt)
    if head is not None:
        arr[:len(head)] = np.array(head, dtype=dt)
    return arr


def _make_mixed_divisor(dtype, n):
    dt = np.dtype(dtype)
    divisors = NONZERO_DIVISORS if dt.kind == 'i' else NONZERO_DIVISORS_U
    y = np.zeros(n, dtype=dt)
    idx = np.arange(1, n, 2)
    y[idx] = np.array([divisors[i % len(divisors)] for i in range(len(idx))], dtype=dt)
    return y


def _all_ones(dt):
    dt = np.dtype(dt)
    return dt.type(-1) if dt.kind == 'i' else dt.type(np.iinfo(dt).max)


def _rem_expected(x, y):
    # Zero-divisor lanes: all-ones of the result type (compiler guarantee).
    # Non-zero lanes: C-truncation remainder (np.fmod), what the compiled
    # kernel produces there.
    with np.errstate(divide='ignore', invalid='ignore'):
        ref = np.fmod(x, y)
    return np.where(y == 0, _all_ones(x.dtype), ref).astype(x.dtype)


def _div_expected(x, y):
    # Zero-divisor lanes saturate by the sign of the dividend; non-zero
    # lanes are truncated division (x - x % y) // y.
    info = np.iinfo(x.dtype)
    with np.errstate(divide='ignore', invalid='ignore'):
        trunc = (x - np.fmod(x, y)) // y
    sat = np.where(x == 0, 0, np.where(x > 0, info.max, info.min))
    return np.where(y == 0, sat, trunc).astype(x.dtype)


# ---------------------------------------------------------------------------
# compiled device execution: one short-lived child process per case
# ---------------------------------------------------------------------------

COMPILED_CASES = {
    'rem_vec': (k_rem_vec, True),
    'rem_splat_int32': (k_rem_splat_i32, False),
    'rem_splat_int64': (k_rem_splat_i64, False),
    'rem_literal_int32': (k_rem_literal_i32, False),
    'rem_literal_int64': (k_rem_literal_i64, False),
    'div_vec': (k_div_vec, True),
    'div_splat_int32': (k_div_splat_i32, False),
    'div_splat_int64': (k_div_splat_i64, False),
    'div_literal_int32': (k_div_literal_i32, False),
    'div_literal_int64': (k_div_literal_i64, False),
    'urem_vec_u16': (k_urem_vec_u16, True),
    'urem_vec_u32': (k_urem_vec_u32, True),
    'urem_vec_u64': (k_urem_vec_u64, True),
}


def _run_compiled_case(key, x_np, n, y_np=None):
    """Run one compiled case in a child process and return its output."""
    return _run_child_case(key, x_np, n, y_np, interpret=False)


def _run_child_case(key, x_np, n, y_np=None, interpret=False):
    """Run one case in a short-lived child process and return its output."""
    kernel, uses_y = COMPILED_CASES[key]
    assert uses_y or y_np is None
    with tempfile.TemporaryDirectory(prefix='ta2518_') as td:
        np.savez(os.path.join(td, 'inputs.npz'), x=x_np, y=y_np if y_np is not None else np.zeros(0, dtype=x_np.dtype))
        cmd = [sys.executable, os.path.abspath(__file__), '--case', key, '--io', td]
        if interpret:
            cmd.append('--interpret')
        env = dict(os.environ, TORCH_DEVICE_BACKEND_AUTOLOAD='0')
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600, env=env,
                              cwd=os.path.dirname(os.path.abspath(__file__)))
        result_path = os.path.join(td, 'result.json')
        if not os.path.exists(result_path):
            raise RuntimeError(f"case {key} produced no result (exit {proc.returncode}): {proc.stderr[-400:]}")
        result = json.load(open(result_path))
        if result['status'] != 'ok':
            raise RuntimeError(f"case {key} failed on device: {result['error']}")
        return np.load(os.path.join(td, 'out.npy'))


def _run_interpreted_case(key, x_np, n, y_np=None):
    """Run the same case with TRITON_INTERPRET=1 over CPU tensors."""
    return _run_child_case(key, x_np, n, y_np, interpret=True)


# ---------------------------------------------------------------------------
# child-process case bodies (invoked via --case/--io, see __main__ below)
# ---------------------------------------------------------------------------


def _child_main(key, io_dir, interpret):
    import torch
    if not interpret:
        import torch_npu  # device tensors need the backend registered
    inp = np.load(os.path.join(io_dir, 'inputs.npz'))
    x_np = inp['x']
    y_np = inp['y'] if inp['y'].size else None
    kernel = COMPILED_CASES[key][0]
    status = {'status': 'error', 'error': 'not-run'}
    try:
        device = 'cpu' if interpret else 'npu'
        x = torch.from_numpy(x_np.copy()).to(device)
        args = [x]
        if y_np is not None:
            args.append(torch.from_numpy(y_np.copy()).to(device))
        out = torch.zeros_like(x)
        args.append(out)
        kernel[(1, 1, 1)](*args, BLOCK=x_np.shape[0])
        if not interpret:
            torch.npu.synchronize()
        np.save(os.path.join(io_dir, 'out.npy'), out.detach().cpu().numpy())
        status = {'status': 'ok'}
    except Exception as exc:  # keep whatever the child managed to report
        status = {'status': 'error', 'error': f'{type(exc).__name__}: {str(exc)[:300]}'}
    with open(os.path.join(io_dir, 'result.json'), 'w') as f:
        json.dump(status, f)


# ---------------------------------------------------------------------------
# compiled device matrix: % by zero (core promise: all-ones, every form)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('dtype,head', [
    ('int32', HEAD),
    ('int64', HEAD),
])
@requires_verified_baseline_arch
def test_rem_loaded_mixed(dtype, head):
    x = _make_array(dtype, BLOCK, head=head, lo=-1000000, hi=1000000)
    y = _make_mixed_divisor(dtype, BLOCK)
    out = _run_compiled_case('rem_vec', x, BLOCK, y)
    zero_lanes = y == 0
    assert (out[zero_lanes] == _all_ones(dtype)).all(), \
        f"zero-divisor lanes must be exactly the all-ones value for {dtype}"
    assert (out[~zero_lanes] == _rem_expected(x, y)[~zero_lanes]).all(), \
        f"non-zero lanes must match C-truncation (np.fmod) for {dtype}"


@pytest.mark.parametrize('form', ['splat', 'literal'])
@pytest.mark.parametrize('dtype', ['int32', 'int64'])
@requires_verified_baseline_arch
def test_rem_zero_forms(dtype, form):
    x = _make_array(dtype, BLOCK, head=HEAD, lo=-1000000, hi=1000000)
    out = _run_compiled_case(f'rem_{form}_{dtype}', x, BLOCK)
    assert (out == _all_ones(dtype)).all(), \
        f"x % 0 ({form} divisor) must be the all-ones value for {dtype}"


# ---------------------------------------------------------------------------
# compiled device matrix: // by zero (core promise: saturating)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('dtype', ['int32', 'int64'])
@requires_verified_baseline_arch
def test_div_loaded_mixed(dtype):
    x = _make_array(dtype, BLOCK, head=HEAD, lo=-1000000, hi=1000000)
    y = _make_mixed_divisor(dtype, BLOCK)
    out = _run_compiled_case('div_vec', x, BLOCK, y)
    ref = _div_expected(x, y)
    zero_lanes = y == 0
    assert (out[zero_lanes] == ref[zero_lanes]).all(), \
        f"zero-divisor lanes must saturate to {{0, +MAX, -MIN}} for {dtype}"
    assert (out[~zero_lanes] == ref[~zero_lanes]).all(), \
        f"non-zero lanes must match truncated division for {dtype}"


@requires_verified_baseline_arch
def test_div_zero_forms_int32():
    x = _make_array('int32', BLOCK, head=HEAD, lo=-1000000, hi=1000000)
    ref = _div_expected(x, np.zeros(BLOCK, dtype=np.int32))
    for form in ('splat', 'literal'):
        out = _run_compiled_case(f'div_{form}_int32', x, BLOCK)
        assert (out == ref).all(), \
            f"x // 0 ({form} divisor) must saturate by dividend sign for int32"


# ---------------------------------------------------------------------------
# observed widths (recorded behaviour, inside the all-ones promise)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('dtype,head,lo,hi', [
    ('int8', HEAD8, -100, 100),
    ('int16', HEAD, -10000, 10000),
    ('uint8', HEADU8, 0, 256),
])
@requires_verified_baseline_arch
def test_rem_observed_widths(dtype, head, lo, hi):
    x = _make_array(dtype, SMALL, head=head, lo=lo, hi=hi)
    y = _make_mixed_divisor(dtype, SMALL)
    out = _run_compiled_case('rem_vec', x, SMALL, y)
    zero_lanes = y == 0
    assert (out[zero_lanes] == _all_ones(dtype)).all(), \
        f"zero-divisor lanes must be the all-ones value for {dtype}"
    assert (out[~zero_lanes] == _rem_expected(x, y)[~zero_lanes]).all(), \
        f"non-zero lanes must match C-truncation remainder for {dtype}"


@requires_verified_baseline_arch
def test_div_observed_uint8():
    x = _make_array('uint8', SMALL, head=HEADU8, lo=0, hi=256)
    y = _make_mixed_divisor('uint8', SMALL)
    out = _run_compiled_case('div_vec', x, SMALL, y)
    ref = _div_expected(x, y)
    assert (out == ref).all(), "uint8 // 0 must be {0, 255}, non-zero lanes truncated"


# ---------------------------------------------------------------------------
# interpreter parity (TRITON_INTERPRET=1 must match the compiled values)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('dtype', ['int32', 'int64'])
@requires_verified_baseline_arch
def test_interpreter_parity_rem_loaded(dtype):
    x = _make_array(dtype, BLOCK, head=HEAD, lo=-1000000, hi=1000000)
    y = _make_mixed_divisor(dtype, BLOCK)
    compiled = _run_compiled_case('rem_vec', x, BLOCK, y)
    interp = _run_interpreted_case('rem_vec', x, BLOCK, y)
    assert (interp == compiled).all(), f"interpreter must match compiled x % y for {dtype}"


@pytest.mark.parametrize('form', ['splat', 'literal'])
@pytest.mark.parametrize('dtype', ['int32', 'int64'])
@requires_verified_baseline_arch
def test_interpreter_parity_rem_zero_forms(dtype, form):
    x = _make_array(dtype, BLOCK, head=HEAD, lo=-1000000, hi=1000000)
    key = f'rem_{form}_{dtype}'
    compiled = _run_compiled_case(key, x, BLOCK)
    interp = _run_interpreted_case(key, x, BLOCK)
    assert (interp == compiled).all(), \
        f"interpreter must match compiled x % 0 ({form}) for {dtype}"


@pytest.mark.parametrize('dtype', ['int32', 'int64'])
@requires_verified_baseline_arch
def test_interpreter_parity_div_loaded(dtype):
    x = _make_array(dtype, BLOCK, head=HEAD, lo=-1000000, hi=1000000)
    y = _make_mixed_divisor(dtype, BLOCK)
    compiled = _run_compiled_case('div_vec', x, BLOCK, y)
    interp = _run_interpreted_case('div_vec', x, BLOCK, y)
    assert (interp == compiled).all(), f"interpreter must match compiled x // y for {dtype}"


@requires_verified_baseline_arch
def test_interpreter_parity_div_zero_forms_int32():
    x = _make_array('int32', BLOCK, head=HEAD, lo=-1000000, hi=1000000)
    for form in ('splat', 'literal'):
        key = f'div_{form}_int32'
        compiled = _run_compiled_case(key, x, BLOCK)
        interp = _run_interpreted_case(key, x, BLOCK)
        assert (interp == compiled).all(), \
            f"interpreter must match compiled x // 0 ({form}) for int32"


@pytest.mark.parametrize('dtype,head,lo,hi', [
    ('int8', HEAD8, -100, 100),
    ('int16', HEAD, -10000, 10000),
    ('uint8', HEADU8, 0, 256),
])
@requires_verified_baseline_arch
def test_interpreter_parity_rem_observed_widths(dtype, head, lo, hi):
    x = _make_array(dtype, SMALL, head=head, lo=lo, hi=hi)
    y = _make_mixed_divisor(dtype, SMALL)
    compiled = _run_compiled_case('rem_vec', x, SMALL, y)
    interp = _run_interpreted_case('rem_vec', x, SMALL, y)
    assert (interp == compiled).all(), f"interpreter must match compiled x % y for {dtype}"


# ---------------------------------------------------------------------------
# interpreter urem contract for widths the compiler cannot build yet
# ---------------------------------------------------------------------------


@pytest.mark.parametrize('key,dtype,signed', [
    ('urem_vec_u16', 'uint16', 'int16'),
    ('urem_vec_u32', 'uint32', 'int32'),
    ('urem_vec_u64', 'uint64', 'int64'),
])
def test_interpreter_urem_all_ones_contract(key, dtype, signed):
    x = _make_array(signed, SMALL, lo=None, hi=None)
    y = _make_mixed_divisor(signed, SMALL)
    interp = _run_interpreted_case(key, x, SMALL, y).astype(np.dtype(dtype))
    xu = x.astype(np.dtype(dtype))
    yu = y.astype(np.dtype(dtype))
    assert (interp[yu == 0] == _all_ones(dtype)).all(), \
        f"interpreter urem {dtype} zero-divisor lanes must be all-ones"
    assert (interp[yu != 0] == _rem_expected(xu, yu)[yu != 0]).all(), \
        f"interpreter urem {dtype} non-zero lanes must match C-truncation remainder"


# ---------------------------------------------------------------------------
# interpreter-only sanity checks (no device, no child process; always run)
# ---------------------------------------------------------------------------


def test_baseline_arch_gate_predicate():
    # The arch gate is pure string logic, so the skip path (foreign SoC,
    # unresolvable target) is checkable on any host, including the verified
    # host where the gate is open and the device cases run.
    assert _is_verified_baseline_arch("Ascend910B4")
    assert _is_verified_baseline_arch("Ascend910B3")
    for foreign in ("Ascend910_95", "Ascend950DT", "Ascend910A", "Ascend310B6", "", None):
        assert not _is_verified_baseline_arch(foreign)


def test_interpreter_scalar_zero_rhs_direct():
    # create_srem/create_urem must fold a 0-dim scalar zero divisor to the
    # all-ones value exactly like a splat tensor does. Direct numpy-level
    # call: no kernel launch, no device, no child process.
    from triton.runtime.interpreter import TensorHandle
    from triton.backends.ascend.ascend_interpreter import AscendInterpreterBuilder
    builder = AscendInterpreterBuilder()
    x32 = TensorHandle(np.array([12, -12, 100, -1], dtype=np.int32), tl.int32)
    zero32 = TensorHandle(np.array(0, dtype=np.int32), tl.int32)
    with np.errstate(divide='ignore', invalid='ignore'):
        srem = builder.create_srem(x32, zero32)
    assert (srem.data == _all_ones('int32')).all(), \
        "0-dim scalar zero divisor must give the all-ones value like a splat"
    x8 = TensorHandle(np.array([12, 100, 3, 7], dtype=np.uint8), tl.uint8)
    zero8 = TensorHandle(np.array(0, dtype=np.uint8), tl.uint8)
    with np.errstate(divide='ignore', invalid='ignore'):
        urem = builder.create_urem(x8, zero8)
    assert (urem.data == _all_ones('uint8')).all(), \
        "unsigned 0-dim scalar zero divisor must give 2**8 - 1"


# ---------------------------------------------------------------------------
# known gaps, pinned as xfail (keep the suite honest; do not silently skip)
# ---------------------------------------------------------------------------


@pytest.mark.xfail(
    strict=False, reason="Known gap: unsigned remainder wider than 8 bits fails to compile "
    "on Ascend 910B / CANN 9.2.0 - u16 hits hivm.hir.vcast "
    "'uint16_t_to_float_truncmode' unsupported, u32/u64 hit 'failed to "
    "legalize operation hivm.hir.vmodui'. Follow-up: lower urem in the "
    "triton-ascend pipeline. See issue #2518.")
@pytest.mark.parametrize('key,dtype', [
    ('urem_vec_u16', 'uint16'),
    ('urem_vec_u32', 'uint32'),
    ('urem_vec_u64', 'uint64'),
])
@requires_verified_baseline_arch
def test_urem_wide_widths(key, dtype):
    signed = {'uint16': 'int16', 'uint32': 'int32', 'uint64': 'int64'}[dtype]
    x = _make_array(signed, SMALL, lo=None, hi=None)
    y = _make_mixed_divisor(signed, SMALL)
    out = _run_compiled_case(key, x, SMALL, y).astype(np.dtype(dtype))
    xu = x.astype(np.dtype(dtype))
    yu = y.astype(np.dtype(dtype))
    assert (out == _rem_expected(xu, yu)).all(), \
        f"wide unsigned remainder must follow the all-ones contract for {dtype}"


@pytest.mark.xfail(
    strict=False, reason="Known gap: sub-32-bit x // 0 goes through the float-decomposition "
    "path on Ascend 910B / CANN 9.2.0 and returns {-1, 0} instead of the "
    "saturating {0, +MAX, -MIN}; the //0 promise therefore covers i32 "
    "and i64 only. See issue #2518.")
@pytest.mark.parametrize('dtype', ['int8', 'int16'])
@requires_verified_baseline_arch
def test_div_zero_saturates_small_widths(dtype):
    head = HEAD8 if dtype == 'int8' else HEAD
    lo, hi = (-100, 100) if dtype == 'int8' else (-10000, 10000)
    x = _make_array(dtype, SMALL, head=head, lo=lo, hi=hi)
    y = np.zeros(SMALL, dtype=np.dtype(dtype))
    out = _run_compiled_case('div_vec', x, SMALL, y)
    ref = _div_expected(x, y)
    assert (out == ref).all(), f"i8/i16 // 0 should saturate but does not: {dtype}"


@pytest.mark.xfail(
    strict=False, reason="Known gap: x // 0 with a splat/literal zero divisor returns the "
    "dividend for int64 on Ascend 910B / CANN 9.2.0 (the saturating "
    "result is produced for a loaded zero divisor only). The //0 "
    "promise covers all divisor forms for i32 and the loaded divisor "
    "for i64. See issue #2518.")
@pytest.mark.parametrize('form', ['splat', 'literal'])
@requires_verified_baseline_arch
def test_div_zero_forms_int64_saturate(form):
    x = _make_array('int64', BLOCK, head=HEAD, lo=-1000000, hi=1000000)
    out = _run_compiled_case(f'div_{form}_int64', x, BLOCK)
    ref = _div_expected(x, np.zeros(BLOCK, dtype=np.int64))
    assert (out == ref).all(), \
        f"x // 0 ({form} divisor) should saturate for int64 but returns the dividend"


if __name__ == "__main__":
    if "--case" in sys.argv:
        _child_main(sys.argv[sys.argv.index("--case") + 1], sys.argv[sys.argv.index("--io") + 1],
                    interpret="--interpret" in sys.argv)
        sys.exit(0)
    sys.exit(pytest.main([__file__, "-v"]))
