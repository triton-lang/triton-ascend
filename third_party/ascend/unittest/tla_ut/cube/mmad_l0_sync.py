"""A5 standalone MMAD: automatic intra-core sync versus explicit events.

The current A5 TLA pipeline always uses GSS when auto-injection is enabled.
The launch option sync_solver=False does NOT turn auto-injection off. This
test uses a scoped subprocess proxy to add the real NPUIR switch to the
actual compilation, without changing product code. Auto cases contain no
explicit events; manual cases contain events and disable auto-injection.
There is no device mode with both sources of synchronization disabled.

TLA_SYNC_DUMP_DIR optionally selects an artifact root. Otherwise artifacts
are retained in a new system temporary directory. Each invocation gets a
fresh JIT function and Triton disk cache, so the command cannot be hidden by
a cache hit. Input IR, post-sync IR, command, logs and binary are retained.
"""
import json
import os
import re
import shlex
import shutil
import sys
import tempfile
from contextlib import contextmanager
from functools import partial
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import triton.language as tl
import triton.language.extra.tla as tla

from utils.tla_test_utils import (
    Options,
    check_kernel_binary,
    compile_offline,
    lower_offline,
    matrix,
)


class A5Options(Options):
    arch = "Ascend950PR_9589"


# Below/at/above the CATLASS PIPE_M barrier threshold of 10 output fractals.
# The fourth case stresses repeated single-slot reuse across many K slices.
CONFIGS = [
    dict(name=f"{shape}_{mode}", M=m, N=n, K=k, L0_K=32, SLOTS=slots, mode=mode) for shape, m, n, k, slots in (
        ("fractals9_single", 48, 48, 128, 1),
        ("fractals10_pingpong", 32, 80, 192, 2),
        ("fractals12_pingpong", 48, 64, 192, 2),
        ("long_k_single", 32, 64, 512, 1),
    ) for mode in ("auto_v1", "auto_v2", "manual")
]


@partial(tla.jit, do_not_specialize=["Clear"])
def sync_mmad(A, B, First, Final, Clear, M: tl.constexpr, N: tl.constexpr, K: tl.constexpr, L0_K: tl.constexpr,
              SLOTS: tl.constexpr, MANUAL: tl.constexpr):
    """Store P=A@B, then reuse C to compute P (clear) or 2P (accumulate)."""
    tl.static_assert(M % 16 == 0 and N % 16 == 0 and L0_K % 16 == 0)
    tl.static_assert(K % L0_K == 0 and (SLOTS == 1 or SLOTS == 2))
    tl.static_assert(SLOTS * 2 * M * L0_K <= 64 * 1024)
    tl.static_assert(SLOTS * 2 * N * L0_K <= 64 * 1024)
    tl.static_assert(4 * M * N <= 256 * 1024)
    tl.static_assert(2 * K * (M + N) <= 512 * 1024)
    with tla.scope.cube():
        a_l1 = tla.alloc((K // 16, M // 16, 16, 16), tl.float16, space="l1")
        b_l1 = tla.alloc((N // 16, K // 16, 16, 16), tl.float16, space="l1")
        a0 = tla.alloc((L0_K // 16, M // 16, 16, 16), tl.float16, space="l0a")
        b0 = tla.alloc((L0_K // 16, N // 16, 16, 16), tl.float16, space="l0b")
        if SLOTS == 2:
            a1 = tla.alloc((L0_K // 16, M // 16, 16, 16), tl.float16, space="l0a")
            b1 = tla.alloc((L0_K // 16, N // 16, 16, 16), tl.float16, space="l0b")
        acc = tla.alloc((N // 16, M // 16, 16, 16), tl.float32, space="l0c")
        tla.load_gm_to_l1(tla.view(A, (M, K)), a_l1)
        tla.load_gm_to_l1(tla.view(B, (K, N)), b_l1)
        if MANUAL:
            tla.sync.set_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
            tla.sync.wait_flag(tla.sync.PIPE.MTE2, tla.sync.PIPE.MTE1, 0)
            tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 0)
            if SLOTS == 2:
                tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 1)
        for pass_id in tl.static_range(2):
            for ki in tl.static_range(K // L0_K):
                if ki % SLOTS == 0:
                    a_tile = a0
                    b_tile = b0
                else:
                    a_tile = a1
                    b_tile = b1
                if MANUAL:
                    tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, ki % SLOTS)
                tla.copy_l1_to_l0(
                    a_l1, a_tile,
                    tla.CopyL1ToL0Params(is_transpose=False, k_part_idx=ki, k_part=L0_K, k_part_ceil=L0_K,
                                         k_part_loop=K // L0_K, k_part_actual=L0_K, m=M))
                tla.copy_l1_to_l0(
                    b_l1, b_tile,
                    tla.CopyL1ToL0Params(is_transpose=False, k_part_idx=ki, k_part=L0_K, k_part_ceil=L0_K,
                                         k_part_loop=K // L0_K, k_part_actual=L0_K, m=N))
                if MANUAL:
                    tla.sync.set_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
                    tla.sync.wait_flag(tla.sync.PIPE.MTE1, tla.sync.PIPE.M, 0)
                if pass_id == 1 and ki == 0:
                    tla.cube.mmad_l0(a_tile, b_tile, acc, m=M, n=N, k=L0_K, clear_accum=Clear,
                                     unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
                else:
                    tla.cube.mmad_l0(a_tile, b_tile, acc, m=M, n=N, k=L0_K, clear_accum=(ki == 0),
                                     unit_flag_ctrl=tla.UnitFlagMode.DISABLED)
                if MANUAL:
                    tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, ki % SLOTS)
            if MANUAL:
                tla.sync.set_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
                tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.F, 0)
            if pass_id == 0:
                tla.store_l0c_to_gm(
                    acc, tla.view(First, (M, N)),
                    tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            unit_flag_ctrl=tla.UnitFlagMode.DISABLED))
                if MANUAL:
                    # FIX must finish reading old C before the next MMAD writes it.
                    tla.sync.set_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, 0)
                    tla.sync.wait_flag(tla.sync.PIPE.F, tla.sync.PIPE.M, 0)
            else:
                tla.store_l0c_to_gm(
                    acc, tla.view(Final, (M, N)),
                    tla.FixpipeL0C2GMParams(dma_mode=tla.FixpipeDMAMode.NZ2ND,
                                            unit_flag_ctrl=tla.UnitFlagMode.DISABLED))
        if MANUAL:
            tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 0)
            if SLOTS == 2:
                tla.sync.wait_flag(tla.sync.PIPE.M, tla.sync.PIPE.MTE1, 1)
            tla.sync.barrier(tla.sync.PIPE.ALL)


def _constants(case):
    values = {key: case[key] for key in ("M", "N", "K", "L0_K", "SLOTS")}
    return dict(values, MANUAL=case["mode"] == "manual")


def _artifact_dir(case, stage):
    root = os.environ.get("TLA_SYNC_DUMP_DIR")
    if root:
        Path(root).mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix=f"mmadl0_{stage}_{case['name']}_", dir=root)).resolve()
    print("ARTIFACTS:", directory, flush=True)
    return directory


def _events(text, kind):
    return re.findall(rf"hivm\.hir\.{kind}_flag\[<(PIPE_\w+)>, <(PIPE_\w+)>, <(EVENT_ID\d+)>\]", text)


def _check_events(text):
    sets, waits = set(_events(text, "set")), set(_events(text, "wait"))
    for source, target in (("PIPE_MTE2", "PIPE_MTE1"), ("PIPE_MTE1", "PIPE_M"), ("PIPE_M", "PIPE_MTE1"),
                           ("PIPE_M", "PIPE_FIX"), ("PIPE_FIX", "PIPE_M")):
        assert any((s, t) == (source, target)
                   for s, t, _ in sets & waits), (f"missing matching set/wait for {source}->{target}")


def _check_input(text, case):
    mmads = re.findall(r"hivm\.hir\.mmadL0[^\n]*", text)
    tiles = case["K"] // case["L0_K"]
    assert len(mmads) == 2 * tiles
    assert text.count("hivm.hir.l12l0 ") == 4 * tiles
    assert text.count("hivm.hir.fixpipe ") == 2
    assert all("unit_flag<DISABLED>" in op and " bias(" not in op for op in mmads)
    dynamic_i1 = set(re.findall(r"(%[\w]+): i1\b", text))
    operands = re.search(r"ins\((.*?) :", mmads[tiles]).group(1).split(", ")
    assert operands[7] in dynamic_i1, "second-pass clear must stay a runtime i1"
    assert "hivm.hir.pipe_barrier[<PIPE_M>]" not in text
    if case["mode"] == "manual":
        _check_events(text)
        assert "hivm.hir.pipe_barrier[<PIPE_ALL>]" in text
    else:
        for name in ("set_flag", "wait_flag", "pipe_barrier"):
            assert f"hivm.hir.{name}" not in text, "automatic input must not supply sync"


class _CompilerProxy:
    """Delegate all subprocess APIs, modifying only this TLA compiler command."""

    def __init__(self, real, case, directory):
        self.real, self.case, self.directory = real, case, directory
        self.commands = []

    def __getattr__(self, name):
        return getattr(self.real, name)

    def run(self, command, *args, **kwargs):
        if (not isinstance(command, (list, tuple)) or not command or Path(command[0]).name != "bishengir-compile"
                or "--enable-tla-mode=true" not in command):
            return self.real.run(command, *args, **kwargs)
        # Avoid duplicate LLVM command-line options from debug or future backends.
        keys = ("--disable-hivm-auto-inject-sync", "--hivm-sync-solver-version", "--enable-hivm-unit-flag-sync",
                "--save-temps")
        for key in keys:
            assert key not in command, f"unexpected separate-value option {key}"
        command = [arg for arg in command if not any(arg.startswith(key + "=") for key in keys)]
        manual = self.case["mode"] == "manual"
        version = "v1" if self.case["mode"] == "auto_v1" else "v2"
        command += [
            f"--disable-hivm-auto-inject-sync={str(manual).lower()}", f"--hivm-sync-solver-version={version}",
            "--enable-hivm-unit-flag-sync=true", f"--save-temps={self.directory / 'temps'}"
        ]
        assert not self.commands, "unexpected recompilation of the same test specialization"
        self.commands.append(command)
        shutil.copyfile(command[1], self.directory / "input.mlir")
        _check_input((self.directory / "input.mlir").read_text(), self.case)
        (self.directory / "command.json").write_text(json.dumps(command, indent=2))
        (self.directory / "command.txt").write_text(shlex.join(command) + "\n")
        try:
            result = self.real.run(command, *args, **kwargs)
        except self.real.CalledProcessError as error:
            self._logs(error)
            raise
        self._logs(result)
        return result

    def _logs(self, result):
        for name in ("stdout", "stderr"):
            text = getattr(result, name, None) or b""
            if isinstance(text, bytes):
                text = text.decode("utf-8", errors="replace")
            (self.directory / f"compiler.{name}.txt").write_text(text, encoding="utf-8")


@contextmanager
def _compile_session(case, stage):
    import triton.backends.ascend.compiler as backend
    from triton import knobs
    from triton.language.extra.tla.jit import TLAJITFunction
    from triton.runtime import driver

    target = driver.active.get_current_target()
    assert backend._is_a5_target_arch(target.arch), f"requires A5, got {target}"
    directory = _artifact_dir(case, stage)
    (directory / "temps").mkdir()
    (directory / "case.json").write_text(json.dumps(dict(case, target=str(target)), indent=2))
    proxy = _CompilerProxy(backend.subprocess, case, directory)
    # A fresh JIT object avoids in-process reuse; the unique disk cache avoids
    # reusing a binary compiled under different NPUIR sync options.
    fn = TLAJITFunction(sync_mmad.fn, do_not_specialize=["Clear"])
    with knobs.cache.scope(), knobs.compilation.scope(), patch.object(backend, "subprocess", proxy):
        knobs.cache.dir = str(directory / "cache")
        knobs.compilation.always_compile = True
        knobs.compilation.override = False
        yield fn, proxy, directory


def _check_compiled(kernel, proxy, directory):
    assert len(proxy.commands) == 1, "no real TLA bishengir-compile call observed"
    check_kernel_binary(kernel)
    (directory / "kernel.npubin").write_bytes(kernel.asm["npubin"])
    path = directory / "temps" / "module.hivm.opt.mlir"
    assert path.is_file(), "missing post-sync IR; check this toolchain's save-temps filenames"
    text = path.read_text()
    assert "call @mmadl0_a5_half_to_float(" in text, "missing standalone MMAD lowering"
    assert "hivm.hir.mmadL0" not in text
    _check_events(text)
    assert "hivm.hir.pipe_barrier[<PIPE_ALL>]" in text
    if proxy.case["mode"] == "manual":
        source = (directory / "input.mlir").read_text()
        for kind in ("set", "wait"):
            assert sorted(_events(source, kind)) == sorted(_events(
                text, kind)), ("auto-injection disabled: explicit event multiset unexpectedly changed")


@matrix(CONFIGS)
def test_ir(case, output_dir=None):
    directory = Path(output_dir) / case["name"] if output_dir else _artifact_dir(case, "ir")
    module, ctx = compile_offline(sync_mmad, dict(A="*fp16", B="*fp16", First="*fp32", Final="*fp32", Clear="i1"),
                                  _constants(case), output_dir=directory, options=A5Options())
    _check_input(str(lower_offline(module, ctx, output_dir=directory)), case)
    print("INPUT IR PASS:", case["name"], "(before NPUIR auto-sync)")


def _buffers(case):
    import torch
    x = torch.empty((case["M"], case["K"]), dtype=torch.float16, device="npu")
    y = torch.empty((case["K"], case["N"]), dtype=torch.float16, device="npu")
    # 1024-byte guards preserve alignment and detect out-of-range GM stores.
    parents = [
        torch.full((case["M"] * case["N"] + 512, ), -8191.0, dtype=torch.float32, device="npu") for _ in range(2)
    ]
    outputs = [parent[256:-256].view(case["M"], case["N"]) for parent in parents]
    return x, y, outputs, parents


@matrix(CONFIGS)
def test_compile(case):
    import torch_npu  # noqa: F401
    x, y, (first, final), _ = _buffers(case)
    with _compile_session(case, "compile") as (fn, proxy, directory):
        kernel = fn.warmup(x, y, first, final, True, grid=(1, ), **_constants(case), unit_flag=True)
        _check_compiled(kernel, proxy, directory)


@matrix(CONFIGS)
def test_run(case):
    import torch
    import torch_npu  # noqa: F401
    x, y, (first, final), parents = _buffers(case)
    with _compile_session(case, "run") as (fn, proxy, directory):
        kernel = fn.warmup(x, y, first, final, True, grid=(1, ), **_constants(case), unit_flag=True)
        _check_compiled(kernel, proxy, directory)
        # Dyadic FP16 data yields exact FP32 products/sums for these tile sizes.
        # Repeated launches change inputs and runtime clear while sharing a binary.
        for seed in (7, 19, 31):
            generator = torch.Generator().manual_seed(seed)
            a = (torch.randint(-4, 5, x.shape, generator=generator).float() / 8).half()
            b = (torch.randint(-4, 5, y.shape, generator=generator).float() / 8).half()
            x.copy_(a)
            y.copy_(b)
            product = a.float() @ b.float()
            for clear in (True, False, True, False):
                first.fill_(float("nan"))
                final.fill_(float("nan"))
                print("RUN:", case["name"], "seed", seed, "clear", clear, flush=True)
                launched = fn[(1, )](x, y, first, final, clear, **_constants(case), unit_flag=True)
                torch.npu.synchronize()
                assert launched.asm["npubin"] == kernel.asm["npubin"]
                torch.testing.assert_close(first.cpu(), product, rtol=0, atol=0)
                torch.testing.assert_close(final.cpu(), product if clear else 2 * product, rtol=0, atol=0)
                for parent in parents:
                    host = parent.cpu()
                    assert (host[:256] == -8191.0).all() and (host[-256:] == -8191.0).all()
        assert len(proxy.commands) == 1, "runtime clear unexpectedly caused recompilation"
        (directory / "run-passed.txt").write_text("12 launches passed both outputs and GM guards\n")
