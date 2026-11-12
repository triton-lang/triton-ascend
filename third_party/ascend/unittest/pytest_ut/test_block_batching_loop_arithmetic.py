"""Compile-time loop arithmetic: production helpers under UBSan and real TTIR."""
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess

import pytest
from triton._C.libtriton import ascend
from test_block_batching_rewrite import parse

pytestmark = pytest.mark.backend('native')


@pytest.fixture(scope='session', autouse=True)
def assign_npu():
    yield


def test_production_loop_helpers_under_ubsan(tmp_path):
    ascend_root = Path(__file__).resolve().parents[2]
    repo = ascend_root.parents[1]
    llvm_config = shutil.which('llvm-config')
    llvm_root = os.environ.get('LLVM_SYSPATH')
    if llvm_root:
        llvm_config = str(Path(llvm_root) / 'bin/llvm-config')
    if not llvm_config:
        for cache in (repo / 'build').glob('*/CMakeCache.txt'):
            match = re.search(r'^LLVM_DIR:[^=]+=(.+)/lib/cmake/llvm$', cache.read_text(), re.M)
            if match:
                llvm_config = str(Path(match[1]) / 'bin/llvm-config')
                break
    compiler = os.environ.get('CXX', 'g++')
    if not llvm_config or not Path(llvm_config).is_file() or not shutil.which(compiler):
        pytest.skip('UBSan helper test needs a C++ compiler and LLVM development installation')
    include = subprocess.check_output([llvm_config, '--includedir'], text=True).strip()
    libs = shlex.split(
        subprocess.check_output([llvm_config, '--ldflags', '--libs', 'support', '--system-libs'], text=True))
    source = ascend_root / 'lib/TritonToGraph/BlockBatching'
    executable = tmp_path / 'loop_arithmetic_ubsan'
    command = [
        compiler, '-std=c++17', '-O1', '-g', '-fsanitize=undefined', '-fno-sanitize-recover=all',
        '-I' + str(ascend_root / 'include'), '-I' + include,
        str(ascend_root / 'unittest/BlockBatching/LoopArithmeticTest.cpp')
    ]
    command += [
        str(source / name)
        for name in ('BatchExpr.cpp', 'BatchAffine.cpp', 'BatchRangeAnalysis.cpp', 'BatchLoopSchedule.cpp')
    ]
    subprocess.run(command + libs + ['-o', str(executable)], check=True, capture_output=True, text=True)
    result = subprocess.run([str(executable)], check=True, capture_output=True, text=True)
    assert 'regressions passed' in result.stdout
    assert not result.stderr


def constant_recurrence(lower, upper, step, bits):
    return f'''module {{
  tt.func public @static_recurrence(%x: !tt.ptr<f32>, %o: !tt.ptr<f32>, %n: i{bits}) {{
    %p = tt.get_program_id x : i32
    %xp = tt.addptr %x, %p : !tt.ptr<f32>, i32
    %op = tt.addptr %o, %p : !tt.ptr<f32>, i32
    %lo = arith.constant {lower} : i{bits}
    %hi = arith.constant {upper if upper is not None else 16} : i{bits}
    %step = arith.constant {step} : i{bits}
    %zero = arith.constant 0.0 : f32
    %sum = scf.for %iv = %lo to {'%hi' if upper is not None else '%n'} step %step
        iter_args(%acc = %zero) -> (f32) : i{bits} {{
      %v = tt.load %xp : !tt.ptr<f32>
      %next = arith.addf %acc, %v : f32
      scf.yield %next : f32
    }}
    tt.store %op, %sum : !tt.ptr<f32>
    tt.return
  }}
}}'''


@pytest.mark.parametrize('upper', [0, 1, 2, 3, 4, 5])
def test_static_group_and_epilogue_schedule(tmp_path, upper):
    module = parse(tmp_path, constant_recurrence(0, upper, 1, 32))
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert (report['vv_lowering'] == 'sequential_iterations_v1') == (upper >= 2)
    assert module.verify()


@pytest.mark.parametrize('lower,upper,step,bits', [
    (0, 8_000_000_000_000_000_000, 4_000_000_000_000_000_000, 64),
    (-(1 << 63), (1 << 63) - 1, 1, 64),
    (0, (1 << 31) - 1, 1 << 30, 32),
    (0, None, 1 << 30, 32),
    ((1 << 63) - 4, (1 << 63) - 1, 1, 64),
])
def test_static_unsafe_arithmetic_never_selects_iteration_groups(tmp_path, lower, upper, step, bits):
    module = parse(tmp_path, constant_recurrence(lower, upper, step, bits))
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0)
    assert report.get('vv_lowering') != 'sequential_iterations_v1', report
    assert module.verify()


def test_dynamic_bounds_still_select_guarded_iteration_groups(tmp_path):
    module = parse(tmp_path, constant_recurrence(0, None, 1, 32))
    report = ascend.passes.ttir.rewrite_block_batching(module, 2, 0)
    assert report['compile_status'] == 'REWRITTEN', report
    assert report['vv_lowering'] == 'sequential_iterations_v1'
    assert module.verify()
