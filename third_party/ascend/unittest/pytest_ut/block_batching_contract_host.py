# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Isolated host launch fixtures for the shared execution contract."""
import importlib
from pathlib import Path
import sys
import types
import pytest

PACKAGE = "_block_batching_host_test"
if PACKAGE not in sys.modules:
    package = types.ModuleType(PACKAGE)
    package.__path__ = [str(Path(__file__).parents[2] / "backend")]
    sys.modules[PACKAGE] = package
guard_module = importlib.import_module(PACKAGE + ".block_batching_guard")
report_module = importlib.import_module(PACKAGE + ".block_batching_report")
batching = importlib.import_module(PACKAGE + ".block_batching")
check_guard = guard_module.check_guard


def arg(index, bits=64):
    return ["arg", index, bits]


def const(value, bits=64):
    return ["const", value, bits]


def pid(axis):
    return ["pid", axis, 32]


def add(a, b, bits=64):
    return ["addi", bits, a, b]


def mul(a, b, bits=64):
    return ["muli", bits, a, b]


def rows_guard(block=8, column_axis=1):
    lane = ["range", 0, block, 32]
    column = lane if column_axis is None else add(mul(pid(column_axis), const(block)), lane)
    offset = add(mul(pid(0), arg(3)), column)
    return {
        "version":
        1, "axis":
        0, "accesses": [
            {"root": 0, "offset": offset, "mask": ["lt", column, arg(2)], "write": False, "bytes": 4},
            {"root": 1, "offset": offset, "mask": ["lt", column, arg(2)], "write": True, "bytes": 4},
        ]
    }


@pytest.fixture
def bundle_environment(monkeypatch):
    from collections import namedtuple
    modules = {}
    for name in ('triton', 'triton.runtime', 'triton.runtime.cache', 'triton.runtime.errors', 'triton.compiler',
                 'triton.compiler.errors'):
        module = types.ModuleType(name)
        module.__path__ = []
        monkeypatch.setitem(sys.modules, name, module)
        modules[name] = module
    knobs = types.SimpleNamespace(ascend=types.SimpleNamespace(compile_only=False))
    modules['triton'].knobs = knobs

    class Cache:

        def put(self, data, name, **kwargs):
            return '/tmp/' + name

    modules['triton.runtime.cache'].get_cache_manager = lambda key: Cache()
    modules['triton.compiler.errors'].MLIRCompilationError = type('MLIRCompilationError', (RuntimeError, ), {})
    modules['triton.runtime.errors'].OutOfResources = type('OutOfResources', (RuntimeError, ), {})
    utils = types.ModuleType(PACKAGE + '.utils')
    utils.backend_policy = 'torch_npu'
    monkeypatch.setitem(sys.modules, PACKAGE + '.utils', utils)
    monkeypatch.setattr(batching, '_capture_state', lambda: False)
    monkeypatch.setattr(batching, '_core_floor', lambda: 0)
    monkeypatch.setattr(batching, '_freeze_source', lambda backend, src, opts, revision: src)
    launches = []

    class Kernel:

        def __init__(self, name, report=None):
            self.name = self.hash = name
            fields = {'name': name}
            if report is not None: fields['block_batching'] = report
            self.metadata = namedtuple('Metadata', fields)(**fields)
            self.function = 101 if name == 'baseline' else 202
            self.packed_metadata = {'hash': name}
            self._run = None
            self.initializations = 0

        def _init_handles(self):
            if self._run is None:
                self.initializations += 1
                self._run = True

        def launch_metadata(self, grid, stream, *args):
            return {'selected': self.name}

        def run(self, *launch):
            assert launch[4] == self.function
            assert launch[5]['hash'] == self.name
            assert launch[6]['selected'] == self.name
            if not knobs.ascend.compile_only: launches.append((self.name, launch))

    report = {
        'compile_status': 'COMPILED_GUARDED', 'selected_axis': 0, 'owner': 'block_batching', 'guard':
        rows_guard(4, None), 'candidate_factor': 3, 'implementation_version': batching.IMPLEMENTATION_VERSION,
        'lane_width': 3, 'cv_mode': 'disabled'
    }
    base, candidate = Kernel('baseline'), Kernel('candidate', report)
    wrapper = batching.BatchKernel(base, candidate, batching.BatchRequest(3, 0), report, [0, 1, 2, 3])
    launch = (2, 1, 1, 42, -999, {}, None, None, None, 10000, 100000, 4, 4)
    return types.SimpleNamespace(wrapper=wrapper, base=base, candidate=candidate, report=report, launch=launch,
                                 launches=launches, knobs=knobs, Kernel=Kernel)
