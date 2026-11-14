# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Selected CV bundles keep runtime safety checks without a second performance veto."""
import importlib
from pathlib import Path
import sys
import threading
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch


class TestSelectedDispatch(unittest.TestCase):

    def setUp(self):
        package = ModuleType('_selected_dispatch_test')
        package.__path__ = [str(Path(__file__).resolve().parents[2] / 'backend')]
        self.patcher = patch.dict(sys.modules, {package.__name__: package})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.batching = importlib.import_module(package.__name__ + '.block_batching')
        self.events = []
        self.bundle = object.__new__(self.batching.BatchKernel)
        for role in ('baseline', 'candidate'):
            setattr(
                self.bundle, role,
                SimpleNamespace(function=role, packed_metadata=role, launch_metadata=lambda *args: None,
                                run=lambda *args: self.events.append(args[4])))
        self.bundle._primary = self.bundle.candidate
        self.bundle.request = self.batching.BatchRequest(2, 0)
        self.bundle.block_batching = dict(cv_mode='independent_batch_v1', selected_axis=0)
        self.bundle.block_auto_selection = dict(model_complete=True, selected_factor=2)
        self.bundle._local = threading.local()
        self.bundle._preserve_grid = False
        self.bundle.positions, self.bundle.hash = [0], 'bundle'
        self.bundle._guard = lambda args, grid: (True, 'PROVEN_AT_RUNTIME')
        self.knobs = SimpleNamespace(ascend=SimpleNamespace(compile_only=False))

    def dispatch(self, grid=4, capture=False):
        triton = ModuleType('triton')
        triton.knobs = self.knobs
        with patch.dict(sys.modules, {'triton': triton}), \
             patch.object(self.batching, '_capture_state', return_value=capture), \
             patch.object(self.batching, '_core_floor', return_value=56):
            self.bundle._dispatch(grid, 1, 1, 0, None, None, None, None, None, 1234)
        return self.bundle.last_launch

    def test_automatic_cv_uses_the_scored_small_grid_candidate(self):
        launch = self.dispatch()
        self.assertEqual(self.events, ['candidate'])
        self.assertEqual(launch['applied_factor'], 2)

    def test_manual_cv_and_automatic_vv_retain_the_existing_floor(self):
        del self.bundle.block_auto_selection
        self.assertEqual(self.dispatch()['reason'], 'GRID_BELOW_CORE_FLOOR')
        self.bundle.block_auto_selection = dict(selected_factor=2)
        self.bundle.block_batching['cv_mode'] = 'disabled'
        self.assertEqual(self.dispatch()['reason'], 'GRID_BELOW_CORE_FLOOR')
        self.assertEqual(self.events, ['baseline', 'baseline'])

    def test_selected_cv_still_checks_aliases(self):
        self.bundle._guard = lambda args, grid: (False, 'WRITE_ALIAS')
        self.assertEqual(self.dispatch()['reason'], 'WRITE_ALIAS')
        self.assertEqual(self.events, ['baseline'])

    def test_selected_cv_still_checks_divisibility_and_capture(self):
        self.assertEqual(self.dispatch(grid=3)['reason'], 'CV_GRID_NOT_DIVISIBLE')
        self.assertEqual(self.dispatch(capture=True)['reason'], 'CAPTURE_BASELINE')
        self.assertEqual(self.events, ['baseline', 'baseline'])
