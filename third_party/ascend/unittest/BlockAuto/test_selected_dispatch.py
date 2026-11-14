# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Selected CV bundles keep runtime safety checks without a second performance veto."""
import importlib
from pathlib import Path
import sys
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch


class TestSelectedDispatch(unittest.TestCase):

    def setUp(self):
        package = ModuleType('_selected_dispatch_test')
        package.__path__ = [str(Path(__file__).resolve().parents[2] / 'backend')]
        self.patcher = patch.dict(sys.modules, {package.__name__: package})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.batching = importlib.import_module(package.__name__ + '.block_batching')
        self.events = []
        self.guard = Mock(return_value=(True, 'PROVEN_AT_RUNTIME'))

        def kernel(role):
            return SimpleNamespace(hash=role, metadata=SimpleNamespace(_asdict=lambda: {}), function=role,
                                   packed_metadata=role, launch_metadata=lambda *args: None,
                                   run=lambda *args: self.events.append(args[4]))

        self.baseline, self.candidate = kernel('baseline'), kernel('candidate')
        guard = importlib.import_module(package.__name__ + '.block_batching_guard')
        make_guard = patch.object(guard, 'RuntimeGuard', return_value=self.guard)
        make_guard.start()
        self.addCleanup(make_guard.stop)
        self.bundle = self.make_bundle('independent_batch_v1')
        self.knobs = SimpleNamespace(ascend=SimpleNamespace(compile_only=False))

    def make_bundle(self, cv_mode):
        report = dict(cv_mode=cv_mode, selected_axis=0, lane_width=2, guard={}, owner='block_batching')
        bundle = self.batching.BatchKernel(self.baseline, self.candidate, self.batching.BatchRequest(2, 0), report, [0],
                                           publish_manifest=False)
        bundle.block_auto_selection = dict(model_complete=True, selected_factor=2)
        return bundle

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
        self.bundle = self.make_bundle('disabled')
        self.assertEqual(self.dispatch()['reason'], 'GRID_BELOW_CORE_FLOOR')
        self.assertEqual(self.events, ['baseline', 'baseline'])

    def test_selected_cv_still_checks_aliases(self):
        self.guard.return_value = (False, 'WRITE_ALIAS')
        self.assertEqual(self.dispatch()['reason'], 'WRITE_ALIAS')
        self.assertEqual(self.events, ['baseline'])

    def test_selected_cv_still_checks_divisibility_and_capture(self):
        self.assertEqual(self.dispatch(grid=3)['reason'], 'CV_GRID_NOT_DIVISIBLE')
        self.assertEqual(self.dispatch(capture=True)['reason'], 'CAPTURE_BASELINE')
        self.assertEqual(self.events, ['baseline', 'baseline'])
