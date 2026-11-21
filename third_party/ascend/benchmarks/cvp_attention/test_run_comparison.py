import copy
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import types
import unittest

from run_comparison import SCHEDULE, infer_csv_device, profile_block, qualified_native, qualified_recipe, select_block


class Recorder:

    def __init__(self):
        self.active = False
        self.calls = []
        self.syncs = 0
        self.sleeps = []
        self.steps = 0
        self.arguments = None
        self.experimental = None
        self.schedule = None
        self.torch = types.SimpleNamespace(npu=types.SimpleNamespace(synchronize=self.sync))
        self.profiler = types.SimpleNamespace(AiCMetrics=types.SimpleNamespace(PipeUtilization='pipe'),
                                              ProfilerLevel=types.SimpleNamespace(Level1='level1'),
                                              ProfilerActivity=types.SimpleNamespace(NPU='npu'),
                                              _ExperimentalConfig=self.config, schedule=self.make_schedule,
                                              tensorboard_trace_handler=lambda path: ('handler', path),
                                              profile=self.profile)
        self.npu = types.SimpleNamespace(profiler=self.profiler)

    def sync(self):
        self.syncs += 1

    def config(self, **kwargs):
        self.experimental = kwargs
        return kwargs

    def make_schedule(self, **kwargs):
        self.schedule = kwargs
        return kwargs

    def call(self):
        self.calls.append(self.active)
        return len(self.calls)

    def profile(self, **kwargs):
        self.arguments = kwargs
        return self

    def __enter__(self):
        self.active = True
        return self

    def __exit__(self, *args):
        self.active = False

    def step(self):
        self.steps += 1


def csv_bytes(count=10, operator='_attn_fwd'):
    return ('Device_id,OP Type,Core Type,Count,Total Time(us),Min Time(us),Avg Time(us),Max Time(us)\n'
            '0,%s,AI_CORE,%d,%d,9,10,11\n' % (operator, count, count * 10)).encode()


def qualification():
    case = dict(case_id='n1024-hd64', shape=[128, 8, 1024, 64], block_m=128, block_n=256)
    record = dict(preset='candidate', options={'test': True}, hardware_correctness='passed-for-this-case',
                  native_sha256='a' * 64)
    checks = [dict(passed=True, finite=True), dict(passed=True, finite=True)]
    result = dict(
        finished=True, exit=0, measurement_mode='correctness_only', timed_task_count=0, case=case, numerics=[
            dict(seed=s, kind='normal', passed=True, inputs=['input'], inputs_after=['input'],
                 variants=[dict(preset='candidate', passed=True)]) for s in (12, 0, 42)
        ], stress=[
            dict(preset=arm, iteration=i, checks=copy.deepcopy(checks))
            for arm in ('candidate', 'ascendc')
            for i in (32, 64)
        ], stress_inputs=['input'], stress_inputs_after=['input'], native_compiles=[record])
    return case, result


class CollectorTest(unittest.TestCase):

    def test_exact_original_profile_recipe(self):
        r = Recorder()
        admissions = []
        initialized = []
        with tempfile.TemporaryDirectory() as d:
            latest, recipe = profile_block(r.torch, r.npu, r.call,
                                           Path(d) / 'block', lambda: admissions.append(1),
                                           initialized=lambda x: initialized.append(
                                               (x, r.active)), sleeper=r.sleeps.append)
        self.assertEqual(latest, 14)
        self.assertEqual(r.calls, [False] + [True] * 13)
        self.assertEqual(r.syncs, 14)
        self.assertEqual(r.sleeps, [1] * 13)
        self.assertEqual(r.steps, 13)
        self.assertEqual(len(admissions), 14)
        self.assertEqual(initialized, [(1, False)])
        self.assertEqual(r.schedule, SCHEDULE)
        self.assertEqual(r.experimental, dict(aic_metrics='pipe', profiler_level='level1', l2_cache=False))
        self.assertEqual(r.arguments['activities'], ['npu'])
        self.assertFalse(r.arguments['with_stack'])
        self.assertFalse(r.arguments['record_shapes'])
        self.assertFalse(r.arguments['profile_memory'])
        self.assertEqual(recipe['profile_calls'], 13)
        self.assertEqual(recipe['selected_active_calls'], 10)

    def test_failed_initialization_never_enters_profiler(self):
        r = Recorder()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(ValueError, 'image'):
                profile_block(r.torch, r.npu, r.call,
                              Path(d) / 'block', lambda: None, initialized=lambda x:
                              (_ for _ in ()).throw(ValueError('image mismatch')), sleeper=r.sleeps.append)
        self.assertEqual(r.calls, [False])
        self.assertIsNone(r.arguments)

    def test_existing_block_directory_is_not_reused(self):
        r = Recorder()
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaises(FileExistsError):
                profile_block(r.torch, r.npu, r.call, d, lambda: None, sleeper=r.sleeps.append)
        self.assertEqual(r.calls, [])

    def test_csv_selection_preserves_one_exact_record(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'op_statistic.csv'
            p.write_bytes(csv_bytes())
            block = select_block(d, device_id=0, arm='cvpipeline', core_type='AI_CORE')
        self.assertEqual(block['task_count'], 10)
        self.assertEqual(block['mean_duration_us'], 10)
        self.assertEqual(block['source_record_index'], 0)

    def test_csv_with_wrong_active_count_is_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'op_statistic.csv').write_bytes(csv_bytes(count=9))
            with self.assertRaisesRegex(ValueError, 'count'):
                select_block(d, device_id=0, arm='cvpipeline', core_type='AI_CORE')

    def test_ambiguous_profiler_directories_are_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            for name in ('a', 'b'):
                p = Path(d) / name
                p.mkdir()
                (p / 'op_statistic.csv').write_bytes(csv_bytes())
            with self.assertRaisesRegex(ValueError, 'More than one'):
                select_block(d, device_id=0, arm='cvpipeline', core_type='AI_CORE')

    def test_missing_csv_has_a_bounded_failure(self):
        with tempfile.TemporaryDirectory() as d:
            with self.assertRaisesRegex(RuntimeError, 'did not produce'):
                select_block(d, device_id=0, arm='cvpipeline', core_type='AI_CORE', timeout=0)

    def test_mapping_uses_reported_id_from_the_exact_operator(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'op_statistic.csv'
            p.write_bytes(csv_bytes().replace(b'0,_attn_fwd', b'3,_attn_fwd'))
            self.assertEqual(infer_csv_device(d, arm='cvpipeline', core_type='AI_CORE'), 3)

    def test_mapping_rejects_an_unrelated_attention_operator(self):
        with tempfile.TemporaryDirectory() as d:
            (Path(d) / 'op_statistic.csv').write_bytes(csv_bytes(operator='unrelated'))
            with self.assertRaises(ValueError):
                infer_csv_device(d, arm='cvpipeline', core_type='AI_CORE')

    def test_qualification_requires_stress_options_and_input_integrity(self):
        case, good = qualification()
        self.assertEqual(qualified_native(good, case, 'candidate', {'test': True})['native_sha256'], 'a' * 64)
        for mutation in ('unfinished', 'seed', 'stress', 'inputs', 'options', 'geometry'):
            bad = copy.deepcopy(good)
            if mutation == 'unfinished': bad['finished'] = False
            if mutation == 'seed': bad['numerics'].pop()
            if mutation == 'stress': bad['stress'].pop()
            if mutation == 'inputs': bad['numerics'][0]['inputs_after'] = ['changed']
            if mutation == 'options': bad['native_compiles'][0]['options'] = {'test': False}
            if mutation == 'geometry': bad['case']['block_n'] = 128
            with self.subTest(mutation=mutation), self.assertRaises(ValueError):
                qualified_native(bad, case, 'candidate', {'test': True})

    def test_real_recipe_records_declared_host_options_not_filtered_kwargs(self):
        case, result = qualification()
        case['initial_candidate'] = 'candidate'
        presets = json.loads(Path(__file__).with_name('recipes.json').read_text())['presets']
        actual = copy.deepcopy(presets['common'])
        result['native_compiles'][0]['options'] = copy.deepcopy(actual)
        declared, effective, record = qualified_recipe(result, case, {'candidate': actual})
        self.assertIs(declared['enable_mixed_cv'], True)
        self.assertNotIn('enable_mixed_cv', effective)
        self.assertEqual(record['options'], declared)
        self.assertEqual(record['native_sha256'], 'a' * 64)

    def test_declared_option_mismatch_does_not_get_normalized_away(self):
        case, result = qualification()
        case['initial_candidate'] = 'candidate'
        actual = json.loads(Path(__file__).with_name('recipes.json').read_text())['presets']['common']
        result['native_compiles'][0]['options'] = copy.deepcopy(actual)
        changed = dict(actual, enable_fused_multiply_add=False)
        with self.assertRaises(ValueError):
            qualified_recipe(result, case, {'candidate': changed})

    def test_help_is_available_without_torch_or_npu(self):
        p = subprocess.run(
            [sys.executable, '-B', str(Path(__file__).with_name('run_comparison.py')), '--help'], capture_output=True,
            text=True)
        self.assertEqual(p.returncode, 0, p.stderr)
        self.assertIn('--qualified-sha256', p.stdout)


if __name__ == '__main__': unittest.main()
