# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""TTIR scoring, isolated selection and bounded backend compilation; CPU-only."""
from block_batching_test_utils import assign_npu  # noqa: F401 - CPU-only fixture
from block_batching_test_utils import parse
import importlib
import json
import hashlib
import os
import subprocess
import sys
from pathlib import Path
import tempfile
import unittest
from contextlib import ExitStack
from types import SimpleNamespace, ModuleType
from unittest.mock import Mock, patch

from triton._C.libtriton import ascend, ir
from triton.backends.ascend import block_batching as batching
from triton.backends.ascend.compiler import AscendBackend
from triton.backends.compiler import GPUTarget


def test_data_only_scalar_through_generator_arithmetic(tmp_path):
    module = parse(
        tmp_path, '''module {
      tt.func public @gen(%out: !tt.ptr<i32>, %seed: i32) {
        %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
        %s = tt.splat %seed : i32 -> tensor<8xi32>
        %v = arith.addi %s, %r : tensor<8xi32>
        %bits = arith.muli %v, %v : tensor<8xi32>
        %keep = arith.cmpi ult, %bits, %v : tensor<8xi32>
        %data = arith.select %keep, %bits, %v : tensor<8xi1>, tensor<8xi32>
        %p = tt.splat %out : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
        %ptrs = tt.addptr %p, %r : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
        tt.store %ptrs, %data : tensor<8x!tt.ptr<i32>>
        tt.return
      }
    }''')
    assert list(ascend.block_data_only_arguments(module, 2)) == [1]
    assert list(ascend.block_data_only_arguments(module, 3)) == []


def test_address_mask_and_control_scalars_keep_their_values(tmp_path):
    module = parse(
        tmp_path, '''module {
      tt.func public @mixed(%direct: i32, %data: i32, %out: !tt.ptr<i32>,
                            %indirect: i32, %bound: i32, %count: index, %branch: i1,
                            %slot: !tt.ptr<!tt.ptr<i32>>, %escaped: i32) {
        %c0 = arith.constant 0 : index
        %c1 = arith.constant 1 : index
        %p = tt.addptr %out, %direct : !tt.ptr<i32>, i32
        %offset = arith.addi %indirect, %indirect : i32
        %q = tt.addptr %p, %offset : !tt.ptr<i32>, i32
        %mask = arith.cmpi slt, %direct, %bound : i32
        scf.for %i = %c0 to %count step %c1 {
          scf.if %branch {
            tt.store %q, %direct : !tt.ptr<i32>
          }
        }
        %stored_ptr = tt.addptr %p, %escaped : !tt.ptr<i32>, i32
        tt.store %slot, %stored_ptr : !tt.ptr<!tt.ptr<i32>>
        tt.store %p, %data, %mask : !tt.ptr<i32>
        tt.return
      }
    }''')
    assert list(ascend.block_data_only_arguments(module, 9)) == [1]


class TestTtirAnalysis(unittest.TestCase):

    def score(self, text, scalar_arguments=None, **options):
        context = ir.context()
        ir.load_dialects(context)
        AscendBackend(GPUTarget("npu", "Ascend950", 32)).load_dialects(context)
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "kernel.mlir"
            source.write_text(text)
            module = ir.parse_mlir_module(str(source), context)
        request = dict(contract=dict(grid=[256, 1, 1]), factor=1, axis=-1, cv=False, arch="Ascend950", cores=56)
        request.update(options)
        if scalar_arguments is not None:
            request["scalar_arguments"] = scalar_arguments
        original = str(module)
        result = json.loads(ascend.score_block_ttir(module, json.dumps(request)))
        self.assertEqual(str(module), original)
        return result

    def test_dynamic_loop_declines_instead_of_assuming_one_iteration(self):
        result = self.score('''module {
          tt.func public @kernel(%n: index, %out: !tt.ptr<i32>) {
            %zero = arith.constant 0 : index
            %one = arith.constant 1 : index
            scf.for %i = %zero to %n step %one {
              %value = arith.addi %i, %one : index
              %stored = arith.index_cast %value : index to i32
              tt.store %out, %stored : !tt.ptr<i32>
              scf.yield
            }
            tt.return
          }
        }''')
        self.assertEqual(result["status"], "TTIR_MODEL_UNSUPPORTED")
        self.assertIn("dynamic loop", result["error"])

    def test_rejection_message_survives_exception_destruction(self):
        # Disable glibc's tcache so freed exception storage is overwritten.
        script = '''
import runpy, sys
from pathlib import Path
sys.path.insert(0, str(Path(sys.argv[1]).resolve().parent))
tests = runpy.run_path(sys.argv[1])
case = tests['TestTtirAnalysis']()
for _ in range(20):
    case.test_dynamic_loop_declines_instead_of_assuming_one_iteration()
    case.test_missing_target_resources_cannot_use_a5_defaults()
'''
        env = dict(os.environ, GLIBC_TUNABLES="glibc.malloc.tcache_count=0", MALLOC_PERTURB_="165")
        result = subprocess.run([sys.executable, "-c", script, __file__], env=env, capture_output=True, text=True,
                                timeout=60)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def test_launch_scalar_specializes_loop_without_changing_executable_ir(self):
        text = '''module {
          tt.func public @kernel(%n: index, %out: !tt.ptr<i32>) {
            %zero = arith.constant 0 : index
            %one = arith.constant 1 : index
            scf.for %i = %zero to %n step %one {
              %stored = arith.index_cast %i : index to i32
              tt.store %out, %stored : !tt.ptr<i32>
            }
            tt.return
          }
        }'''
        result = self.score(text, [7, None])
        self.assertEqual(result["status"], "TTIR_COST_ESTIMATED")
        self.assertEqual(result["memory_requests"], 7)
        self.assertFalse(result["loop_trip_upper_bound"])

    def test_persistent_loop_uses_grid_range_for_worst_core(self):
        result = self.score('''module {
          tt.func public @kernel(%out: !tt.ptr<i32>) {
            %pid = tt.get_program_id x : i32
            %grid = tt.get_num_programs x : i32
            %upper = arith.constant 1000 : i32
            scf.for %i = %pid to %upper step %grid : i32 {
              tt.store %out, %i : !tt.ptr<i32>
            }
            tt.return
          }
        }''')
        self.assertEqual(result["status"], "TTIR_COST_ESTIMATED")
        self.assertEqual(result["memory_requests"], 4)
        self.assertTrue(result["loop_trip_upper_bound"])

    def test_frontend_liveness_never_claims_final_ub_validation(self):
        result = self.score('''module {
          tt.func public @kernel(%x: tensor<128xf32>) {
            %v = arith.mulf %x, %x : tensor<128xf32>
            tt.return
          }
        }''')
        self.assertEqual(result["status"], "TTIR_COST_ESTIMATED")
        self.assertFalse(result["resource_checked"])

    def test_equivalent_duplicate_expressions_have_the_same_cost(self):
        template = '''module {
          tt.func public @kernel(%x: tensor<128xf32>, %out: tensor<128x!tt.ptr<f32>>) {
            %a = arith.mulf %x, %x : tensor<128xf32>
            DUPLICATE
            %sum = arith.addf %a, RHS : tensor<128xf32>
            tt.store %out, %sum : tensor<128x!tt.ptr<f32>>
            tt.return
          }
        }'''
        shared = self.score(template.replace("DUPLICATE", "").replace("RHS", "%a"))
        duplicate = self.score(
            template.replace("DUPLICATE", "%b = arith.mulf %x, %x : tensor<128xf32>").replace("RHS", "%b"))
        self.assertEqual(shared["status"], "TTIR_COST_ESTIMATED")
        self.assertEqual(shared["score"], duplicate["score"])
        self.assertEqual(shared["estimated_live_bytes"], duplicate["estimated_live_bytes"])

    def test_scan_cost_accounts_for_retaining_all_prefix_values(self):
        reduction = '''module {
          tt.func public @kernel(%x: tensor<512xf32>, %out: !tt.ptr<f32>) {
            %v = "tt.reduce"(%x) ({
            ^bb0(%a: f32, %b: f32):
              %c = arith.addf %a, %b : f32
              tt.reduce.return %c : f32
            }) {axis = 0 : i32} : (tensor<512xf32>) -> f32
            tt.store %out, %v : !tt.ptr<f32>
            tt.return
          }
        }'''
        scan = '''module {
          tt.func public @kernel(%x: tensor<512xf32>, %out: tensor<512x!tt.ptr<f32>>) {
            %v = "tt.scan"(%x) ({
            ^bb0(%a: f32, %b: f32):
              %c = arith.addf %a, %b : f32
              tt.scan.return %c : f32
            }) {axis = 0 : i32, reverse = false} : (tensor<512xf32>) -> tensor<512xf32>
            tt.store %out, %v : tensor<512x!tt.ptr<f32>>
            tt.return
          }
        }'''
        self.assertGreater(self.score(scan)["vector_work"], self.score(reduction)["vector_work"])

    def test_small_dma_batches_do_not_charge_bytes_for_idle_tail_cores(self):

        def candidate(factor):
            size = 512 * factor
            text = '''module {
              tt.func public @kernel(%ptr: !tt.ptr<f32>) {
                %r = tt.make_range {start = 0 : i32, end = SIZE : i32} : tensor<SIZExi32>
                %p = tt.splat %ptr : !tt.ptr<f32> -> tensor<SIZEx!tt.ptr<f32>>
                %q = tt.addptr %p, %r : tensor<SIZEx!tt.ptr<f32>>, tensor<SIZExi32>
                %x = tt.load %q : tensor<SIZEx!tt.ptr<f32>>
                %y = arith.mulf %x, %x : tensor<SIZExf32>
                tt.store %q, %y : tensor<SIZEx!tt.ptr<f32>>
                tt.return
              }
            }'''.replace("SIZE", str(size))
            return self.score(text, factor=factor, axis=0)

        original, merged = candidate(1), candidate(4)
        self.assertEqual(original["status"], "TTIR_COST_ESTIMATED")
        self.assertEqual(merged["status"], "TTIR_COST_ESTIMATED")
        self.assertEqual(original["blocks"] * original["bytes_per_program"],
                         merged["blocks"] * merged["bytes_per_program"])
        self.assertLess(merged["score"], original["score"])

    def test_batched_scans_retain_serial_helper_cost_per_row(self):

        def candidate(factor):
            shape = "512" if factor == 1 else f"{factor}x512"
            axis = 0 if factor == 1 else 1
            text = '''module {
              tt.func public @kernel(%p: tensor<SHAPEx!tt.ptr<f32>>) {
                %x = tt.load %p : tensor<SHAPEx!tt.ptr<f32>>
                %y = "tt.scan"(%x) ({
                ^bb0(%a: f32, %b: f32):
                  %c = arith.addf %a, %b : f32
                  tt.scan.return %c : f32
                }) {axis = AXIS : i32, reverse = false} :
                    (tensor<SHAPExf32>) -> tensor<SHAPExf32>
                tt.store %p, %y : tensor<SHAPEx!tt.ptr<f32>>
                tt.return
              }
            }'''.replace("SHAPE", shape).replace("AXIS", str(axis))
            return self.score(text, factor=factor, axis=0)

        original = candidate(1)
        self.assertEqual(original["status"], "TTIR_COST_ESTIMATED")
        for factor in (2, 4):
            merged = candidate(factor)
            self.assertEqual(merged["status"], "TTIR_COST_ESTIMATED")
            self.assertGreater(merged["score"], original["score"])

    def test_small_predicted_gain_remains_eligible(self):
        request = dict(
            action="rank_ttir", contract=dict(grid=[256, 1, 1]), candidates=[
                dict(factor=1, axis=-1, status="TTIR_COST_ESTIMATED", score=100, cores=56, blocks=256),
                dict(factor=4, axis=0, status="TTIR_COST_ESTIMATED", score=99.5, cores=56, blocks=64)
            ])
        result = json.loads(ascend.select_block_candidates(json.dumps(request)))
        self.assertEqual(result["factor"], 4)

    def test_missing_baseline_cost_forces_f1(self):
        request = dict(
            action="rank_ttir", contract=dict(grid=[256, 1, 1]), candidates=[
                dict(factor=1, axis=-1, status="TTIR_MODEL_UNSUPPORTED"),
                dict(factor=2, axis=0, status="TTIR_COST_ESTIMATED", score=1, cores=56, blocks=128)
            ])
        result = json.loads(ascend.select_block_candidates(json.dumps(request)))
        self.assertEqual(result["factor"], 1)
        self.assertFalse(result["model_complete"])

    def test_legacy_targets_select_a_proven_group_with_target_core_count(self):
        text = """module { tt.func public @kernel(%out: !tt.ptr<i32>) {
          %zero = arith.constant 0 : i32
          tt.store %out, %zero : !tt.ptr<i32>
          tt.return
        }}"""
        # The A5 cost model does not transfer, so every SOC of the earlier
        # generation scores on legality and occupancy instead.
        for arch in ("Ascend910_9391", "Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4"):
            rows = []
            for factor in (1, 2, 4):
                row = self.score(text, arch=arch, cores=20, factor=factor, axis=-1 if factor == 1 else 0, cv=False,
                                 merge_kind="vv")
                self.assertEqual(row["model"], "ttir-legacy-grouping-v1", arch)
                rows.append(row)
            result = json.loads(
                ascend.select_block_candidates(
                    json.dumps(dict(action="rank_ttir", contract=dict(grid=[256, 1, 1]), candidates=rows))))
            self.assertEqual(result["factor"], 2, arch)
            self.assertEqual(result["candidates"]["a0_f2"]["cores"], 20, arch)

    def test_missing_target_resources_cannot_use_a5_defaults(self):
        result = self.score("module { tt.func public @kernel() { tt.return } }", cores=0)
        self.assertEqual(result["status"], "TTIR_MODEL_UNSUPPORTED")


def selection_report(factor=2):
    key = "unmerged" if factor == 1 else f"a0_f{factor}"
    return dict(
        selection=dict(factor=factor, selected_axis=-1 if factor == 1 else 0, selected_key=key, model_complete=True,
                       reason="LOWEST_TTIR_SCORE",
                       candidates={key: dict(factor=factor, axis=0, requested_axis=0, status="TTIR_COST_ESTIMATED")}),
        plan=dict(candidates=[dict(axis=0, factor=factor)]), prepared=None if factor == 1 else dict(
            variant="candidate", digest=hashlib.sha256(b"ttir").hexdigest(), report=dict(selected_axis=0)))


class TestSelectionCache(unittest.TestCase):

    def setUp(self):
        from triton import knobs
        from triton.backends.ascend import block_auto
        self.auto = block_auto
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        scope = knobs.cache.scope()
        scope.__enter__()
        self.addCleanup(scope.__exit__, None, None, None)
        knobs.cache.dir = str(self.root / "cache")
        self.body = dict(variant="source-target-options-env", contract=dict(grid=[256, 1, 1]), scalar_arguments=[123],
                         target=dict(arch="Ascend950"), knobs={})
        self.cache = block_auto._SelectionCache(self.body)
        self.output = self.root / "selected.ttir"
        self.output.write_bytes(b"ttir")

    def test_decision_survives_new_cache_object_and_binds_full_request(self):
        report = selection_report()
        self.cache.store(report, self.output)
        self.assertEqual(self.auto._SelectionCache(self.body).load(), report)
        for field, value in (("variant", "changed"), ("contract", dict(grid=[128, 1, 1])), ("scalar_arguments", [124]),
                             ("target", dict(arch="Ascend910_9391")), ("knobs", dict(debug=True))):
            with self.subTest(field=field):
                changed = self.auto._SelectionCache(dict(self.body, **{field: value}))
                self.assertIsNone(changed.load())

    def test_normal_f1_needs_no_snapshot(self):
        report = selection_report(1)
        self.cache.store(report, self.root / "absent")
        self.assertEqual(self.cache.load(), report)

    def test_corrupt_record_is_a_miss(self):
        self.cache.store(selection_report(), self.output)
        record = Path(self.cache.cache.get_file("block_auto.selection"))
        for data in ("{", json.dumps(dict(version=99)), record.read_text().replace("LOWEST_TTIR_SCORE", "changed")):
            record.write_text(data)
            self.assertIsNone(self.cache.load())

    def test_snapshot_is_checked_only_when_requested(self):
        report = selection_report()
        self.cache.store(report, self.output)
        prepared = report["prepared"]
        path = Path(self.cache.cache.get_file(f"selected-{prepared['digest']}.ttir"))
        restored = self.root / "restored"
        self.cache.restore_ttir(prepared, restored)
        self.assertEqual(restored.read_bytes(), b"ttir")
        path.write_bytes(b"corrupt")
        self.assertEqual(self.cache.load(), report)
        with self.assertRaises(self.auto._SelectionMiss):
            self.cache.restore_ttir(prepared, restored)
        path.unlink()
        with self.assertRaises(self.auto._SelectionMiss):
            self.cache.restore_ttir(prepared, restored)

    def test_retryable_outcomes_and_unaudited_refusals_are_not_cached(self):
        for change in ("failure", "candidate", "model_limit", "rejection", "decline"):
            report = selection_report(1)
            candidate = report["selection"]["candidates"]["unmerged"]
            if change == "failure":
                report["selection"]["failure_kind"] = "Timeout"
            elif change == "candidate":
                candidate["status"] = "TTIR_MODEL_UNSUPPORTED"
            elif change == "model_limit":
                candidate.update(status="TTIR_MODEL_UNSUPPORTED", error="analysis budget exhausted")
            elif change == "rejection":
                candidate.update(status="COMPILE_REJECTED", error="UNSUPPORTED_BATCH_OP")
            else:
                report["plan"]["declines"] = [dict(axis=0, factor=2, stage="ttir", reason="UNSUPPORTED_BATCH_OP")]
            with self.subTest(change=change):
                report["selection"]["model_complete"] = False
                self.assertFalse(self.auto._cacheable_selection(report))

    def test_audited_deterministic_outcomes_are_cached_without_a_priced_baseline(self):
        from triton.backends.ascend.block_batching_worker import _CACHEABLE_DECLINES, _DETERMINISTIC_MODEL_LIMITS
        # A kernel the model cannot price reaches the same verdict every time,
        # so the decision persists instead of being rediscovered per process.
        report = selection_report(1)
        report["selection"].update(model_complete=False, reason="TTIR_BASELINE_MODEL_UNSUPPORTED")
        report["selection"]["candidates"]["unmerged"].update(status="TTIR_MODEL_UNSUPPORTED",
                                                             error=sorted(_DETERMINISTIC_MODEL_LIMITS)[0])
        report["selection"]["candidates"]["a0_f2"] = dict(factor=2, axis=0, requested_axis=0, status="COMPILE_REJECTED",
                                                          error="UNSUPPORTED_BYTE_OPERATION")
        report["plan"]["declines"] = [dict(axis=0, factor=2, stage="ttir", reason="UNSUPPORTED_BYTE_OPERATION")]
        self.assertIn("UNSUPPORTED_BYTE_OPERATION", _CACHEABLE_DECLINES)
        self.assertTrue(self.auto._cacheable_selection(report))


class TestCompileContextCache(unittest.TestCase):

    def setUp(self):
        from triton import knobs
        from triton.backends.ascend import compiler
        self.compiler, self.knobs = compiler, knobs
        self.backend = compiler.AscendBackend(GPUTarget("npu", "Ascend950", 32))
        self.cache = {}
        self.options = dict(block_batching_factor="auto")
        stack = ExitStack()
        self.addCleanup(stack.close)
        for group in (knobs.compilation, knobs.ascend, knobs.cache):
            stack.enter_context(group.scope())
        knobs.compilation.always_compile = False
        knobs.ascend.compile_only = False
        stack.enter_context(patch.dict("os.environ", NPU_DEVICE_LIMIT="28,56"))
        self.cores = stack.enter_context(patch.object(compiler, "_graph_optimize_core_num", return_value=28))
        self.limit = stack.enter_context(patch.object(compiler, "_compile_device_limit", return_value=(False, 28, 56)))
        self.parse = stack.enter_context(patch.object(self.backend, "parse_options", wraps=self.backend.parse_options))

    def prepare(self, options=None, cache=None):
        return self.backend.prepare_block_compile_options(self.options if options is None else options,
                                                          cache=self.cache if cache is None else cache)

    def test_shapes_and_runtime_arguments_do_not_repeat_target_preparation(self):
        first = self.prepare()
        second = self.prepare(dict(self.options, size=512, block_auto_contract="new", block_batching_grid=(32, 1, 1)))
        self.assertEqual(first["block_compile_context"], second["block_compile_context"])
        self.assertEqual(second["size"], 512)
        self.parse.assert_called_once()
        self.limit.assert_called_once()

    def test_device_cache_clear_and_configuration_changes_refresh(self):
        self.prepare()
        self.prepare(cache={})
        self.cache.clear()
        self.prepare()
        with patch.dict("os.environ", NPU_DEVICE_LIMIT="14,28"):
            self.prepare()
        self.knobs.cache.dir += "-other"
        self.prepare()
        self.knobs.ascend.npu_compiler_path = "/other/compiler"
        # The real option parser validates this executable; substitute only
        # that parser while checking that the preparation cache misses.
        parsed = self.parse.return_value = SimpleNamespace(block_compile_context=None, enable_graph_optimize=True,
                                                           compile_on_910_95=True, is_pure_simt=False)
        self.parse.side_effect = lambda _: parsed
        self.prepare()
        self.prepare(dict(self.options, enable_graph_optimize=False))
        self.assertEqual(self.parse.call_count, 7)

    def test_explicit_context_and_compile_only_retain_validation(self):
        self.prepare()
        self.knobs.ascend.compile_only = True
        with self.assertRaisesRegex(ValueError, "explicit block_compile_context"):
            self.prepare()
        context = ("Ascend950", 14, (True, 14, 28), None)
        self.assertEqual(
            self.prepare(dict(self.options, block_compile_context=context))["block_compile_context"], context)
        with self.assertRaises(ValueError):
            self.prepare(dict(self.options, block_compile_context=("Ascend950", 14, None, None)))

    def test_failed_target_queries_remain_retryable(self):
        self.limit.side_effect = ValueError("bad limit")
        with self.assertRaisesRegex(ValueError, "bad limit"):
            self.prepare()
        self.assertEqual(self.cache, {})
        self.limit.side_effect = None
        self.cores.return_value = 0
        self.prepare()
        self.prepare()
        self.assertEqual(self.cache, {})
        self.cores.return_value = 28
        self.prepare()
        self.prepare()
        self.assertEqual(self.parse.call_count, 4)

    def test_forced_compilation_reuses_only_launch_target_preparation(self):
        self.knobs.compilation.always_compile = True
        first = self.prepare()
        self.assertEqual(self.prepare(), first)
        self.parse.assert_called_once()
        with patch.dict("os.environ", NPU_DEVICE_LIMIT="14,28"):
            self.prepare()
        self.assertEqual(self.parse.call_count, 2)
        # Actual compilation does not supply the per-device launch cache.
        self.backend.prepare_block_compile_options(self.options)
        self.backend.prepare_block_compile_options(self.options)
        self.assertEqual(self.parse.call_count, 4)
        self.cache.clear()
        self.prepare()
        self.assertEqual(self.parse.call_count, 5)

    def test_simt_keeps_refreshing_its_runtime_configuration(self):
        self.parse.side_effect = lambda _: SimpleNamespace(block_compile_context=None, enable_graph_optimize=True,
                                                           compile_on_910_95=True, is_pure_simt=True)
        with patch.object(self.compiler, "get_simt_stack_limit", side_effect=[1152, 2048]):
            self.assertEqual(self.prepare()["block_compile_context"][3], 1152)
            self.assertEqual(self.prepare()["block_compile_context"][3], 2048)
        self.assertEqual(self.cache, {})


class TestRankedRecovery(unittest.TestCase):

    def run_selection(self, *, selected=4, decline=False, failure=None, arch="Ascend950", saved=None, binary_hit=True):
        from triton.backends.ascend import block_auto as dispatch, block_batching_worker as worker
        backend = Mock(target=GPUTarget("npu", arch, 32))
        backend.parse_options.side_effect = lambda options: SimpleNamespace(block_batching_backend_revision="test", **
                                                                            options)
        source = SimpleNamespace(fn=SimpleNamespace(arg_names=["X"]))
        baseline = SimpleNamespace(hash="f1")
        events = []
        env = {"TRITON_ENABLE_SANITIZER": "1", "NPU_DEVICE_LIMIT": "12,24"}

        def compile_f1(*args):
            events.append("f1")
            return baseline

        plan = dict(candidates=[dict(axis=0, factor=2), dict(axis=0, factor=4)])
        key = "unmerged" if selected == 1 else f"a0_f{selected}"
        selection = dict(
            factor=selected, selected_axis=-1 if selected == 1 else 0, selected_key=key,
            candidates={key: dict(factor=selected, axis=0, requested_axis=0,
                                  status="TTIR_COST_ESTIMATED")}, model_complete=True, reason="TTIR_SELECTED")
        prepared = None if selected == 1 else dict(report=dict(selected_axis=0))

        def attempt(operation, body, *, timeout):
            self.assertGreater(timeout, 0)
            self.assertEqual(operation, "select")
            self.assertEqual(body["env_vars"], env)
            events.append("ttir")
            if failure:
                raise worker.BatchCompileError("worker failed", kind=failure, stage="select")
            return SimpleNamespace(payload=dict(report=dict(selection=selection, plan=plan, prepared=prepared)))

        def compile_candidate(baseline, options, request, positions, *, allow_layout_retry=True, cached_only=False):
            self.assertFalse(allow_layout_retry)
            self.assertIn(request.factor, (2, 4))
            events.append("binary-hit" if cached_only else request.factor)
            if cached_only and not binary_hit:
                return None
            return SimpleNamespace(candidate=None if decline else object(),
                                   block_batching=dict(reason="VV_TARGET_RESOURCE_REJECTED"))

        contract = dict(grid=[256, 1, 1], arguments=dict(X=dict(kind="tensor")))
        options = dict(block_auto_contract=json.dumps(contract), block_batching_factor="auto")
        with ExitStack() as stack:
            store = saved if saved is not None else Mock(load=lambda: None)
            stack.enter_context(patch.object(dispatch, "_SelectionCache", return_value=store))
            for name, value in [("_freeze_refusal", None), ("_candidate_refusal", None), ("_runtime_positions", [0]),
                                ("_freeze_source", Mock())]:
                stack.enter_context(patch.object(worker, name, return_value=value))
            stack.enter_context(patch.object(worker, "_variant_key", return_value="f1"))
            stack.enter_context(
                patch.object(worker, "_worker_request", side_effect=lambda b, s, o, e, v: dict(env_vars=e)))
            stack.enter_context(patch.object(worker, "_attempt", side_effect=attempt))
            stack.enter_context(patch.object(worker.CandidateCompiler, "compile", side_effect=compile_candidate))
            stack.enter_context(patch.object(batching, "BatchKernel", return_value=SimpleNamespace()))
            result = dispatch.compile_auto_bundle(backend, source, options, compile_f1, env_vars=env)
        return result.block_auto_selection, events

    def test_only_winner_and_f1_enter_backend(self):
        result, events = self.run_selection()
        self.assertEqual(events, ["ttir", "f1", 4])
        self.assertEqual(result["selected_factor"], 4)

    def test_cached_binary_does_not_start_worker_or_read_ttir(self):
        saved = Mock()
        saved.load.return_value = selection_report()
        saved.restore_ttir.side_effect = AssertionError("binary hit must not read TTIR")
        result, events = self.run_selection(selected=2, saved=saved)
        self.assertEqual(events, ["f1", "binary-hit"])
        self.assertEqual(result["selected_factor"], 2)
        saved.store.assert_not_called()

    def test_evicted_binary_uses_saved_ttir_without_reselection(self):
        saved = Mock(load=lambda: selection_report())
        result, events = self.run_selection(selected=2, saved=saved, binary_hit=False)
        self.assertEqual(events, ["f1", "binary-hit", 2])
        saved.restore_ttir.assert_called_once()

    def test_evicted_binary_and_ttir_reselect(self):
        from triton.backends.ascend import block_auto
        saved = Mock(load=lambda: selection_report(), restore_ttir=Mock(side_effect=block_auto._SelectionMiss))
        result, events = self.run_selection(selected=2, saved=saved, binary_hit=False)
        self.assertEqual(events, ["f1", "binary-hit", "ttir", "f1", 2])
        saved.store.assert_called_once()

    def test_complete_f1_decision_is_cached_but_failures_are_not(self):
        for options, stored in ((dict(selected=1), True), (dict(decline=True), False), (dict(failure="Timeout"),
                                                                                        False)):
            saved = Mock(load=lambda: None)
            self.run_selection(saved=saved, **options)
            self.assertEqual(saved.store.called, stored)

    def test_cached_f1_does_not_start_a_worker(self):
        saved = Mock(load=lambda: selection_report(1))
        result, events = self.run_selection(saved=saved)
        self.assertEqual(events, ["f1"])
        self.assertEqual(result["selected_factor"], 1)
        saved.restore_ttir.assert_not_called()

    def test_forced_compile_ignores_decision_cache(self):
        from triton import knobs
        saved = Mock(load=Mock(side_effect=AssertionError("forced compile read selection cache")))
        with knobs.compilation.scope():
            knobs.compilation.always_compile = True
            _, events = self.run_selection(saved=saved)
        self.assertEqual(events, ["ttir", "f1", 4])
        saved.store.assert_not_called()

    def test_baseline_winner_only_compiles_f1(self):
        result, events = self.run_selection(selected=1)
        self.assertEqual(events, ["ttir", "f1"])
        self.assertEqual(result["selected_factor"], 1)

    def test_rejected_winner_keeps_f1_without_compiling_runner_up(self):
        result, events = self.run_selection(decline=True)
        self.assertEqual(events, ["ttir", "f1", 4])
        self.assertEqual(result["selected_factor"], 1)
        self.assertEqual(result["reason"], "BASELINE_AFTER_CANDIDATE_REJECTION")

    def test_legacy_targets_use_the_selection_worker(self):
        for arch in ("Ascend910_9391", "Ascend910B1", "Ascend910B4"):
            with self.subTest(arch=arch):
                result, events = self.run_selection(arch=arch, selected=2)
                self.assertEqual(events, ["ttir", "f1", 2])
                self.assertEqual(result["selected_factor"], 2)

    def test_abnormal_worker_exits_do_not_masquerade_as_declines(self):
        from triton.backends.ascend.block_batching_worker import BatchCompileError
        for kind in ("Signal", "InternalError"):
            with self.subTest(kind=kind), self.assertRaises(BatchCompileError) as caught:
                self.run_selection(failure=kind)
            self.assertEqual(caught.exception.kind, kind)

    def test_selection_timeout_keeps_retryable_f1(self):
        result, events = self.run_selection(failure="Timeout")
        self.assertEqual(events, ["ttir", "f1"])
        self.assertEqual(result["failure_kind"], "Timeout")

    def test_auto_does_not_retry_a_rejected_layout(self):
        from triton.backends.ascend import block_batching_worker as worker
        compiler = worker.CandidateCompiler(Mock(), Mock(), {})
        decline = dict(reason="LAYOUT_TARGET_RESOURCE_REJECTED", stage="npubin", diagnostic="UB capacity exceeded",
                       factor=4, axis=0)
        with patch.object(compiler, "_variant", return_value=(decline, None)) as attempt, \
             patch.object(worker, "rejected_bundle", return_value="fallback"):
            result = compiler.compile(Mock(), dict(block_batching_layout="auto"), batching.BatchRequest(4, 0), [],
                                      allow_layout_retry=False)
        self.assertEqual(result, "fallback")
        attempt.assert_called_once()


BACKEND = Path(__file__).resolve().parents[2] / "backend"


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


PACKAGE = "_block_auto_private_test"


class Tensor:
    dtype = "torch.float32"
    shape = (256, 512)

    def stride(self):
        return (512, 1)

    def storage_offset(self):
        return 0


class TestHostIntegration(unittest.TestCase):

    def setUp(self):
        self.calls = []
        self.config = module(PACKAGE + ".reuse.config", enabled=lambda: False)
        self.patcher = patch.dict(
            sys.modules, {
                PACKAGE: module(PACKAGE, __path__=[str(BACKEND)]),
                PACKAGE + ".reuse": module(PACKAGE + ".reuse", __path__=[]),
                PACKAGE + ".reuse.config": self.config,
            })
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.batching = importlib.import_module(PACKAGE + ".block_batching")
        self.dispatch = self.batching
        capture = patch.object(self.batching, "_capture_state", return_value=False)
        self.capture = capture.start()
        self.addCleanup(capture.stop)
        self.backend = SimpleNamespace(target=SimpleNamespace(arch="Ascend950"),
                                       prepare_block_compile_options=lambda values, **_: dict(values))
        self.fn = SimpleNamespace(fn=lambda: None, cache_key="test-kernel", arg_names=["x", "size", "BLOCK"],
                                  device_caches={0: [{}, {}, None, None, self.bind]})
        self.kwargs = dict(block_batching_factor="auto", BLOCK=512)
        self.result = object()

    def bind(self, *args, **kwargs):
        self.calls.append("bind")
        return dict(x=args[0], size=args[1], BLOCK=kwargs["BLOCK"]), [("*fp32", None), ("i32", None),
                                                                      ("constexpr", 512)], kwargs

    def tail(self, *args, **kwargs):
        self.calls.append((args, kwargs))
        return self.result

    def run_handler(self, grid=(256, ), warmup=False):
        return self.batching.get_launch_handler().run(self.fn, (Tensor(), 131072), self.kwargs, grid, warmup, 0, 42,
                                                      self.backend, self.tail)

    def test_cached_proof_shares_data_values_only_within_one_abi(self):

        def bind(output, seed, offset, **options):
            # A kernel that disables alignment specialization for these runtime integers.
            seed_type = "i32" if seed < 2**31 else "u32"
            return dict(output=output, seed=seed, offset=offset), [("*fp32", None), (seed_type, None),
                                                                   ("i32", None)], options

        self.fn.arg_names = ["output", "seed", "offset"]
        self.fn.device_caches[0][4] = bind
        kernel = Mock(spec=self.batching.BatchKernel)
        kernel.block_auto_data_arguments = ("seed", )
        seen = []

        def tail(args, kwargs, *more, **hooks):
            seen.append((args[1:], json.loads(kwargs["block_auto_contract"])["arguments"]))
            return kernel

        def launch(seed, offset=32):
            args = (Tensor(), seed, offset)
            self.batching.get_launch_handler().run(self.fn, args, {}, (256, ), False, 0, 42, self.backend, tail)
            self.assertEqual(seen[-1][0], (seed, offset))
            return seen[-1][1]

        self.assertEqual(launch(8)["seed"]["value"], 8)  # A cold family needs the native TTIR proof.
        shared = launch(8)
        self.assertNotIn("value", shared["seed"])
        self.assertEqual(launch(16), shared)
        self.assertEqual(launch(16, 64)["offset"]["value"], 64)
        for value in (-1, 0, 1):
            self.assertEqual(launch(value)["seed"]["value"], value)
        self.assertEqual(launch(2**31)["seed"]["value"], 2**31)  # A new ABI needs its own proof.
        self.fn.cache_key = "different-source"
        self.assertEqual(launch(8)["seed"]["value"], 8)

    def test_auto_contract_resolves_grid_once_and_excludes_constexpr(self):

        def grid(bound):
            self.calls.append("grid")
            return (bound["size"] // bound["BLOCK"], )

        self.assertIs(self.run_handler(grid), self.result)
        self.assertEqual(self.calls[:2], ["bind", "grid"])
        self.assertEqual(len(self.calls), 3)
        args, kwargs = self.calls[-1]
        self.assertEqual(args[2], (256, ))
        contract = json.loads(args[1]["block_auto_contract"])
        self.assertEqual(contract["argument_order"], ["x", "size"])
        self.assertEqual(contract["grid"], [256, 1, 1])
        self.assertEqual(contract["arguments"]["x"]["shape"], [256, 512])
        self.assertNotIn("block_auto_contract", self.kwargs)

    def test_omitted_factor_matches_explicit_auto(self):
        explicit, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), 131072), self.kwargs, (256, ), False, 0,
                                                   self.backend)
        self.kwargs.pop("block_batching_factor")
        self.calls.clear()
        self.run_handler()
        prepared = self.calls[-1][0][1]
        self.assertEqual(prepared, explicit)
        self.assertEqual(list(prepared), list(explicit))
        self.assertEqual(prepared["block_batching_factor"], "auto")
        self.assertNotIn("block_batching_factor", self.kwargs)

    def test_generic_autotune_default_and_explicit_options(self):
        from contextlib import nullcontext
        from triton.runtime import autotuner

        def launch(*args, grid, **options):
            prepared, _ = self.dispatch.prepare_launch(self.fn, args, options, grid, False, 0, self.backend)
            # An omitted factor is the ordinary compilation, which the option
            # default already spells as 1, so the key stays out of the options.
            seen.append(prepared.get("block_batching_factor", 1))
            return prepared

        def bench(fn, quantiles):
            fn()
            return 1.0

        def hook(*args, **kwargs):
            self.assertIsNone(autotuner._autotuning_kernel.get())

        self.fn.run = self.fn.warmup = launch
        for source, factor in (("default", None), ("call", "auto"), ("config", "auto"), ("call", 2), ("config", 2),
                               ("call", 4), ("config", 4)):
            with self.subTest(source=source, factor=factor):
                seen = []
                self.calls.clear()
                self.capture.reset_mock()
                option = {} if factor is None else dict(block_batching_factor=factor)
                configs = [
                    autotuner.Config(dict(BLOCK=b, **(option if source == "config" else {})), pre_hook=hook)
                    for b in (256, 512)
                ]
                tuned = autotuner.Autotuner(self.fn, self.fn.arg_names, configs, ["size"], None, None, pre_hook=hook,
                                            post_hook=hook, do_bench=bench)
                options = dict(grid=(256, ), **(option if source == "call" else {}))
                with patch.object(autotuner, "_exact_request_scope", nullcontext):
                    tuned.warmup(Tensor(), 131072, **options)
                    tuned.run(Tensor(), 131072, **options)
                    tuned.run(Tensor(), 131072, **options)  # Cached best configuration.
                self.assertEqual(seen, [1 if factor is None else factor] * 6)
                if factor is None:
                    self.assertEqual(self.calls, [])
                    self.capture.assert_not_called()
                self.assertIsNone(autotuner._autotuning_kernel.get())
                direct = launch(Tensor(), 131072, grid=(256, ), BLOCK=512)
                self.assertEqual(direct["block_batching_factor"], "auto")

    def test_generic_autotune_scope_resets_on_failure(self):
        from contextlib import nullcontext
        from triton.runtime import autotuner

        def fail(*args, **kwargs):
            self.assertIs(autotuner._autotuning_kernel.get(), self.fn.fn)
            raise RuntimeError("kernel failed")

        self.fn.run = self.fn.warmup = fail
        config = autotuner.Config(dict(BLOCK=512))
        tuned = autotuner.Autotuner(self.fn, self.fn.arg_names, [config], [], None, None, do_bench=lambda fn, **_: fn())
        tuned.nargs = {}
        with patch.object(autotuner, "_exact_request_scope", nullcontext):
            for invoke in (lambda: tuned._bench(config=config), tuned.run, tuned.warmup):
                with self.assertRaisesRegex(RuntimeError, "kernel failed"):
                    invoke()
                self.assertIsNone(autotuner._autotuning_kernel.get())

    def test_generic_autotune_scope_is_kernel_and_thread_local(self):
        from concurrent.futures import ThreadPoolExecutor
        from triton.runtime import autotuner

        self.kwargs.pop("block_batching_factor")
        other = lambda: None
        with ThreadPoolExecutor(max_workers=1) as pool, autotuner._autotune_scope(self.fn.fn):
            self.assertIsNone(pool.submit(autotuner._autotuning_kernel.get).result())
            with autotuner._autotune_scope(other):
                self.run_handler()
                self.assertEqual(self.calls[-1][0][1]["block_batching_factor"], "auto")
            self.assertIs(autotuner._autotuning_kernel.get(), self.fn.fn)
            self.run_handler()
            self.assertNotIn("block_batching_factor", self.calls[-1][0][1])
            self.fn.arg_names.append("block_batching_factor")
            options, _ = self.dispatch.prepare_launch(self.fn, (), self.kwargs, (256, ), True, 0, self.backend)
            self.assertIs(options, self.kwargs)  # A kernel argument is not a compiler option.
        self.assertIsNone(autotuner._autotuning_kernel.get())

    def test_incomplete_tensor_metadata_does_not_break_default_launch(self):
        wrapper = SimpleNamespace(dtype="fp32", shape=(256, 512), stride=lambda: (512, 1))
        self.assertEqual(self.dispatch._argument(wrapper), dict(kind="unsupported"))

    def test_float_launch_arguments_keep_finite_values_only(self):
        for value in (0.75, float("inf"), float("nan")):
            with self.subTest(value=value):
                options, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), value), self.kwargs, (256, ), False, 0,
                                                          self.backend)
                argument = json.loads(options["block_auto_contract"])["arguments"]["size"]
                expected = dict(kind="scalar", value=value) if value == 0.75 else dict(kind="unsupported")
                self.assertEqual(argument, expected)

    def test_explicit_factors_keep_manual_dispatch(self):
        for factor in (1, 2, 4):
            context = ("Ascend950", 24 // factor, (True, 24 // factor, 48 // factor), None)
            self.backend.prepare_block_compile_options = Mock(
                side_effect=lambda values, **_: dict(values, block_compile_context=context))
            self.kwargs = dict(BLOCK=512, block_batching_axis=0)
            self.kwargs["block_batching_factor"] = factor
            self.calls.clear()
            self.assertIs(self.run_handler(), self.result)
            self.assertEqual(len(self.calls), 1)
            if factor == 1:
                self.assertIs(self.calls[0][0][1], self.kwargs)
                self.backend.prepare_block_compile_options.assert_not_called()
            else:
                self.assertEqual(self.calls[0][0][1], dict(self.kwargs, block_compile_context=context))
                self.backend.prepare_block_compile_options.assert_called_once_with(self.kwargs,
                                                                                   cache=self.fn.device_caches[0][1])

    def test_warmup_uses_the_same_contract_as_launch(self):
        self.kwargs.pop("block_batching_factor")
        self.run_handler(warmup=True)
        warmed = self.calls[-1][0][1]
        self.run_handler()
        self.assertEqual(warmed, self.calls[-1][0][1])
        self.assertEqual(warmed["block_batching_factor"], "auto")
        incomplete = SimpleNamespace(dtype="fp32", shape=(1, ), stride=lambda: (1, ))
        options, _ = self.dispatch.prepare_launch(self.fn, (incomplete, 131072), self.kwargs, (256, ), True, 0,
                                                  self.backend)
        self.assertNotIn("block_batching_factor", options)
        self.assertNotIn("block_auto_contract", options)

    def test_capture_uses_baseline_without_binding_or_grid_evaluation(self):
        self.kwargs.pop("block_batching_factor")

        def unavailable(_):
            self.fail("grid must stay lazy")

        for warmup, capture in ((True, True), (False, True), (False, None)):
            self.calls.clear()
            self.capture.return_value = capture
            self.run_handler(unavailable, warmup)
            self.assertEqual(len(self.calls), 1)
            values, _ = self.calls[0]
            self.assertNotIn("block_batching_factor", values[1])
            self.assertIs(values[2], unavailable)

    def test_non_a5_and_invalid_grid_keep_baseline(self):
        self.kwargs.pop("block_batching_factor")
        for target, grid in (("Ascend310B4", (256, )), ("Ascend950", (0, ))):
            self.backend.target.arch = target
            self.calls.clear()
            self.run_handler(grid)
            self.assertNotIn("block_batching_factor", self.calls[-1][0][1])

    def test_legacy_targets_enter_auto_without_a5_empirical_override(self):
        # A5 calibrated the short-batch heuristic; the earlier generation
        # admits the same automatic launches but scores them without it.
        for target in ("Ascend910_9391", "Ascend910B1", "Ascend910B2", "Ascend910B3", "Ascend910B4"):
            self.backend.target.arch = target
            self.calls.clear()
            self.run_handler()
            options = self.calls[-1][0][1]
            self.assertEqual(options["block_batching_factor"], "auto", target)
            self.assertFalse(json.loads(options["block_auto_contract"])["cv_empirical_short_batch"], target)

    def test_single_program_keeps_normal_cache_without_contract(self):
        for grid, axis in (((1, ), -1), ((1, 1, 1), -1), ((8, 1), 1)):
            self.kwargs["block_batching_axis"] = axis
            self.calls.clear()
            self.run_handler(grid)
            options = self.calls[-1][0][1]
            self.assertNotIn("block_batching_factor", options)
            self.assertNotIn("block_auto_contract", options)

    def test_other_frontends_keep_their_original_compilation_path(self):
        self.fn.ASTSource = SimpleNamespace(frontend_kind="tla")
        self.kwargs.pop("block_batching_factor")
        self.run_handler()
        self.assertEqual(len(self.calls), 1)
        self.assertNotIn("block_batching_factor", self.calls[0][0][1])

    def test_shape_and_scalar_values_are_part_of_jit_option_identity(self):
        first, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), 131072), self.kwargs, (256, ), False, 0,
                                                self.backend)
        second, _ = self.dispatch.prepare_launch(self.fn, (Tensor(), 65536), self.kwargs, (128, ), False, 0,
                                                 self.backend)
        self.assertNotEqual(first["block_auto_contract"], second["block_auto_contract"])

    def test_reuse_keeps_original_grid_and_unfragmented_options(self):
        self.config.enabled = lambda: True
        grid = Mock(side_effect=AssertionError("reuse must receive the unevaluated grid"))

        def reuse(fn, args, kwargs, grid, warmup, device, stream, backend, tail):
            self.assertIs(fn, self.fn)
            self.assertNotIn("block_auto_contract", kwargs)
            self.assertNotIn("block_batching_factor", kwargs)
            return tail(args, kwargs, grid, warmup, device, stream)

        with patch.dict(sys.modules,
                        {PACKAGE + ".reuse.bridge": module("bridge", get_handler=lambda: SimpleNamespace(run=reuse))}):
            self.assertIs(self.run_handler(grid), self.result)
        grid.assert_not_called()
        self.assertNotIn("bind", self.calls)
        self.assertEqual(len([item for item in self.calls if isinstance(item, tuple)]), 1)

    def test_equivalent_auto_decisions_share_device_handle_and_identity(self):
        cache = {}

        class Report(dict):

            def __repr__(self):
                representations.append(self)
                return super().__repr__()

        representations = []

        class Kernel:
            hash = "same-artifact"
            block_batching = Report(reason="DISABLED")
            block_auto_selection = Report(selected_factor=1, reason="GRID_BELOW_CORE_FLOOR")

            def __init__(self):
                self._init_handles = Mock()

        first, second = Kernel(), Kernel()
        with patch.object(self.batching, "BatchKernel", Kernel):
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, first), first)
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, first), first)
            first._init_handles.assert_called_once()
            self.assertEqual(len(representations), 2)
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, second), first)
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, second), first)
            self.assertEqual(len(representations), 4)
            self.assertEqual(first._init_handles.call_count, 2)
            second._init_handles.assert_not_called()
            # A different device/native cache cannot reuse the previous handle.
            self.assertIs(self.dispatch.initialize_auto_kernel({}, second), second)
            changed = Kernel()
            changed.block_auto_selection = {"selected_factor": 1, "reason": "MODEL_INCOMPLETE"}
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, changed), changed)
            cache.clear()
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, second), second)

    def test_failed_auto_initialization_is_retried(self):

        class Kernel:
            hash = "artifact"
            block_batching = {"reason": "DISABLED"}
            block_auto_selection = {"selected_factor": 1}

        kernel = Kernel()
        kernel._init_handles = Mock(side_effect=[RuntimeError("load failed"), None])
        cache = {}
        with patch.object(self.batching, "BatchKernel", Kernel):
            with self.assertRaisesRegex(RuntimeError, "load failed"):
                self.dispatch.initialize_auto_kernel(cache, kernel)
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, kernel), kernel)
            self.assertIs(self.dispatch.initialize_auto_kernel(cache, kernel), kernel)
        self.assertEqual(kernel._init_handles.call_count, 2)

    def test_launch_exception_is_not_replayed(self):

        def fail(*args, **kwargs):
            self.calls.append("launch")
            raise RuntimeError("launch error")

        self.tail = fail
        with self.assertRaisesRegex(RuntimeError, "launch error"):
            self.run_handler()
        self.assertEqual(self.calls.count("launch"), 1)
