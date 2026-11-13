# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU regressions for the ordinary Ascend continuation and bundle protocol.

This module also runs with stdlib unittest on a host without Triton or a device.
Only compiler artifacts and the device launch boundary are substituted.
"""
from collections import namedtuple
from concurrent.futures import ThreadPoolExecutor
import hashlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
import tempfile
import threading
import time
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

try:
    import pytest
except ImportError:
    pass
else:

    @pytest.fixture(scope="module", autouse=True)
    def assign_npu():
        """These tests must never initialize a device."""


BACKEND = Path(__file__).resolve().parents[2] / "backend"
PACKAGE = "_block_auto_host_test"


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


def load(name):
    full = PACKAGE + "." + name
    spec = importlib.util.spec_from_file_location(full, BACKEND / (name + ".py"))
    result = importlib.util.module_from_spec(spec)
    sys.modules[full] = result
    spec.loader.exec_module(result)
    return result


class Tensor:
    dtype = "torch.float32"
    safe = True

    def __init__(self, shape=(128, ), stride=(1, ), offset=0):
        self.shape, self._stride, self.offset = shape, stride, offset

    def stride(self):
        return self._stride

    def storage_offset(self):
        return self.offset

    def data_ptr(self):
        raise AssertionError("selection must not query a physical device pointer")


class TestAutomaticJIT(unittest.TestCase):

    def setUp(self):
        self.launches, self.compiles, self.guards = [], [], []
        self.model_calls, self.freeze_calls = [], []
        self.selection = 2
        self.selection_axis = None
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.cache_root = Path(temporary.name)

        def cache_manager(key):
            directory = self.cache_root / key
            directory.mkdir(exist_ok=True)

            def put(text, name, **kwargs):
                path = directory / name
                path.write_text(text)
                return str(path)

            return SimpleNamespace(
                put=put, get_file=lambda name: str(directory / name) if (directory / name).is_file() else None)

        package = module(PACKAGE, __path__=[str(BACKEND)])
        knobs = SimpleNamespace(ascend=SimpleNamespace(compile_only=False),
                                runtime=SimpleNamespace(launch_enter_hook=None, launch_exit_hook=None))
        self.modules = {
            PACKAGE:
            package,
            "triton":
            module("triton", knobs=knobs),
            "triton.compiler":
            module("triton.compiler"),
            "triton.compiler.errors":
            module("triton.compiler.errors", MLIRCompilationError=type("MLIRCompilationError", (Exception, ), {})),
            "triton.runtime":
            module("triton.runtime"),
            "triton.runtime.errors":
            module("triton.runtime.errors", OutOfResources=type("OutOfResources", (Exception, ), {})),
            "triton.runtime.cache":
            module("triton.runtime.cache", get_cache_manager=cache_manager),
            PACKAGE + ".utils":
            module(PACKAGE + ".utils", backend_policy=None),
            PACKAGE + ".block_batching_report":
            module(PACKAGE + ".block_batching_report", VALIDATION_POLICY="vv-policy", CV_VALIDATION_POLICY="cv-policy",
                   validate_vv_loop_contract=lambda _: False),
            PACKAGE + ".block_batching_guard_v3":
            module(PACKAGE + ".block_batching_guard_v3", validate_plan_binding=lambda _: True),
            PACKAGE + ".reuse":
            module(PACKAGE + ".reuse", __path__=[]),
            PACKAGE + ".reuse.config":
            module(PACKAGE + ".reuse.config", enabled=lambda: False),
            PACKAGE + ".block_auto_model":
            module(PACKAGE + ".block_auto_model", VERSION="test-model", select_candidates=self.select),
        }

        def guard(plan, factor):

            def check(arguments, grid):
                self.guards.append((arguments, grid))
                return arguments[0].safe, "OK" if arguments[0].safe else "ALIAS_REJECTED"

            return check

        self.modules[PACKAGE + ".block_batching_guard"] = module(PACKAGE + ".block_batching_guard", RuntimeGuard=guard,
                                                                 integer=int, I32_MAX=2147483647)
        self.patcher = patch.dict(sys.modules, self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.batching = load("block_batching")
        self.dispatch = load("block_auto_dispatch")
        self.batching._capture_state = lambda: False
        self.batching._core_floor = lambda **kwargs: 0
        self.batching._freeze_source = self.freeze
        self.backend = SimpleNamespace(
            target=SimpleNamespace(arch="Ascend950PR_9579"), parse_options=lambda opts: SimpleNamespace(
                is_pure_simt=False, compile_mode="simd", block_batching_backend_revision="compiler-v1"))
        self.fn = SimpleNamespace(arg_names=["x", "n", "BLOCK"])
        self.fn.device_caches = {0: ({}, {}, None, self.backend, self.bind)}
        self.src = SimpleNamespace(fn=self.fn, signature={"x": "*fp32", "n": "i32", "BLOCK": "constexpr"},
                                   constants={"BLOCK": 32}, hash=lambda: "source-v1")

    def bind(self, x, n=128, **kwargs):
        return {"x": x, "n": n, "BLOCK": 32}, [("*fp32", None), ("i32", None), ("constexpr", 32)], dict(kwargs)

    def freeze(self, backend, src, options, revision):
        self.freeze_calls.append((src, options, revision))
        return src

    def select(self, candidates, contract):
        self.model_calls.append((candidates, contract))
        available = [item for item in candidates if item.get("kernel") is not None]
        matches = [
            item for item in available
            if item["factor"] == self.selection and (self.selection_axis is None or item["axis"] == self.selection_axis)
        ]
        chosen = matches[0] if matches else next(item for item in available if item["factor"] == 1)
        return chosen["factor"], {
            "reason": "CPU_TEST_SCORE", "model_complete": True, "selected_axis": chosen["axis"], "selected_key":
            self.dispatch._candidate_key(chosen["axis"], chosen["factor"])
        }

    def compile(self, options, frozen):
        self.compiles.append((dict(options), frozen))
        factor = options["block_batching_factor"]
        self.assertIs(type(factor), int)
        self.assertNotIn("block_auto_contract", options)
        self.assertTrue(options["block_auto_collect"])
        binary = ("binary" + str(factor)).encode()
        report = {
            "compile_status": "COMPILED_GUARDED", "owner": "block_batching", "cv_mode": "disabled", "validation_policy":
            "vv-policy", "implementation_version": self.batching.IMPLEMENTATION_VERSION, "lane_width": factor,
            "selected_axis": 0, "guard": {"plan_fingerprint": "plan"}
        }
        fields = {
            "block_batching": report, "block_batching_binary_sha256": hashlib.sha256(binary).hexdigest(),
            "block_batching_plan_sha256": "plan"
        }
        kernel = SimpleNamespace(hash="hash" + str(factor), kernel=binary,
                                 metadata=namedtuple("Metadata", fields)(**fields), function="function" + str(factor),
                                 packed_metadata="packed" + str(factor), launch_metadata=lambda *args: None, _run=True,
                                 _init_handles=lambda: None, run=lambda *args: self.launches.append((factor, args)))
        return kernel

    def prepare(self, tensor=None, n=128, grid=(128, ), **kwargs):
        return self.dispatch.prepare_launch(self.fn, (tensor or Tensor(), n), kwargs, grid, False, 0, self.backend)

    def bundle(self, **kwargs):
        options, _ = self.prepare(**kwargs)
        return self.batching.compile_bundle(self.backend, self.src, options, self.compile)

    def test_contract_contains_runtime_shape_stride_scalar_but_not_constexpr_or_pointer(self):
        options, _ = self.prepare(Tensor((4, 32), (32, 1), 8), n=77)
        contract = json.loads(options["block_auto_contract"])
        self.assertEqual(contract["argument_order"], ["x", "n"])
        self.assertEqual(
            contract["arguments"]["x"],
            {"kind": "tensor", "dtype": "torch.float32", "shape": [4, 32], "stride": [32, 1], "storage_offset": 8})
        self.assertEqual(contract["arguments"]["n"], {"kind": "scalar", "value": 77})
        self.assertEqual(options["block_batching_factor"], "auto")

    def test_decision_key_changes_with_shape_stride_scalar_grid(self):
        original = self.prepare()[0]["block_auto_contract"]
        for arguments in ({"tensor": Tensor((256, ))}, {"tensor": Tensor(stride=(2, ))}, {"n": 129}, {"grid": (129, )}):
            self.assertNotEqual(original, self.prepare(**arguments)[0]["block_auto_contract"])
        self.assertEqual(original, self.prepare(Tensor())[0]["block_auto_contract"])
        with patch.dict("os.environ", {"NPU_DEVICE_LIMIT": "0:16"}):
            self.assertNotEqual(original, self.prepare()[0]["block_auto_contract"])

    def test_callable_is_evaluated_once_and_matches_static_key(self):
        calls = []

        def grid(bound):
            calls.append(bound)
            return (bound["n"], )

        options, resolved = self.prepare(grid=grid)
        self.assertEqual(resolved, (128, ))
        self.assertEqual(len(calls), 1)
        self.assertEqual(options["block_auto_contract"], self.prepare()[0]["block_auto_contract"])

    def test_warmup_and_manual_factors_do_not_bind_or_resolve(self):

        def fail(*args, **kwargs):
            raise AssertionError("unexpected binding or grid callback")

        self.fn.device_caches[0] = (None, None, None, None, fail)
        for options, warmup in (({}, True), ({"block_batching_factor": 1}, False), ({"block_batching_factor":
                                                                                     4}, False)):
            actual, grid = self.dispatch.prepare_launch(self.fn, (), options, fail, warmup, 0, self.backend)
            self.assertIs(actual, options)
            self.assertIs(grid, fail)

    def test_explicit_auto_resolves_the_same_contract_as_default(self):
        explicit, _ = self.prepare(block_batching_factor="auto")
        ordinary, _ = self.prepare()
        self.assertEqual(explicit, ordinary)
        self.assertEqual(self.bundle(block_batching_factor="auto").request.factor, 2)

    def test_explicit_auto_warmup_and_empty_grid_use_integer_baseline(self):

        def fail(*args, **kwargs):
            raise AssertionError("warmup must not bind or resolve a callable")

        self.fn.device_caches[0] = (None, None, None, None, fail)
        result, grid = self.dispatch.prepare_launch(self.fn, (), {"block_batching_factor": "auto"}, fail, True, 0,
                                                    self.backend)
        self.assertEqual(result["block_batching_factor"], 1)
        self.assertIs(grid, fail)
        self.fn.device_caches[0] = ({}, {}, None, self.backend, self.bind)
        result, grid = self.prepare(grid=(0, ), block_batching_factor="auto")
        self.assertEqual(result["block_batching_factor"], 1)
        self.assertEqual(grid, (0, ))

    def test_kernel_argument_named_like_option_is_not_parsed_as_option(self):
        tensor = Tensor()
        self.fn.arg_names.append("block_auto_collect")

        def binder(*args, **kwargs):
            bound, specialization, options = self.bind(*args, **kwargs)
            bound["block_auto_collect"] = options.pop("block_auto_collect")
            return bound, specialization + [("*fp32", None)], options

        self.fn.device_caches[0] = ({}, {}, None, self.backend, binder)
        parse = self.backend.parse_options
        seen = []

        def parse_options(options):
            self.assertNotIn("block_auto_collect", options)
            seen.append(options)
            return parse(options)

        self.backend.parse_options = parse_options
        result, _ = self.prepare(block_auto_collect=tensor)
        self.assertIs(result["block_auto_collect"], tensor)
        self.assertEqual(len(seen), 1)
        self.assertIn("block_auto_collect", json.loads(result["block_auto_contract"])["arguments"])

    def test_capture_or_unknown_capture_query_preserves_warmup_baseline_key(self):

        def fail(*args, **kwargs):
            raise AssertionError("capture must not bind, compile or resolve a callable")

        self.fn.device_caches[0] = (None, None, None, None, fail)
        for state in (True, None):
            self.batching._capture_state = lambda: state
            for original in ({}, {"block_batching_factor": "auto"}):
                warmup, _ = self.dispatch.prepare_launch(self.fn, (), original, fail, True, 0, self.backend)
                actual, grid = self.dispatch.prepare_launch(self.fn, (), original, fail, False, 0, self.backend)
                self.assertEqual(actual, warmup)
                self.assertIs(grid, fail)

    def test_automatic_capture_bypasses_enabled_reuse_but_manual_keeps_it(self):
        forwarded, tails = [], []

        def tail(args, kwargs, grid, warmup, device, stream):
            tails.append(kwargs)
            return kwargs

        def reuse(fn, args, kwargs, grid, warmup, device, stream, backend, jit_tail):
            forwarded.append(kwargs)
            return jit_tail(args, kwargs, grid, warmup, device, stream)

        sys.modules[PACKAGE + ".reuse.config"].enabled = lambda: True
        sys.modules[PACKAGE + ".reuse.bridge"] = module(PACKAGE + ".reuse.bridge",
                                                        get_handler=lambda: SimpleNamespace(run=reuse))

        def fail(*args, **kwargs):
            raise AssertionError("capture must not bind or resolve the launch grid")

        self.fn.device_caches[0] = (None, None, None, None, fail)
        handler = self.batching.get_launch_handler()
        for state in (True, None):
            self.batching._capture_state = lambda: state
            for options in ({}, {"block_batching_factor": "auto"}):
                result = handler.run(self.fn, (), options, fail, False, 0, 0, self.backend, tail)
                self.assertEqual(result.get("block_batching_factor", 1), 1)
                self.assertNotIn("block_auto_contract", result)
            self.assertFalse(forwarded)
        handler.run(self.fn, (), {"block_batching_factor": 1}, fail, False, 0, 0, self.backend, tail)
        self.assertEqual(len(forwarded), 1)
        self.assertEqual(len(tails), 5)

    def test_automatic_eager_launch_still_enters_enabled_reuse(self):
        forwarded = []

        def reuse(fn, args, kwargs, grid, warmup, device, stream, backend, jit_tail):
            forwarded.append(kwargs)
            return jit_tail(args, kwargs, grid, warmup, device, stream)

        sys.modules[PACKAGE + ".reuse.config"].enabled = lambda: True
        sys.modules[PACKAGE + ".reuse.bridge"] = module(PACKAGE + ".reuse.bridge",
                                                        get_handler=lambda: SimpleNamespace(run=reuse))
        result = self.batching.get_launch_handler().run(self.fn, (Tensor(), 128), {}, (128, ), False, 0, 0,
                                                        self.backend, lambda args, kwargs, *rest: kwargs)
        self.assertEqual(result["block_batching_factor"], "auto")
        self.assertEqual(len(forwarded), 1)

    def test_empty_grid_preserves_baseline_path(self):
        options, grid = self.prepare(grid=(0, ))
        self.assertNotIn("block_batching_factor", options)
        self.assertEqual(grid, (0, ))

    def test_callable_error_propagates_without_compile(self):

        def fail(bound):
            raise ValueError("bad grid")

        with self.assertRaisesRegex(ValueError, "bad grid"):
            self.prepare(grid=fail)
        self.assertFalse(self.compiles)

    def test_bundle_freezes_once_compiles_three_and_preserves_shared_baseline(self):
        result = self.bundle()
        self.assertEqual([options["block_batching_factor"] for options, _ in self.compiles], [1, 2, 4])
        self.assertEqual(len(self.freeze_calls), 1)
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual(result.request.factor, 2)
        self.assertEqual(result.baseline.hash, "hash1")
        self.assertNotIn("block_batching_grid", self.compiles[0][0])
        self.assertFalse(self.launches)

    def test_guards_are_rechecked_on_every_dispatch(self):
        result = self.bundle()
        tensor = Tensor()
        for safe in (True, False, True):
            tensor.safe = safe
            result.run(128, 1, 1, 0, result.function, result.packed_metadata, None, None, None, tensor, 128, 32)
        self.assertEqual([factor for factor, _ in self.launches], [2, 1, 2])
        self.assertEqual(len(self.guards), 3)
        self.assertEqual(self.launches[1][1][4:6], ("function1", "packed1"))

    def test_candidate_admission_failure_has_no_layout_retry(self):
        original = self.compile
        attempts = []

        def compile(options, frozen):
            factor = options["block_batching_factor"]
            attempts.append(factor)
            if factor == 2:
                raise self.batching.LayoutTargetRejected(2, -1, "UB rejection")
            return original(options, frozen)

        options, _ = self.prepare()
        result = self.batching.compile_bundle(self.backend, self.src, options, compile)
        self.assertEqual(attempts, [1, 2, 4])
        self.assertEqual(result.request.factor, 1)
        self.assertEqual(result.block_auto_selection["rejections"][0]["factor"], 2)

    def test_baseline_and_internal_compiler_errors_propagate(self):
        options, _ = self.prepare()

        def baseline_error(options, frozen):
            raise RuntimeError("baseline compiler failed")

        with self.assertRaisesRegex(RuntimeError, "baseline compiler failed"):
            self.batching.compile_bundle(self.backend, self.src, options, baseline_error)
        original = self.compile

        def internal_error(options, frozen):
            if options["block_batching_factor"] == 2:
                raise subprocess.CalledProcessError(139, "compiler", stderr=b"LLVM ERROR: verifier")
            return original(options, frozen)

        with self.assertRaises(subprocess.CalledProcessError):
            self.batching.compile_bundle(self.backend, self.src, options, internal_error)

    def test_binary_mismatch_cannot_enter_scoring(self):
        original = self.compile

        def compile(options, frozen):
            kernel = original(options, frozen)
            if options["block_batching_factor"] == 2:
                kernel.kernel = b"changed"
            return kernel

        options, _ = self.prepare()
        result = self.batching.compile_bundle(self.backend, self.src, options, compile)
        self.assertEqual(result.request.factor, 1)
        self.assertEqual([item["factor"] for item in self.model_calls[0][0] if item.get("kernel") is not None], [1, 4])
        self.assertEqual(self.model_calls[0][0][-1]["status"], "COMPILE_REJECTED")

    def test_invalid_core_floor_propagates_and_zero_is_preserved(self):
        with patch.dict("os.environ", {"TA_BLOCK_BATCHING_CORE_FLOOR": "invalid"}):
            with self.assertRaises(ValueError):
                self.prepare()
        with patch.dict("os.environ", {"TA_BLOCK_BATCHING_CORE_FLOOR": "0"}):
            self.assertEqual(json.loads(self.prepare()[0]["block_auto_contract"])["core_floor"], "0")

    def test_handler_uses_same_tail_and_jit_cache(self):
        cache, callbacks = {}, []

        def tail(args, kwargs, grid, warmup, device, stream):
            key = kwargs["block_auto_contract"]
            if key not in cache:
                cache[key] = self.batching.compile_bundle(self.backend, self.src, kwargs, self.compile)
            return cache[key]

        def grid(bound):
            callbacks.append(1)
            return (bound["n"], )

        handler = self.batching.get_launch_handler()
        first = handler.run(self.fn, (Tensor(), 128), {}, grid, False, 0, 0, self.backend, tail)
        second = handler.run(self.fn, (Tensor(), 128), {}, grid, False, 0, 0, self.backend, tail)
        third = handler.run(self.fn, (Tensor(), 256), {}, grid, False, 0, 0, self.backend, tail)
        self.assertIs(first, second)
        self.assertIsNot(first, third)
        self.assertEqual(len(callbacks), 3)
        self.assertEqual(len(self.compiles), 6)

    def test_persistent_cache_restores_only_baseline_and_choice_without_scoring(self):
        original = self.bundle()
        self.dispatch = load("block_auto_dispatch")
        restored = self.bundle()
        self.assertEqual(restored.request.factor, original.request.factor)
        self.assertEqual([options["block_batching_factor"] for options, _ in self.compiles], [1, 2, 4, 1, 2])
        self.assertEqual(len(self.model_calls), 1)
        self.assertTrue(restored.block_auto_selection["decision_cache_hit"])

    def test_corrupt_cached_choice_is_recomputed_within_three_variants(self):
        self.bundle()
        path = next(self.cache_root.glob("*/block_auto.selection.json"))
        data = json.loads(path.read_text())
        data["selected_binary"]["binary_sha256"] = "tampered"
        path.write_text(json.dumps(data))
        result = self.bundle()
        self.assertEqual(len(self.model_calls), 2)
        self.assertFalse(result.block_auto_selection["decision_cache_hit"])
        self.assertEqual([options["block_batching_factor"] for options, _ in self.compiles], [1, 2, 4, 1, 2, 4])

    def test_cached_baseline_needs_no_candidate_compile(self):
        self.selection = 1
        self.bundle()
        result = self.bundle()
        self.assertEqual(result.request.factor, 1)
        self.assertEqual([options["block_batching_factor"] for options, _ in self.compiles], [1, 2, 4, 1])
        self.assertEqual(len(self.model_calls), 1)

    def cv_compiler(self):
        original = self.compile

        def compile(options, frozen):
            kernel = original(options, frozen)
            factor, axis = options["block_batching_factor"], options["block_batching_axis"]
            fields = dict(kernel.metadata._asdict(), mix_mode="mix", parallel_mode="simd")
            if factor > 1:
                fields["block_batching"] = dict(fields["block_batching"], selected_axis=axis)
                kernel.kernel = f"candidate-{axis}-{factor}".encode()
                kernel.hash = f"hash-{axis}-{factor}"
                fields["block_batching_binary_sha256"] = hashlib.sha256(kernel.kernel).hexdigest()
            kernel.metadata = namedtuple("CVMetadata", fields)(**fields)
            return kernel

        return compile

    def test_cv_two_f4_axes_survive_selection_and_persistent_restore(self):
        plan = {
            "candidates": [{"key": "a1_f4", "axis": 1, "factor": 4, "merge_kind": "M"},
                           {"key": "a2_f4", "axis": 2, "factor": 4, "merge_kind": "N"}], "axis_roles":
            {"1": "M", "2": "N"}, "front_end_probe_count":
            6
        }
        sys.modules[PACKAGE + ".block_auto_cv"] = module(PACKAGE + ".block_auto_cv",
                                                         candidate_plan=lambda *a, **k: plan)
        self.selection, self.selection_axis = 4, 2
        options, _ = self.prepare(grid=(8, 8, 8))
        compiler = self.cv_compiler()
        first = self.batching.compile_bundle(self.backend, self.src, options, compiler)
        self.assertEqual((first.request.axis, first.request.factor), (2, 4))
        self.assertEqual([(item["axis"], item["factor"]) for item in self.model_calls[0][0]], [(-1, 1), (1, 4), (2, 4)])
        restored = self.batching.compile_bundle(self.backend, self.src, options, compiler)
        self.assertEqual((restored.request.axis, restored.request.factor), (2, 4))
        self.assertEqual(len(self.model_calls), 1)
        self.assertEqual([(opts["block_batching_axis"], opts["block_batching_factor"]) for opts, _ in self.compiles],
                         [(-1, 1), (1, 4), (2, 4), (-1, 1), (2, 4)])

    def test_empty_cv_plan_returns_baseline_without_model_or_backend_candidates(self):
        plan = {"candidates": [], "reason": "AXIS_ROLE_UNKNOWN", "front_end_probe_count": 6}
        sys.modules[PACKAGE + ".block_auto_cv"] = module(PACKAGE + ".block_auto_cv",
                                                         candidate_plan=lambda *a, **k: plan)
        options, _ = self.prepare()
        result = self.batching.compile_bundle(self.backend, self.src, options, self.cv_compiler())
        self.assertEqual(result.request.factor, 1)
        self.assertEqual(result.block_auto_selection["reason"], "AXIS_ROLE_UNKNOWN")
        self.assertEqual(len(self.compiles), 1)
        self.assertFalse(self.model_calls)

    def test_same_key_recursion_fails_without_deadlock(self):
        options, _ = self.prepare()
        original = self.dispatch._compile_auto_bundle

        def recursive(*args):
            return self.dispatch.compile_auto_bundle(*args)

        self.dispatch._compile_auto_bundle = recursive
        with self.assertRaisesRegex(RuntimeError, "recursive automatic"):
            self.dispatch.compile_auto_bundle(self.backend, self.src, options, self.compile)
        self.dispatch._compile_auto_bundle = original
        self.assertEqual(self.bundle().request.factor, 2)

    def test_concurrent_same_key_coalesces_only_inflight_work(self):
        options, _ = self.prepare()
        original = self.dispatch._compile_auto_bundle
        barrier, entered, finish = threading.Barrier(4), threading.Event(), threading.Event()
        count = []

        def slow(*args):
            count.append(1)
            entered.set()
            if not finish.wait(2):
                raise RuntimeError("test synchronization timeout")
            return original(*args)

        self.dispatch._compile_auto_bundle = slow

        def work():
            barrier.wait()
            return self.dispatch.compile_auto_bundle(self.backend, self.src, options, self.compile)

        with ThreadPoolExecutor(max_workers=4) as pool:
            futures = [pool.submit(work) for _ in range(4)]
            self.assertTrue(entered.wait(2))
            time.sleep(0.03)
            finish.set()
            results = [future.result(timeout=2) for future in futures]
        self.assertEqual(len(count), 1)
        self.assertTrue(all(result is results[0] for result in results))
        self.assertFalse(self.dispatch._pending)


if __name__ == "__main__":
    unittest.main()
