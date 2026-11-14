# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""CPU-only checks; no Triton driver or device is initialized."""
import importlib
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import patch

BACKEND = Path(__file__).resolve().parents[2] / "backend"
PACKAGE = "_block_auto_entry_test"


def module(name, **values):
    result = ModuleType(name)
    result.__dict__.update(values)
    return result


class TestHostEntry(unittest.TestCase):

    def setUp(self):
        self.events = []
        self.fn, self.kernel = object(), object()
        self.args, self.kwargs, self.grid = (object(), ), {"BLOCK": 128}, (17, )
        self.backend = SimpleNamespace()
        self.values = (self.args, self.kwargs, self.grid, False, 0, 42)
        package = module(PACKAGE, __path__=[str(BACKEND)])
        config = module(PACKAGE + ".reuse.config", enabled=lambda: False)
        self.modules = {
            PACKAGE: package,
            PACKAGE + ".reuse": module(PACKAGE + ".reuse", __path__=[]),
            PACKAGE + ".reuse.config": config,
        }
        self.patcher = patch.dict(sys.modules, self.modules)
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.dispatch = importlib.import_module(PACKAGE + ".block_auto_dispatch")
        self.handler = self.dispatch.get_launch_handler()

    def tail(self, *args, **kwargs):
        self.events.append((args, kwargs))
        return self.kernel

    def run_handler(self, values=None):
        return self.handler.run(self.fn, *(values or self.values), self.backend, self.tail)

    def test_absent_provider_preserves_original_arguments_and_result(self):
        self.assertIs(self.run_handler(), self.kernel)
        self.assertEqual(len(self.events), 1)
        args, kwargs = self.events[0]
        self.assertFalse(kwargs)
        for actual, expected in zip(args, self.values):
            self.assertIs(actual, expected)
        self.assertNotIn(PACKAGE + ".block_auto_model", sys.modules)

    def test_absent_provider_does_not_bind_or_evaluate_grid(self):

        def grid(_):
            raise AssertionError("must remain owned by the normal JIT")

        self.values = (self.args, self.kwargs, grid, False, 0, 42)
        self.run_handler()
        self.assertIs(self.events[0][0][2], grid)

    def test_existing_reuse_path_is_preserved(self):
        self.modules[PACKAGE + ".reuse.config"].enabled = lambda: True
        calls = []

        def reuse(fn, *arguments):
            calls.append((fn, arguments))
            return arguments[-1](*arguments[:-2])

        bridge = module(PACKAGE + ".reuse.bridge", get_handler=lambda: SimpleNamespace(run=reuse))
        with patch.dict(sys.modules, {PACKAGE + ".reuse.bridge": bridge}):
            self.assertIs(self.run_handler(), self.kernel)
        self.assertIs(calls[0][0], self.fn)
        self.assertIs(calls[0][1][-2], self.backend)
        self.assertEqual(len(self.events), 1)

    def test_warmup_does_not_invoke_candidate_provider(self):

        def fail(*args, **kwargs):
            raise AssertionError("warmup cannot create merged candidates")

        self.backend.run_block_auto = fail
        self.assertIs(self.run_handler((*self.values[:3], True, *self.values[4:])), self.kernel)

    def test_provider_receives_cpu_selector_and_single_continuation(self):
        select = object()
        calls = []

        def provider(fn, *arguments, select_candidates, continuation):
            calls.append((fn, select_candidates))
            return continuation(*arguments)

        self.backend.run_block_auto = provider
        with patch.dict(sys.modules, {PACKAGE + ".block_auto_model": module("model", select_candidates=select)}):
            self.assertIs(self.run_handler(), self.kernel)
        self.assertEqual(calls, [(self.fn, select)])
        self.assertEqual(len(self.events), 1)

    def test_callback_failure_is_not_replayed(self):

        def fail(*args, **kwargs):
            raise RuntimeError("launch callback failed")

        with self.assertRaisesRegex(RuntimeError, "launch callback failed"):
            self.handler.run(self.fn, *self.values, self.backend, fail)
        self.assertFalse(self.events)


class TestModelContracts(unittest.TestCase):

    def setUp(self):
        self.patcher = patch.dict(sys.modules, {PACKAGE: module(PACKAGE, __path__=[str(BACKEND)])})
        self.patcher.start()
        self.addCleanup(self.patcher.stop)
        self.cv = importlib.import_module(PACKAGE + ".block_auto_cv")
        self.native = importlib.import_module(PACKAGE + ".block_auto_models.native_model")

    def probe(self, axis, factor):
        return dict(compile_status="REWRITTEN", selected_axis=axis, requested_factor=factor,
                    cv_mode="spatial_concat_v1", cv_lowering="spatial_m_v1" if axis == 0 else "spatial_n_v1")

    def test_cv_shortlist_contains_two_proved_axes(self):
        result = self.cv.candidate_plan({"grid": [8, 8, 1]}, probe=self.probe)
        self.assertEqual(result["reason"], "CV_M4_N4")
        self.assertEqual([(r["axis"], r["factor"]) for r in result["candidates"]], [(0, 4), (1, 4)])

    def test_cv_small_grid_never_invents_a_merge(self):
        result = self.cv.candidate_plan({"grid": [1, 1, 1]}, probe=self.probe)
        self.assertEqual(result["candidates"], [])

    def test_cv_rejected_frontend_roles_are_excluded(self):

        def reject(axis, factor):
            raise ValueError("not transformable")

        result = self.cv.candidate_plan({"grid": [8, 8, 1]}, probe=reject)
        self.assertEqual(result["candidates"], [])

    def test_model_passes_candidate_llvm_without_library_inputs(self):
        binding = SimpleNamespace(analyze_block_candidate=lambda request: request)
        modules = {"triton._C.libtriton": module("libtriton", ascend=binding)}
        metadata = dict(target={"arch": "Ascend950PR_9579"}, parallel_mode="simd", name="kernel",
                        auto_blockify_enabled=True)
        with patch.dict(sys.modules, modules), patch.object(self.native, "launch_configuration",
                                                            return_value=([], [64, 1, 1], 56, 56)):
            llvm = {"aiv": "candidate LLVM containing linked helper bodies"}
            config = self.native.analyze(llvm, "", "", metadata, {}, 0, 1, raw="raw LLVM")
        self.assertEqual(config["modules"], llvm)
        self.assertEqual(
            set(config),
            {"mode", "modules", "name", "arguments", "grid", "cores", "blocks", "auto_blockify", "raw_llvm"})

    def test_mix_keeps_mode_checks_without_compiler_records(self):
        ir = 'func.func @kernel() attributes {hacc.entry, hivm.vf_mode<MIX>, hivm.func_core_type<AIV>}'
        self.assertEqual(self.native.mixed_parameters(ir),
                         dict(simt_transaction_bytes=64, simt_transition_cycles=32, simt_instruction_cycles=1))
        with self.assertRaisesRegex(ValueError, "one AIV SIMD/SIMT entry"):
            self.native.mixed_parameters(ir.replace("<MIX>", "<SIMD>"))

    def test_cv_ranks_without_compiler_records(self):
        contract = dict(grid=[64, 1, 1], arguments={"x": dict(kind="tensor", dtype="fp32")})
        items = [
            dict(key="unmerged", factor=1, axis=0, metadata={}),
            dict(key="a0_f2", factor=2, axis=0, metadata={"block_batching": {"cv_mode": "independent_batch_v1"}})
        ]
        traces = []
        for item, score in zip(items, (100, 60)):
            item.update(resource_checked=True, launch_contract=contract, ttir="", final_ir="")
            traces.append(
                dict(score=score, launched_groups=28, cube_core_count=28, auto_blockify=True,
                     logical_grid=[64 // item["factor"], 1, 1], mode="cv", counts=dict(mma_calls=1, useful_flops=1024)))
        with patch.object(self.cv, "analyze_candidate", side_effect=traces):
            result = self.cv.rank_cv(items, contract)
        self.assertTrue(result["model_complete"])
        self.assertEqual(result["selected_key"], "a0_f2")

    def test_mix_ub_default_still_must_match_runtime_capacity(self):
        gate = importlib.import_module(PACKAGE + ".block_auto_models.cpu_resource_gate")
        ir = '''func.func @kernel() attributes {hacc.entry, hivm.vf_mode<MIX>, hivm.func_core_type<AIV>}
"UB_ALIGN_SIZE", 256 : i32
"MAXIMUM_D_CACHE_SIZE", 524288 : i32
"MINIMAL_D_CACHE_SIZE", 262144 : i32'''
        metadata = dict(mix_mode="aiv", parallel_mode="mix_simd_simt", shared_mem_dynamic_size=216 * 1024)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "compiler-help.txt"
            path.write_text("--simt-vf-dynamic-size=<int> Default is 216.\n")
            files = {"backend_defaults": {"path": str(path)}}
            usage = dict(capacity_bits=248 * 1024 * 8, address_extent_bits=256)
            with patch.object(gate, "inspect_final_ub", side_effect=lambda *a, **k: dict(usage)):
                self.assertEqual(
                    gate.inspect_vv_ub(ir, "Ascend950", metadata, [], files)["capacity_bits"], 216 * 1024 * 8)
                metadata["shared_mem_dynamic_size"] = 208 * 1024
                with self.assertRaisesRegex(ValueError, "capacities differ"):
                    gate.inspect_vv_ub(ir, "Ascend950", metadata, [], files)

    def test_missing_candidates_keep_baseline_without_loading_a_driver(self):
        model = importlib.import_module(PACKAGE + ".block_auto_model")
        factor, report = model.select_candidates([], {})
        self.assertEqual(factor, 1)
        self.assertFalse(report["model_complete"])

    def test_unmerged_grid_does_not_need_batching_package(self):
        self.assertFalse(self.native.preserves_grid({}, 1))

    def test_iteration_schedule_requires_matching_loop_proof(self):
        loop = dict(id=0, bits=32, lower=0, upper=16, step=1)
        schedule = dict(version=1, kind="sequential_iterations_v1", grid_policy="preserve", factor=2, loop=loop)
        metadata = dict(
            block_batching=dict(vv_lowering="sequential_iterations_v1", lane_width=2, cv_mode="disabled",
                                guard=dict(loop_schedule=schedule, accesses=[dict(loops=[loop])])),
            program_grid_mapping_applied=False, row_coalescing_applied=False, ptsm_cap_authorized=False)
        self.assertTrue(self.native.preserves_grid(metadata, 2))
        metadata["block_batching"]["guard"]["accesses"] = []
        with self.assertRaisesRegex(ValueError, "VV_LOOP_SCHEDULE_MISMATCH"):
            self.native.preserves_grid(metadata, 2)


class LocalCache:

    def __init__(self, directory):
        self.cache_dir = str(directory)
        directory.mkdir(parents=True, exist_ok=True)

    def put(self, data, name, binary=False):
        path = Path(self.cache_dir) / name
        path.write_bytes(data if isinstance(data, bytes) else data.encode())
        return str(path)

    def get_file(self, name):
        path = Path(self.cache_dir) / name
        return str(path) if path.is_file() else None


class TestModelCache(unittest.TestCase):

    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.compiler = self.root / "bisheng"
        self.compiler.touch()
        self.compiler.with_name("llvm-link").touch()
        self.keys = []
        self.cache_module = module("cache", get_cache_manager=self.cache)
        modules = {
            PACKAGE: module(PACKAGE, __path__=[str(BACKEND)]), PACKAGE + ".utils":
            module("utils", _get_bisheng_path=lambda: str(self.compiler)), "triton.runtime.cache": self.cache_module
        }
        patcher = patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        self.artifacts = importlib.import_module(PACKAGE + ".block_auto_artifacts")
        self.optimizer = importlib.import_module(PACKAGE + ".block_auto_models.vv_compilation")

    def cache(self, key):
        self.keys.append(key)
        return LocalCache(self.root / "cache" / key)

    def collect(self, key, text):
        source = self.root / "temps"
        source.mkdir(exist_ok=True)
        (source / "module.hivm.opt.mlir").write_text("module {}")
        (source / "kernel.ll").write_text(text)
        metadata = {"hash": key}
        evidence = self.artifacts.collect(source, b"\x7fELF", self.compiler, metadata, [], cache_factory=self.cache)
        self.assertEqual(evidence["status"], "COLLECTED")
        metadata["block_auto_evidence"] = evidence
        return SimpleNamespace(metadata=SimpleNamespace(_asdict=lambda: metadata),
                               asm=dict(ttir="ttir", ttadapter="ir"))

    def test_artifacts_use_candidate_cache_and_need_no_file_records(self):
        first = self.collect("first", "first module")
        second = self.collect("second", "second module")
        self.compiler.write_text("new compiler installation")
        for kernel, expected in ((first, "first module"), (second, "second module")):
            result = self.artifacts.load_evidence(kernel, cache_factory=self.cache)
            self.assertEqual(result["llvm"], expected)
            self.assertEqual(result["files"]["compiler"], dict(path=str(self.compiler)))
        self.assertEqual(self.keys, ["first", "second", "first", "second"])

    def test_missing_cached_input_remains_unavailable(self):
        kernel = self.collect("missing", "module")
        (self.root / "cache/missing/block_auto_llvm.ll").unlink()
        with self.assertRaisesRegex(ValueError, "cached model input unavailable"):
            self.artifacts.load_evidence(kernel, cache_factory=self.cache)

    def test_optimizer_publishes_once_under_the_normal_candidate_key(self):
        item = dict(metadata={"hash": "candidate"},
                    files={"compiler": {"path": str(self.compiler)}, "llvm":
                           {"path": "candidate.ll"}}, final_ir='hivm.aiv_bitcode = #hivm.aiv_bitcode<"meta.bc">')

        def compile_cpu(command, **kwargs):
            output = Path(command[command.index("-o") + 1])
            output.write_text("optimized module")
            self.assertEqual(kwargs["env"]["ASCEND_RT_VISIBLE_DEVICES"], "")

        with patch.object(self.optimizer.subprocess, "run", side_effect=compile_cpu) as run:
            self.assertEqual(self.optimizer.optimized_llvm(item), "optimized module")
            self.assertEqual(self.optimizer.optimized_llvm(item), "optimized module")
        self.assertEqual(run.call_count, 2)
        self.assertIn("-emit-llvm", run.call_args_list[0].args[0])
        self.assertEqual(self.keys, ["candidate", "candidate"])

    def test_optimizer_failure_does_not_publish_a_cache_entry(self):
        item = dict(metadata={"hash": "failed"},
                    files={"compiler": {"path": str(self.compiler)}, "llvm":
                           {"path": "candidate.ll"}}, final_ir='hivm.aiv_bitcode = #hivm.aiv_bitcode<"meta.bc">')
        with patch.object(self.optimizer.subprocess, "run", side_effect=OSError("optimizer failed")):
            with self.assertRaisesRegex(OSError, "optimizer failed"):
                self.optimizer.optimized_llvm(item)
        self.assertFalse((self.root / "cache/failed/block_auto_optimized.ll").exists())


if __name__ == "__main__":
    unittest.main()
