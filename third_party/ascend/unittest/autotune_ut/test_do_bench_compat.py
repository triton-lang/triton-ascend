# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

import sys
from types import MethodType, SimpleNamespace

import pytest
import triton
import triton.backends.ascend.testing as ascend_testing
import triton.backends.ascend.runtime.autotuner as ascend_autotuner
from triton.runtime.autotuner import Config
from triton.backends.ascend.runtime.autotuner import AutoTilingTuner


def _make_tuner(do_bench):
    tuner = object.__new__(AutoTilingTuner)
    tuner.compile_parallel = False
    tuner.do_bench = do_bench
    tuner.user_defined_do_bench = True
    tuner.print_autotuning = False

    def _make_kernel_call(self, *args, config, **meta):

        def kernel_call(warmup):
            return None

        return kernel_call

    tuner._make_kernel_call = MethodType(_make_kernel_call, tuner)
    return tuner


def _make_run_tuner(configs):
    key = ("disk-cache-key", )
    tuner = object.__new__(AutoTilingTuner)
    tuner.arg_names = []
    tuner.configs = configs
    tuner.cache = {}
    tuner.is_simt_mode = False
    tuner.simt_stack_limit = 8192
    tuner.generate_key_and_configs = lambda *args, **kwargs: key
    tuner.prune_configs = lambda kwargs: configs
    tuner.enable_ubtuner = False
    tuner.cache_results = True
    tuner.print_autotuning = False
    tuner.auto_profile_dir = None
    tuner.nargs = {}
    tuner.pre_hook = lambda kwargs, reset_only=False: None
    tuner.run_kwargs = []
    tuner.fn = SimpleNamespace(run=lambda *args, **kwargs: tuner.run_kwargs.append(kwargs) or "kernel-result")
    return tuner, key


def test_batch_bench_supports_do_bench_with_quantiles():
    record = {}

    def _do_bench(fn, quantiles):
        record["quantiles"] = quantiles
        fn()
        return (1.0, 1.0, 1.0)

    tuner = _make_tuner(_do_bench)
    cfg = Config({})

    result = tuner._batch_bench(configs=[cfg])

    assert result[cfg] == (1.0, 1.0, 1.0)
    assert record["quantiles"] == (0.5, 0.2, 0.8)


def test_batch_bench_requires_do_bench_quantiles_parameter():

    def _do_bench(fn):
        fn()
        return (2.0, 2.0, 2.0)

    tuner = _make_tuner(_do_bench)
    cfg = Config({})

    with pytest.raises(TypeError):
        tuner._batch_bench(configs=[cfg])


def test_batch_bench_npu_env_respects_user_do_bench(monkeypatch):
    calls = {"do_bench": 0}

    def _do_bench(fn, quantiles):
        calls["do_bench"] += 1
        fn()
        return (3.0, 3.0, 3.0)

    def _unexpected_do_bench_npu(*args, **kwargs):
        raise AssertionError("do_bench_npu should not be used when user do_bench is provided")

    tuner = _make_tuner(_do_bench)
    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setattr("triton.backends.ascend.testing.do_bench_npu", _unexpected_do_bench_npu)

    result = tuner._batch_bench(configs=[cfg0, cfg1])

    assert calls["do_bench"] == 2
    assert result[cfg0] == (3.0, 3.0, 3.0)
    assert result[cfg1] == (3.0, 3.0, 3.0)


def test_batch_bench_npu_env_uses_do_bench_npu_without_user_do_bench(monkeypatch):

    def _do_bench(fn, quantiles):
        raise AssertionError("self.do_bench should not be used when no user do_bench is provided")

    calls = {"do_bench_npu": 0}

    def _do_bench_npu(funcs, clear_l2_cache=False, warmup=5, active=30, target_kernel_name=None, **kwargs):
        calls["do_bench_npu"] += 1
        assert len(funcs) == 2
        return [1.0, 2.0]

    tuner = _make_tuner(_do_bench)
    tuner.user_defined_do_bench = False
    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setattr("triton.backends.ascend.testing.do_bench_npu", _do_bench_npu)

    result = tuner._batch_bench(configs=[cfg0, cfg1])

    assert calls["do_bench_npu"] == 1
    assert result[cfg0] == 1.0
    assert result[cfg1] == 2.0


def test_batch_bench_npu_diagnostics_include_config_labels(monkeypatch, capsys):

    def _dummy_kernel():
        return None

    def _do_bench(fn, quantiles):
        raise AssertionError("self.do_bench should not be used when NPU profiling is enabled")

    cfg0 = Config({"ID": 0})
    cfg1 = Config({"ID": 1})

    def _do_bench_npu(
        funcs,
        clear_l2_cache=False,
        warmup=5,
        active=30,
        target_kernel_name=None,
        diagnostic_callback=None,
        diagnostic_labels=None,
        **kwargs,
    ):
        assert len(funcs) == 2
        assert diagnostic_labels == [str(cfg0), str(cfg1)]
        assert diagnostic_callback is not None
        diagnostic_callback("benchmark_config", {
            "config_index": 0,
            "profile_wall_ms": "12.000",
            "config": diagnostic_labels[0],
        })
        return [1.0, 2.0]

    tuner = _make_tuner(_do_bench)
    tuner.user_defined_do_bench = False
    tuner.print_autotuning = True
    tuner.base_fn = _dummy_kernel
    monkeypatch.setenv("TRITON_BENCH_METHOD", "npu")
    monkeypatch.setattr("triton.backends.ascend.testing.do_bench_npu", _do_bench_npu)

    result = tuner._batch_bench(configs=[cfg0, cfg1])

    assert result == {cfg0: 1.0, cfg1: 2.0}
    assert ("stage=benchmark_config, status=summary, config_index=0, profile_wall_ms=12.000" in capsys.readouterr().out)


def test_do_bench_npu_reports_per_config_wall_and_device_timings(monkeypatch):
    clock = {"value": 0.0}

    def advance(seconds):
        clock["value"] += seconds

    def perf_counter():
        return clock["value"]

    def synchronize():
        advance(0.3)

    def func0():
        advance(0.1)

    def func1():
        advance(0.2)

    class FakeProfile:

        def __enter__(self):
            advance(0.01)
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            advance(0.04)

    def profile(**kwargs):
        advance(0.02)
        return FakeProfile()

    fake_profiler = SimpleNamespace(
        _ExperimentalConfig=lambda **kwargs: object(),
        AiCMetrics=SimpleNamespace(PipeUtilization=object()),
        ProfilerLevel=SimpleNamespace(Level1=object()),
        ProfilerActivity=SimpleNamespace(NPU=object()),
        tensorboard_trace_handler=lambda path: object(),
        profile=profile,
    )
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(npu=SimpleNamespace(synchronize=synchronize)))
    monkeypatch.setitem(sys.modules, "torch_npu", SimpleNamespace(profiler=fake_profiler))
    monkeypatch.setattr(ascend_testing.time, "perf_counter", perf_counter)

    def collect_prof_result(*args, return_diagnostics=False, **kwargs):
        assert return_diagnostics is True
        advance(0.05)
        return [250.0, 350.0], [500.0, 700.0]

    def remove_profile(keep_res, torch_path):
        advance(0.03)

    monkeypatch.setattr(ascend_testing, "_collect_prof_result", collect_prof_result)
    monkeypatch.setattr(ascend_testing, "_rm_dic", remove_profile)
    diagnostics = []

    result = ascend_testing.do_bench_npu(
        [func0, func1],
        warmup=1,
        active=1,
        prof_dir="/tmp/fake-npu-profile",
        target_kernel_name="kernel",
        diagnostic_callback=lambda stage, details: diagnostics.append((stage, details)),
        diagnostic_labels=["config-0", "config-1"],
    )

    assert result == [250.0, 350.0]
    assert diagnostics[0] == ("benchmark_config", {
        "config_index": 0,
        "profile_calls": 2,
        "prewarm_enqueue_ms": "100.000",
        "prewarm_synchronize_ms": "300.000",
        "profile_enqueue_ms": "200.000",
        "profile_synchronize_ms": "600.000",
        "profile_l2_clear_ms": "0.000",
        "profile_wall_ms": "800.000",
        "target_device_total_ms": "500.000",
        "target_device_active_mean_ms": "250.000",
        "unaccounted_profile_ms": "300.000",
        "config": "config-0",
    })
    assert diagnostics[1][0] == "benchmark_config"
    assert diagnostics[1][1]["profile_wall_ms"] == "1000.000"
    assert diagnostics[1][1]["unaccounted_profile_ms"] == "300.000"
    assert diagnostics[2] == ("benchmark_profiler", {
        "result": "success",
        "configs": 2,
        "prewarm_calls": 2,
        "profile_calls": 4,
        "profile_create_ms": "20.000",
        "profile_enter_ms": "10.000",
        "profile_loop_ms": "1800.000",
        "profile_finalize_ms": "40.000",
        "collect_results_ms": "50.000",
        "cleanup_ms": "30.000",
        "other_ms": "0.000",
        "total_ms": "2850.000",
        "target_kernel_name": "kernel",
        "profile_results_retained": False,
    })


def test_collect_prof_result_can_return_profiled_device_totals(tmp_path):
    kernel_details = tmp_path / "kernel_details.csv"
    kernel_details.write_text(
        "Name,Type,Duration(us)\n"
        "kernel,AI_CORE,1000\n"
        "kernel,AI_CORE,2000\n"
        "kernel,AI_CORE,3000\n"
        "kernel,AI_CORE,4000\n",
        encoding="utf-8",
    )
    funcs = [lambda: None, lambda: None]

    time_cost, profiled_device_total = ascend_testing._collect_prof_result(
        str(tmp_path),
        funcs,
        num_warmup=1,
        num_active=1,
        target_kernel_name="kernel",
        return_diagnostics=True,
    )

    assert time_cost == [2.0, 4.0]
    assert profiled_device_total == [3.0, 7.0]


def test_autotilingtuner_marks_user_defined_do_bench():
    marker = {"called": False}

    def _do_bench(fn, quantiles):
        marker["called"] = True
        return (0.0, 0.0, 0.0)

    def _dummy_kernel():
        return None

    _dummy_kernel.arg_names = []

    tuner = AutoTilingTuner(
        _dummy_kernel,
        [],
        [Config({})],
        [],
        None,
        None,
        do_bench=_do_bench,
    )

    assert tuner.user_defined_do_bench is True
    assert marker["called"] is False


def test_autotune_stage_timing_prints_start_and_elapsed_time(monkeypatch, capsys):

    def _dummy_kernel():
        return None

    tuner = object.__new__(AutoTilingTuner)
    tuner.print_autotuning = True
    tuner.base_fn = _dummy_kernel
    perf_counter_values = iter((10.0, 10.125))
    monkeypatch.setattr(ascend_autotuner.time, "perf_counter", lambda: next(perf_counter_values))

    start_time = tuner._autotune_stage_start("compile_configs", candidate_configs=2)
    tuner._autotune_stage_end("compile_configs", start_time, valid_configs=1)

    assert capsys.readouterr().out.splitlines() == [
        "Triton autotuning stage: function=_dummy_kernel, stage=compile_configs, status=start, "
        "candidate_configs=2",
        "Triton autotuning stage: function=_dummy_kernel, stage=compile_configs, status=end, "
        "elapsed_ms=125.000, valid_configs=1",
    ]


def test_autotune_stage_timing_is_disabled_without_print_flag(monkeypatch, capsys):
    tuner = object.__new__(AutoTilingTuner)
    tuner.print_autotuning = False
    monkeypatch.setattr(
        ascend_autotuner.time,
        "perf_counter",
        lambda: pytest.fail("perf_counter must not run when stage timing is disabled"),
    )

    start_time = tuner._autotune_stage_start("compile_configs")
    tuner._autotune_stage_end("compile_configs", start_time)

    assert start_time is None
    assert capsys.readouterr().out == ""


def test_batch_bench_prints_compile_and_benchmark_stages(monkeypatch, capsys):

    def _dummy_kernel():
        return None

    tuner = _make_tuner(lambda fn, quantiles: fn() or (1.0, 1.0, 1.0))
    tuner.print_autotuning = True
    tuner.base_fn = _dummy_kernel
    configs = [Config({"ID": 0}), Config({"ID": 1})]
    monkeypatch.delenv("TRITON_BENCH_METHOD", raising=False)

    timings = tuner._batch_bench(configs=configs)

    assert list(timings) == configs
    output = capsys.readouterr().out
    assert "stage=prepare_kernel_calls, status=end" in output
    assert "stage=compile_configs, status=end" in output
    assert "valid_configs=2, failed_configs=0, parallel=False, workers=1" in output
    assert "stage=benchmark_configs, status=end" in output
    assert "benchmark_method=default, benchmark_configs=2" in output


def test_run_prints_prune_selection_launch_and_gc_stages(monkeypatch, capsys):

    def _dummy_kernel():
        return None

    selected = Config({"BLOCK_SIZE": 16})
    other = Config({"BLOCK_SIZE": 32})
    tuner, _ = _make_run_tuner([selected, other])
    tuner.cache_results = False
    tuner.print_autotuning = True
    tuner.base_fn = _dummy_kernel
    tuner._batch_bench = lambda *args, configs, **kwargs: {
        selected: 1.0,
        other: 2.0,
    }
    monkeypatch.setattr(ascend_autotuner.gc, "collect", lambda: None)

    assert tuner.run() == "kernel-result"

    output = capsys.readouterr().out
    assert "stage=prune_configs, status=end" in output
    assert "candidate_configs=2, pruned_configs=2" in output
    assert "stage=select_best_config, status=end" in output
    assert "stage=launch_best_config, status=start, memory_cache=miss, disk_cache=disabled" in output
    assert "stage=garbage_collect, status=end" in output


def test_ascend_autotune_decorator_forwards_do_bench(monkeypatch):
    import triton.backends.ascend.runtime.autotuner as ascend_autotuner

    captured = {}

    class DummyAutoTilingTuner:

        def __init__(self, *args, **kwargs):
            captured["do_bench"] = kwargs.get("do_bench")

    monkeypatch.setattr(ascend_autotuner, "AutoTilingTuner", DummyAutoTilingTuner)

    def _dummy_kernel():
        return None

    _dummy_kernel.arg_names = []
    my_do_bench = lambda kernel_call, quantiles: (0.0, 0.0, 0.0)

    ascend_autotuner.autotune(configs=[object()], key=[], do_bench=my_do_bench)(_dummy_kernel)

    assert captured["do_bench"] is my_do_bench


def test_run_skips_gc_on_autotune_disk_cache_hit(monkeypatch, capsys):

    def _dummy_kernel():
        return None

    configs = [Config({"BLOCK_SIZE": 16}), Config({"BLOCK_SIZE": 32})]
    tuner, key = _make_run_tuner(configs)
    tuner.print_autotuning = True
    tuner.base_fn = _dummy_kernel
    gc_calls = []
    profile_calls = []

    def check_disk_cache(tuning_key, pruned_configs, benchmark):
        assert tuning_key == key
        tuner.cache[tuning_key] = pruned_configs[0]
        return True

    def unexpected_batch_bench(*args, **kwargs):
        raise AssertionError("benchmark must not run on a disk-cache hit")

    tuner.check_disk_cache = check_disk_cache
    tuner._batch_bench = unexpected_batch_bench
    tuner.auto_profile_dir = "profile-output"
    tuner._profile = lambda *args, config, **kwargs: profile_calls.append(config)
    monkeypatch.setattr(ascend_autotuner.gc, "collect", lambda: gc_calls.append(True))

    assert tuner.run() == "kernel-result"
    assert gc_calls == []
    assert profile_calls == []
    output = capsys.readouterr().out
    assert "stage=disk_cache_check_and_update, status=end" in output
    assert "disk_cache=hit, benchmark_included=False" in output
    assert "stage=launch_best_config, status=start, memory_cache=miss, disk_cache=hit" in output


def test_run_keeps_gc_on_autotune_disk_cache_miss(monkeypatch):
    configs = [Config({"BLOCK_SIZE": 16}), Config({"BLOCK_SIZE": 32})]
    tuner, key = _make_run_tuner(configs)
    gc_calls = []
    benchmark_calls = []
    profile_calls = []

    def check_disk_cache(tuning_key, pruned_configs, benchmark):
        assert tuning_key == key
        benchmark()
        return False

    def batch_bench(*args, configs, **kwargs):
        benchmark_calls.append(configs)
        return {configs[0]: 1.0, configs[1]: 2.0}

    tuner.check_disk_cache = check_disk_cache
    tuner._batch_bench = batch_bench
    tuner.auto_profile_dir = "profile-output"
    tuner._profile = lambda *args, config, **kwargs: profile_calls.append(config)
    monkeypatch.setattr(ascend_autotuner.gc, "collect", lambda: gc_calls.append(True))

    assert tuner.run() == "kernel-result"
    assert benchmark_calls == [configs]
    assert gc_calls == [True]
    assert profile_calls == [configs[0]]


def test_run_keeps_gc_on_single_config_cache_miss(monkeypatch):
    tuner, _ = _make_run_tuner([Config({"BLOCK_SIZE": 16, "compile_mode": "simt_only"})])
    gc_calls = []
    profile_calls = []

    def unexpected_disk_cache(*args, **kwargs):
        raise AssertionError("single-config path must not probe disk cache")

    tuner.check_disk_cache = unexpected_disk_cache
    tuner.auto_profile_dir = "profile-output"
    tuner._profile = lambda *args, config, **kwargs: profile_calls.append(config)
    monkeypatch.setattr(ascend_autotuner.gc, "collect", lambda: gc_calls.append(True))

    assert tuner.run() == "kernel-result"
    assert tuner.run_kwargs[-1]["simt_stack_limit"] == 8192
    assert gc_calls == [True]
    assert profile_calls == []
