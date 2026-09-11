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
"""Host-side contracts for launch plans; run in the remote NPU test container."""
from dataclasses import FrozenInstanceError, replace
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import shutil
import subprocess
import threading
import time
from types import SimpleNamespace

import pytest

from triton.backends.ascend import launcher, utils


@pytest.fixture(autouse=True)
def cache_configuration_scope():
    from triton import knobs
    # Attribute overrides and environment changes both belong to this test.
    with knobs.cache.scope():
        yield


def metadata(**changes):
    fields = dict(target=SimpleNamespace(arch="Ascend910B3"), workspace_size=0,
                  lock_init_value=0, sync_block_lock_layout=0, bs_task_type=0,
                  mix_mode="aiv", shared=0, compile_on_910_95=False,
                  parallel_mode="", is_pure_simt=False, debug=False,
                  coalesce_factor=1, coalesce_axis=-1, coalesce_grid_ceil_div=False,
                  has_auto_blockify_blacklist_op=False,
                  program_grid_mapping_applied=False, program_grid_transforms=None,
                  row_coalescing_applied=False, auto_blockify_enabled=False,
                  ptsm_cap_authorized=False)
    fields.update(changes)
    return SimpleNamespace(**fields)


@pytest.fixture
def policy(monkeypatch):
    monkeypatch.setattr(utils, "is_ffts_supported", lambda arch: arch.startswith("Ascend910B"))
    monkeypatch.setattr(utils, "force_disable_ffts", lambda arch: False)
    monkeypatch.setattr(utils, "_is_auto_map_parallel_blocks_enabled", lambda: True)
    for name in ("TRITON_DEVICE_PRINT", "TRITON_ENABLE_TASKQUEUE", "TRITON_GRID_WARN_PRINT"):
        monkeypatch.delenv(name, raising=False)
    return SimpleNamespace(get_aivector_core_num=lambda: 40, get_aicore_num=lambda: 20)


@pytest.mark.parametrize("ceil_div", [False, True])
def test_launch_plan_preserves_independent_grid_policies(policy, ceil_div):
    spec = launcher.make_launch_spec(metadata(coalesce_factor=16, coalesce_axis=1, row_coalescing_applied=True,
                                    coalesce_grid_ceil_div=ceil_div,
                                    has_auto_blockify_blacklist_op=True), policy)
    assert (spec.coalesce_axis, spec.coalesce_factor) == (1, 16)
    assert bool(spec.flags & launcher.COALESCE_CEIL) == ceil_div
    assert not spec.flags & launcher.AUTO_MAP
    assert spec.flags & launcher.FFTS
    assert spec.flags & launcher.TASKQUEUE
    assert spec.physical_blocks == 40
    with pytest.raises(FrozenInstanceError):
        spec.coalesce_factor = 2


@pytest.mark.parametrize("taskqueue", [False, True])
@pytest.mark.parametrize("pure", [False, True])
def test_native_simt_configuration(policy, monkeypatch, taskqueue, pure):
    monkeypatch.setenv("TRITON_ENABLE_TASKQUEUE", str(taskqueue))
    spec = launcher.make_launch_spec(
        metadata(target=SimpleNamespace(arch="Ascend950PR"), compile_on_910_95=True, parallel_mode="mix_simd_simt",
                 is_pure_simt=pure, shared_mem_dynamic_size=221184), policy)
    assert bool(spec.flags & launcher.TASKQUEUE) == taskqueue
    assert bool(spec.flags & launcher.PURE_SIMT) == pure
    assert spec.flags & launcher.DYNAMIC_SHARED
    assert not spec.flags & launcher.FFTS
    assert spec.shared_mem_dynamic_size == 221184


def test_unordered_lock_participants_and_workspace(policy):
    spec = launcher.make_launch_spec(
        metadata(sync_block_lock_layout=(3 << 32) | 2, workspace_size=2048, mix_mode="mix", bs_task_type=32,
                 auto_tile_and_bind_subblock=True), policy)
    assert (spec.ordered_locks, spec.unordered_locks, spec.participant_factor) == (2, 3, 2)
    assert (spec.workspace_size, spec.physical_blocks, spec.task_type, spec.mix_ratio) == (2048, 20, 3, 2)


def test_empty_and_constexpr_arguments():
    assert launcher.argument_types({}) == []
    assert launcher.argument_types({0: "constexpr"}) == [(launcher.CONSTEXPR, -1)]
    assert launcher.argument_types({0: "i1", 1: "u1", 2: "bf16"}) == [(launcher.I32, -1), (launcher.U32, -1),
                                                                      (launcher.F32, -1)]


def test_nested_descriptor_and_tuple_expansion():
    signature = {0: ("i32", ("tensordesc<fp32[16,32]>", "constexpr")), 1: "*fp32"}
    descriptor = SimpleNamespace(base=1234, shape=(16, 32), strides=(32, 1), padding="nan")
    recorded = []
    wrapped = launcher.wrap_handle_tensordesc(lambda *args: recorded.extend(args), signature)
    wrapped(*([None] * 9), (7, (descriptor, ("keep", "tuple"))), 5678)
    assert recorded[9:] == [7, 1234, 16, 32, 32, 1, True, 16, 32, 32, 1, ("keep", "tuple"), 5678]
    assert len(launcher.argument_types(signature)) == len(recorded) - 9
    with pytest.raises(TypeError, match="Tuple argument"):
        wrapped(*([None] * 9), (7, ), 5678)


def test_simple_signature_keeps_native_callable():
    native = object()
    assert launcher.wrap_handle_tensordesc(native, {0: "*fp32", 1: "i32"}) is native


def test_export_path_is_lazy_and_cached(policy, monkeypatch):
    from triton.backends.ascend import driver
    calls = []
    native = object()
    runtime = SimpleNamespace(create_launcher=lambda spec, types: native)
    monkeypatch.setattr(
        driver, "NPUUtils",
        lambda: SimpleNamespace(get_aivector_core_num=policy.get_aivector_core_num, get_aicore_num=policy.
                                get_aicore_num, get_so_path=lambda: "/cache/utils/npu_utils.so"))
    monkeypatch.setattr(driver, "get_runtime", lambda path, debug: (runtime, "/cache/runtime/runtime.so"))
    monkeypatch.setattr(driver, "export_launcher", lambda *args: calls.append(args) or "/cache/export.so")
    src = SimpleNamespace(signature={0: "*fp32"}, fn=SimpleNamespace(arg_names=["x"]))
    instance = driver.NPULauncher(src, metadata())
    assert instance.launch is native
    assert calls == []
    assert instance.get_launcher_so_path() == instance.so_launcher_path == "/cache/export.so"
    assert len(calls) == 1


def test_runtime_cache_tracks_headers_build_inputs_and_root(tmp_path, monkeypatch):
    compiled = []
    build_inputs = {"compiler": "toolchain-a"}
    monkeypatch.setattr(utils, "npu_extension_fingerprint", lambda *a, **k: dict(build_inputs))

    def build(name, source, **kwargs):
        path = Path(source).with_suffix(".so")
        path.write_bytes(b"a compiled module")
        compiled.append(source)
        return str(path)

    monkeypatch.setattr(utils, "_build_npu_ext", build)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache-a"))
    sources = {"launcher_runtime.cpp": "#include \"contract.h\"", "contract.h": "v1"}
    first = launcher._build_shared("__triton_launcher_runtime", sources)
    assert launcher._build_shared("__triton_launcher_runtime", sources) == first
    assert len(compiled) == 1
    changed = launcher._build_shared("__triton_launcher_runtime", {**sources, "contract.h": "v2"})
    assert changed != first
    build_inputs["compiler"] = "toolchain-b"
    assert launcher._build_shared("__triton_launcher_runtime", sources) != first
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache-b"))
    relocated = launcher._build_shared("__triton_launcher_runtime", sources)
    assert str(tmp_path / "cache-b") in relocated
    assert len(compiled) == 4


@pytest.fixture
def prepared_backend(tmp_path, monkeypatch):
    """Exercise actual preparation/cache code with a deterministic native builder."""
    from triton.backends.ascend import driver
    source_dir = tmp_path / "backend"
    native = source_dir / "launcher_src"
    native.mkdir(parents=True)
    original = Path(launcher.__file__).with_name("launcher_src")
    for source in original.iterdir():
        if source.suffix in (".h", ".cpp"):
            (native / source.name).write_bytes(source.read_bytes())
    (source_dir / "npu_utils.cpp").write_text("helper source v1")
    monkeypatch.setattr(driver, "__file__", str(source_dir / "driver.py"))
    monkeypatch.setattr(launcher, "__file__", str(source_dir / "launcher.py"))
    monkeypatch.setattr(driver.NPUUtils, "instance", object.__new__(driver.NPUUtils), raising=False)
    monkeypatch.setattr(driver.NPUUtils, "get_device_core", lambda self: (20, 40))
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.delenv("TRITON_DEVICE_PRINT", raising=False)
    fingerprint = {"compiler": "toolchain-a"}
    monkeypatch.setattr(utils, "npu_extension_fingerprint", lambda *a, **kw: dict(fingerprint))
    builds = []

    def build(name, source, **kw):
        builds.append((name, Path(source).read_text()))
        binary = Path(source).with_suffix(".so")
        binary.write_bytes(b"a native artifact")
        return str(binary)

    monkeypatch.setattr(utils, "_build_npu_ext", build)
    monkeypatch.setattr(driver, "_build_npu_ext", build)
    # Each plan owns its own policy, even though module preparation is shared.
    module = SimpleNamespace(create_launcher=lambda spec, types: SimpleNamespace(spec=spec, types=types))
    monkeypatch.setattr(launcher, "_load_runtime", lambda path: module)
    src = SimpleNamespace(signature={0: "*fp32"}, fn=SimpleNamespace(arg_names=["x"]))
    return SimpleNamespace(driver=driver, root=source_dir, builds=builds, fingerprint=fingerprint, src=src)


def test_repeated_preparation_reuses_keys_and_artifacts(prepared_backend, monkeypatch):
    state = prepared_backend
    first = state.driver.NPULauncher(state.src, metadata())
    assert len(state.builds) == 2
    hashes = []
    original_hash = hashlib.sha256
    monkeypatch.setattr(hashlib, "sha256", lambda *a, **kw: (hashes.append(a), original_hash(*a, **kw))[1])
    for _ in range(5):
        second = state.driver.NPULauncher(state.src, metadata(workspace_size=2048))
        assert second._runtime_path == first._runtime_path
        assert second.launch is not first.launch
        assert second.launch.spec["workspace_size"] == 2048
    assert first.launch.spec["workspace_size"] == 0
    assert len(state.builds) == 2
    assert hashes == []


def test_prepared_sources_and_build_options_invalidate(prepared_backend):
    state = prepared_backend
    first = state.driver.NPULauncher(state.src, metadata())
    header = state.root / "launcher_src/launcher_args.h"
    header.write_text(header.read_text() + "\n// changed layout source\n")
    second = state.driver.NPULauncher(state.src, metadata())
    assert second._runtime_path != first._runtime_path
    assert len(state.builds) == 3
    (state.root / "npu_utils.cpp").write_text("helper source v2")
    third = state.driver.NPULauncher(state.src, metadata())
    assert third._runtime_path != second._runtime_path
    assert len(state.builds) == 5  # New helper identity also changes the dependency config.
    state.fingerprint["compiler"] = "toolchain-b"
    fourth = state.driver.NPULauncher(state.src, metadata())
    assert fourth._runtime_path != third._runtime_path
    assert len(state.builds) == 7


@pytest.mark.parametrize("setting", ["TRITON_CACHE_DIR", "TRITON_HOME", "knob"])
def test_preparation_cache_root_switch_and_deleted_artifacts(prepared_backend, monkeypatch, tmp_path, setting):
    from triton import knobs
    state = prepared_backend
    first = state.driver.NPULauncher(state.src, metadata())
    if setting == "TRITON_HOME":
        monkeypatch.delenv("TRITON_CACHE_DIR")
        monkeypatch.setenv("TRITON_HOME", str(tmp_path / "new-home"))
    elif setting == "knob":
        monkeypatch.setattr(knobs.cache, "dir", str(tmp_path / "new-knob"))
    else:
        monkeypatch.setenv(setting, str(tmp_path / "new-cache"))
    second = state.driver.NPULauncher(state.src, metadata())
    assert second._runtime_path != first._runtime_path
    assert Path(second._runtime_path).parent.name == Path(first._runtime_path).parent.name
    assert len(state.builds) == 4
    Path(second._runtime_path).unlink()
    helper = state.driver.NPUUtils().get_so_path()
    Path(helper).unlink()
    third = state.driver.NPULauncher(state.src, metadata())
    assert third._runtime_path == second._runtime_path
    assert Path(third._runtime_path).exists() and Path(helper).exists()
    assert len(state.builds) == 6


def test_debug_dump_and_lazy_export_after_preparation_reuse(prepared_backend, monkeypatch, tmp_path):
    state = prepared_backend
    first = state.driver.NPULauncher(state.src, metadata())
    assert len(state.builds) == 2 and first._so_launcher_path is None
    for index in range(2):
        dump = tmp_path / f"dump-{index}"
        monkeypatch.setenv("TRITON_DUMP_DIR", str(dump))
        instance = state.driver.NPULauncher(state.src, metadata(debug=True))
        assert list(dump.rglob("launcher_runtime.cpp"))
        assert list(dump.rglob("launcher_config.h"))
        assert instance.get_launcher_so_path() == instance.get_launcher_so_path()
        assert list(dump.rglob("launcher_export.cpp"))
    assert len(state.builds) == 3


def test_device_print_configuration_invalidates_runtime(prepared_backend, monkeypatch):
    state = prepared_backend
    first = state.driver.NPULauncher(state.src, metadata())
    compiler = state.root / "bisheng"
    compiler.write_text("test compiler identity")
    monkeypatch.setattr(utils, "_get_bisheng_path", lambda: str(compiler))
    monkeypatch.setattr(utils, "_find_cann_version_file", lambda: None)
    monkeypatch.setattr(launcher, "extract_device_print_code_from_cann", lambda: "// print code")
    monkeypatch.setenv("TRITON_DEVICE_PRINT", "1")
    printed = state.driver.NPULauncher(state.src, metadata())
    assert printed._runtime_path != first._runtime_path
    assert printed.launch.spec["flags"] & launcher.DEVICE_PRINT
    again = state.driver.NPULauncher(state.src, metadata())
    assert again._runtime_path == printed._runtime_path and len(state.builds) == 3
    monkeypatch.delenv("TRITON_DEVICE_PRINT")
    assert state.driver.NPULauncher(state.src, metadata())._runtime_path == first._runtime_path


def test_custom_cache_queries_and_materialization_are_not_bypassed(prepared_backend, monkeypatch):
    from triton.runtime.cache import FileCacheManager, get_cache_manager
    state = prepared_backend
    queries = []

    class CustomCache(FileCacheManager):

        def get_file(self, filename):
            queries.append(filename)
            return super().get_file(filename)

    def manager(key):
        return CustomCache(get_cache_manager(key).key)

    monkeypatch.setattr(state.driver, "get_cache_manager", manager)
    monkeypatch.setattr(launcher, "get_cache_manager", manager)
    state.driver.NPULauncher(state.src, metadata())
    queries.clear()
    for _ in range(4):
        state.driver.NPULauncher(state.src, metadata())
    assert len(queries) == 8
    assert queries.count("npu_utils.so") == 4
    assert len(state.builds) == 2


def test_remote_cache_keeps_access_accounting(prepared_backend, monkeypatch):
    from triton import knobs
    from triton.runtime.cache import RemoteCacheManager
    state = prepared_backend
    storage, accesses = {}, []

    class RemoteBackend:

        def __init__(self, key):
            self.key = key

        def get(self, names):
            accesses.append((self.key, tuple(names)))
            return {name: storage[self.key, name] for name in names if (self.key, name) in storage}

        def put(self, name, value):
            storage[self.key, name] = value

    monkeypatch.setattr(knobs.cache, "manager_class", RemoteCacheManager)
    monkeypatch.setattr(knobs.cache, "remote_manager_class", RemoteBackend)
    first = state.driver.NPULauncher(state.src, metadata())
    accesses.clear()
    # Even materialized files must not bypass remote access accounting.
    for _ in range(3):
        assert state.driver.NPULauncher(state.src, metadata())._runtime_path == first._runtime_path
    assert len(accesses) == 6
    assert len(state.builds) == 2


def test_preparation_preserves_existing_content_keys(prepared_backend):
    state = prepared_backend
    instance = state.driver.NPULauncher(state.src, metadata())
    from triton.runtime.cache import get_cache_manager
    source = (state.root / "npu_utils.cpp").read_text()
    expected = hashlib.sha256(json.dumps([source, state.fingerprint], sort_keys=True).encode()).hexdigest()
    assert state.driver.NPUUtils().get_so_path() == get_cache_manager(expected).get_file("npu_utils.so")
    sources = launcher._runtime_sources(launcher._runtime_source_identity())
    config = launcher._runtime_config(launcher._cache_relative(state.driver.NPUUtils().get_so_path()),
                                      utils.backend_policy, None)
    key = hashlib.sha256(json.dumps([state.fingerprint, {**sources, **config}], sort_keys=True).encode()).hexdigest()
    assert Path(instance._runtime_path).parent.name == get_cache_manager(key).key


def _concurrent_artifact_worker(root, gate, queue):
    from triton import knobs
    from triton.runtime.cache import get_cache_manager
    knobs.cache.dir = root
    gate.wait(30)

    def build():
        with open(os.path.join(root, "builds.txt"), "a") as log:
            log.write(str(os.getpid()) + "\n")
        time.sleep(0.15)
        return b"compiled artifact"

    path = utils._get_or_build_npu_artifact(get_cache_manager("a" * 64), "test.so", build)
    queue.put(Path(path).read_bytes())


def test_multiple_processes_build_one_cold_artifact(tmp_path):
    context = multiprocessing.get_context("spawn")
    gate, queue = context.Event(), context.Queue()
    processes = [
        context.Process(target=_concurrent_artifact_worker, args=(str(tmp_path), gate, queue)) for _ in range(4)
    ]
    for process in processes:
        process.start()
    gate.set()
    try:
        assert [queue.get(timeout=60) for _ in processes] == [b"compiled artifact"] * 4
        for process in processes:
            process.join(30)
            assert process.exitcode == 0
    finally:
        for process in processes:
            if process.is_alive():
                process.terminate()
                process.join()
        queue.close()
    assert len((tmp_path / "builds.txt").read_text().splitlines()) == 1


def test_distinct_artifacts_build_concurrently_and_failure_retries(tmp_path, monkeypatch):
    from triton.runtime.cache import get_cache_manager
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path))
    barrier = threading.Barrier(2)

    def work(key):

        def build():
            barrier.wait(timeout=10)
            return b"built"

        return utils._get_or_build_npu_artifact(get_cache_manager(key * 64), "test.so", build)

    with ThreadPoolExecutor(max_workers=2) as pool:
        futures = [pool.submit(work, key) for key in ("a", "b")]
        assert all(Path(f.result()).read_bytes() == b"built" for f in futures)
    cache = get_cache_manager("c" * 64)
    with pytest.raises(RuntimeError, match="build failed"):
        utils._get_or_build_npu_artifact(cache, "test.so", lambda: (_ for _ in ()).throw(RuntimeError("build failed")))
    assert cache.get_file("test.so") is None
    path = utils._get_or_build_npu_artifact(cache, "test.so", lambda: b"retry succeeded")
    assert Path(path).read_bytes() == b"retry succeeded"


def test_cann_version_cache_tracks_root_replacement_and_missing_file(tmp_path, monkeypatch):
    monkeypatch.setattr(utils, "get_machine_arch", lambda: "x86_64")
    roots = [tmp_path / "cann-a", tmp_path / "cann-b"]
    for root, version in zip(roots, ("9.1.0", "8.5.0")):
        info = root / "x86_64-linux/ascend_toolkit_install.info"
        info.parent.mkdir(parents=True)
        info.write_text("version=" + version)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(roots[0]))
    assert utils.get_cann_version() == (9, 1, 0)
    monkeypatch.setenv("ASCEND_HOME_PATH", str(roots[1]))
    assert utils.get_cann_version() == (8, 5, 0)
    info = roots[1] / "x86_64-linux/ascend_toolkit_install.info"
    info.unlink()
    assert utils.get_cann_version() is None
    info.write_text("version=9.2.0")
    assert utils.get_cann_version() == (9, 2, 0)


def test_fingerprint_reuses_package_metadata_and_tracks_compiler(tmp_path, monkeypatch):
    import importlib.metadata
    compiler = tmp_path / "compiler"
    compiler.write_text("compiler v1")
    calls = []
    monkeypatch.setattr(utils, "_npu_ext_build_command", lambda name, path, **kw:
                        ([str(compiler), path, *kw.get("extra_cflags", ())], "output.so"))
    monkeypatch.setattr(utils, "get_cann_version", lambda: (9, 1, 0))
    monkeypatch.setattr(utils, "_npu_compiler_version", lambda path, identity: Path(path).read_text())
    monkeypatch.setattr(importlib.metadata, "version", lambda name: calls.append(name) or "test-version")
    utils._npu_package_versions.cache_clear()
    utils._npu_fingerprint.cache_clear()
    try:
        first = utils.npu_extension_fingerprint("test", extra_cflags=("-O3", ))
        assert utils.npu_extension_fingerprint("test", extra_cflags=("-O3", )) == first
        assert calls == ["torch", "torch_npu"]
        changed = utils.npu_extension_fingerprint("test", extra_cflags=("-O0", ))
        assert changed["command"] != first["command"]
        replacement = compiler.with_suffix(".new")
        replacement.write_text("compiler v2")
        replacement.replace(compiler)
        assert utils.npu_extension_fingerprint("test")["compiler_version"] == "compiler v2"
        first["packages"]["torch"] = "mutated"
        assert utils.npu_extension_fingerprint("test")["packages"]["torch"] == "test-version"
    finally:
        utils._npu_package_versions.cache_clear()
        utils._npu_fingerprint.cache_clear()


def test_build_command_reuses_stable_options_but_tracks_environment(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(utils, "get_backend_func", lambda name: calls.append(name) or ["-DTEST_ABI=1"])
    monkeypatch.setattr(utils, "get_cann_version", lambda: (9, 1, 0))
    monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "cann-a"))
    monkeypatch.setenv("CC", "compiler-a")
    utils._npu_ext_build_options.cache_clear()
    try:
        first, _ = utils._npu_ext_build_command("option_test", "/build/test.cpp", extra_cflags=("-O3", ))
        second, _ = utils._npu_ext_build_command("option_test", "/build/test.cpp", extra_cflags=("-O0", ))
        assert first[0] == second[0] == "compiler-a"
        assert "-O3" in first and "-O0" in second and calls == ["get_cc_cmd"]
        monkeypatch.setenv("CC", "compiler-b")
        third, _ = utils._npu_ext_build_command("option_test", "/build/test.cpp")
        assert third[0] == "compiler-b" and calls == ["get_cc_cmd"]
        monkeypatch.setenv("ASCEND_HOME_PATH", str(tmp_path / "cann-b"))
        fourth, _ = utils._npu_ext_build_command("option_test", "/headers/test.h", "/build/test.cpp")
        assert fourth[5] == "-I/headers"
        assert f"-I{tmp_path / 'cann-b/include'}" in fourth
        assert calls == ["get_cc_cmd", "get_cc_cmd"]
    finally:
        utils._npu_ext_build_options.cache_clear()


def _grid_contract(sequence):
    return dict(version=2, extent_source="runtime_original_grid", hidden_extent_axes=[0, 1],
                hidden_argument_order=["originalGridX", "originalGridY"], hidden_argument_types=["i32", "i32"],
                transforms=[dict(order=i, kind="ceil_div", axis=axis, factor=factor,
                                 persistent_coverage=persistent, grid_stride_abi_verified=persistent)
                            for i, (axis, factor, persistent) in enumerate(sequence)])


@pytest.mark.parametrize("sequence,expected", [
    ([(1, 16, False)], launcher.IAT),
    ([(0, 64, True)], launcher.PTSM),
    ([(1, 16, False), (0, 4, True)], launcher.IAT | launcher.PTSM),
])
@pytest.mark.parametrize("as_json", [False, True])
def test_program_grid_plan_modes(policy, sequence, expected, as_json):
    contract = _grid_contract(sequence)
    spec = launcher.make_launch_spec(metadata(
        program_grid_mapping_applied=True,
        program_grid_transforms=json.dumps(contract) if as_json else contract,
        ptsm_cap_authorized=any(t[2] for t in sequence)), policy)
    assert spec.flags & (launcher.IAT | launcher.PTSM) == expected
    assert (spec.coalesce_factor, spec.coalesce_axis) == (1, -1)


@pytest.mark.parametrize("field", ["program_grid_mapping_applied", "row_coalescing_applied",
                                    "auto_blockify_enabled", "ptsm_cap_authorized"])
@pytest.mark.parametrize("value", [None, 0, "false"])
def test_program_grid_requires_explicit_compiler_gates(policy, field, value):
    with pytest.raises(RuntimeError, match=f"compiler metadata missing {field}"):
        launcher.make_launch_spec(metadata(**{field: value}), policy)


@pytest.mark.parametrize("changes,error", [
    ({"coalesce_factor": 16, "coalesce_axis": 1}, "unmatched RowCoalescing"),
    ({"row_coalescing_applied": True}, "requires valid coalesce metadata"),
    ({"program_grid_mapping_applied": True}, "requires program_grid_transforms"),
    ({"program_grid_transforms": _grid_contract([(1, 16, False)])}, "disagrees"),
    ({"ptsm_cap_authorized": True}, "requires a persistent transform"),
    ({"auto_blockify_enabled": True, "ptsm_cap_authorized": True}, "cannot both be true"),
])
def test_program_grid_rejects_inconsistent_metadata(policy, changes, error):
    with pytest.raises(RuntimeError, match=error):
        launcher.make_launch_spec(metadata(**changes), policy)


@pytest.mark.parametrize("changes,error", [
    ({"ptsm_cap_authorized": False}, "lacks PTSM cap authorization"),
    ({"mix_mode": "mix"}, "requires final mix_mode=aiv"),
    ({"row_coalescing_applied": True}, "conflict with legacy RowCoalescing"),
    ({"coalesce_factor": 4}, "conflict with legacy coalesce metadata"),
    ({"auto_blockify_enabled": True}, "conflicts with a rewritten program mapping"),
    ({"program_grid_transforms": _grid_contract([(0, 32, True)])}, "invalid program_grid_transforms"),
])
def test_program_grid_rejects_unsafe_mapping_combinations(policy, changes, error):
    fields = dict(program_grid_mapping_applied=True, program_grid_transforms=_grid_contract([(0, 64, True)]),
                  ptsm_cap_authorized=True)
    fields.update(changes)
    with pytest.raises(RuntimeError, match=error):
        launcher.make_launch_spec(metadata(**fields), policy)


@pytest.mark.parametrize("auto_map", [False, True])
@pytest.mark.parametrize("blacklisted", [False, True])
@pytest.mark.parametrize("auto_blockify", [False, True])
def test_block_cap_preserves_upstream_policy(policy, monkeypatch, auto_map, blacklisted, auto_blockify):
    monkeypatch.setattr(utils, "_is_auto_map_parallel_blocks_enabled", lambda: auto_map)
    spec = launcher.make_launch_spec(metadata(has_auto_blockify_blacklist_op=blacklisted,
                                             auto_blockify_enabled=auto_blockify), policy)
    assert bool(spec.flags & launcher.AUTO_MAP) == (auto_map and not blacklisted)


@pytest.fixture(scope="module")
def native_grid_probe(tmp_path_factory):
    import shutil
    import subprocess
    source = Path(__file__).with_name("launcher_grid_test.cpp")
    output = tmp_path_factory.mktemp("native-grid") / "probe"
    compiler = shutil.which("c++") or utils._get_bisheng_path()
    subprocess.run([compiler, "-std=c++17", "-O2", "-I" + str(Path(launcher.__file__).with_name("launcher_src")),
                    str(source), "-o", str(output)], check=True, capture_output=True, text=True)
    return output


@pytest.mark.parametrize("sequence,mode", [
    ([(1, 16, False)], launcher.IAT),
    ([(0, 64, True)], launcher.PTSM),
    ([(1, 16, False), (0, 4, True)], launcher.IAT | launcher.PTSM),
])
@pytest.mark.parametrize("auto_map", [False, True])
def test_native_grid_matches_compiler_reference(native_grid_probe, sequence, mode, auto_map):
    import random
    import subprocess
    from triton.backends.ascend.program_grid import apply_program_grid_transforms
    contract = _grid_contract(sequence)
    rng = random.Random(2109)
    grids = [(1, 1, 1), (4097, 17, 3), (65, 33, 1), (2147483647, 1, 1)]
    grids += [tuple(rng.randint(1, n) for n in (32768, 128, 4)) for _ in range(32)]
    for physical in (1, 20, 40, 72):
        for grid in grids:
            flags = mode | (launcher.AUTO_MAP if auto_map else 0)
            values = [int(v) for v in subprocess.check_output(
                [str(native_grid_probe), str(flags), str(physical), *map(str, grid)], text=True).split()]
            expected = apply_program_grid_transforms(grid, contract, physical_core_count=physical)
            logical = expected[0] * expected[1] * expected[2]
            assert values[:5] == [*expected, logical, min(logical, physical) if auto_map else logical]
            # lock/workspace, pointer, i8, padding, original X/Y, final X/Y/Z,
            # padding, debug pointer: exact ABI offsets, independent of grid.
            assert values[5:] == [24, 28, 36, 48, 56]
    values = subprocess.check_output([str(native_grid_probe), str(mode), "40", "0", "17", "3"], text=True)
    assert list(map(int, values.split()))[:5] == [0, 17, 3, 0, 0]


@pytest.fixture
def host_compiler(tmp_path, monkeypatch):
    """A native driver whose version comes from a replaceable shared library."""
    cxx = utils._get_cxx()
    source = tmp_path / "driver.cpp"
    source.write_text('''
#include <cstdio>
#include <cstdlib>
extern "C" const char *tool_version();
int main() {
  if (const char *log = std::getenv("TRITON_TEST_VERSION_LOG")) {
    FILE *f = std::fopen(log, "a");
    if (!f) return 1;
    std::fputs("query\\n", f);
    std::fclose(f);
  }
  std::puts(tool_version());
}
''')
    library = tmp_path / "libversion.so"

    def replace_version(version):
        lib_source = tmp_path / "version.cpp"
        lib_source.write_text('extern "C" const char *tool_version() { return "' + version + '"; }')
        replacement = tmp_path / "libversion.new.so"
        subprocess.run([cxx, str(lib_source), "-shared", "-fPIC", "-Wl,--build-id=sha1", "-o",
                        str(replacement)], check=True)
        replacement.replace(library)

    replace_version("test-compiler-v1")
    compiler = tmp_path / "clang"
    subprocess.run([
        cxx,
        str(source), "-L" + str(tmp_path), "-lversion", "-Wl,-rpath,$ORIGIN", "-Wl,--build-id=sha1", "-o",
        str(compiler)
    ], check=True)
    log = tmp_path / "queries.txt"
    monkeypatch.setenv("TRITON_TEST_VERSION_LOG", str(log))
    for name in tuple(os.environ):
        if ((name.startswith("LD_") and name != "LD_LIBRARY_PATH") or name.startswith(("LC_", "CLANG_"))
                or name in ("LANGUAGE", "CCC_OVERRIDE_OPTIONS")):
            monkeypatch.delenv(name)
    monkeypatch.setenv("LANG", "C")
    monkeypatch.setenv("LC_ALL", "C")
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    utils._npu_compiler_version.cache_clear()
    yield SimpleNamespace(path=str(compiler), library=library, log=log, replace_version=replace_version, root=tmp_path)
    utils._npu_compiler_version.cache_clear()
    utils._npu_fingerprint.cache_clear()


def _query_host_compiler(path):
    resolved = os.path.realpath(path)
    return utils._npu_compiler_version(resolved, utils._file_identity(resolved))


def _host_compiler_process(path, cache_dir, start, results):
    from triton import knobs
    with knobs.cache.scope():
        knobs.cache.dir = cache_dir
        utils._npu_compiler_version.cache_clear()
        start.wait(30)
        results.put(_query_host_compiler(path))


def _query_host_compiler_processes(path, cache_dir, count):
    context = multiprocessing.get_context("spawn")
    start, results = context.Event(), context.Queue()
    children = [
        context.Process(target=_host_compiler_process, args=(path, cache_dir, start, results)) for _ in range(count)
    ]
    try:
        for child in children:
            child.start()
        start.set()
        versions = [results.get(timeout=90) for _ in children]
        for child in children:
            child.join(timeout=90)
            assert child.exitcode == 0
        return versions
    finally:
        for child in children:
            if child.is_alive():
                child.terminate()
            child.join(timeout=10)


def test_compiler_version_reused_across_processes_and_coordinated(host_compiler):
    tool = host_compiler
    cache_dir = str(tool.root / "cache")
    # Simultaneous cold processes query once, then another fresh process hits disk.
    assert _query_host_compiler_processes(tool.path, cache_dir, 4) == ["test-compiler-v1"] * 4
    assert tool.log.read_text().splitlines() == ["query"]
    assert _query_host_compiler_processes(tool.path, cache_dir, 1) == ["test-compiler-v1"]
    assert tool.log.read_text().splitlines() == ["query"]


def test_compiler_library_build_id_invalidates_even_with_unchanged_stat(host_compiler, monkeypatch):
    tool = host_compiler
    old_identity = utils._file_identity(tool.library)
    file_identity = utils._file_identity
    monkeypatch.setattr(utils, "_file_identity", lambda path: old_identity
                        if Path(path) == tool.library else file_identity(path))
    assert _query_host_compiler(tool.path) == "test-compiler-v1"
    old_build_id = utils._npu_elf_info(tool.library)[1]
    tool.replace_version("test-compiler-v2")
    assert utils._npu_elf_info(tool.library)[1] != old_build_id
    utils._npu_compiler_version.cache_clear()  # A new process has no LRU entries.
    assert _query_host_compiler(tool.path) == "test-compiler-v2"
    assert len(tool.log.read_text().splitlines()) == 2


def test_compiler_selection_symlink_and_atomic_replacement(host_compiler, monkeypatch):
    tool = host_compiler
    alias = tool.root / "clang++"
    alias.symlink_to(tool.path)
    monkeypatch.delenv("CC", raising=False)
    monkeypatch.setenv("PATH", str(tool.root) + os.pathsep + os.environ["PATH"])
    selected = lambda: shutil.which(utils._get_cxx()) or utils._get_cxx()
    assert _query_host_compiler(selected()) == "test-compiler-v1"
    assert _query_host_compiler(selected()) == "test-compiler-v1"
    replacement_dir = tool.root / "replacement"
    replacement_dir.mkdir()
    replacement = replacement_dir / "clang"
    shutil.copy2(tool.path, replacement)
    shutil.copy2(tool.library, replacement_dir / tool.library.name)
    alias.unlink()
    alias.symlink_to(replacement)
    assert _query_host_compiler(selected()) == "test-compiler-v1"
    monkeypatch.setenv("CC", tool.path)
    assert _query_host_compiler(selected()) == "test-compiler-v1"
    new = tool.root / "clang.new"
    shutil.copy2(tool.path, new)
    new.replace(tool.path)
    assert _query_host_compiler(selected()) == "test-compiler-v1"
    (replacement_dir / "clang++").symlink_to(replacement)
    monkeypatch.delenv("CC")
    monkeypatch.setenv("PATH", str(replacement_dir) + os.pathsep + os.environ["PATH"])
    assert _query_host_compiler(selected()) == "test-compiler-v1"
    # Returning to an already known, unchanged compiler reuses its process entry.
    assert len(tool.log.read_text().splitlines()) == 3


@pytest.mark.parametrize("bad_record", [b"{", b"[]", b'\xff', b'{"key":"wrong","version":"stale"}', None])
def test_compiler_version_repairs_corrupt_metadata(host_compiler, bad_record):
    tool = host_compiler
    assert _query_host_compiler(tool.path) == "test-compiler-v1"
    record, = (tool.root / "cache").rglob("compiler-version.json")
    if bad_record is None:
        data = json.loads(record.read_text())
        data["sha256"] = "corrupt"
        record.write_text(json.dumps(data))
    else:
        record.write_bytes(bad_record)
    utils._npu_compiler_version.cache_clear()
    assert _query_host_compiler(tool.path) == "test-compiler-v1"
    utils._npu_compiler_version.cache_clear()
    assert _query_host_compiler(tool.path) == "test-compiler-v1"
    assert len(tool.log.read_text().splitlines()) == 2


def test_compiler_version_query_failure_retries_without_success_record(host_compiler, monkeypatch):
    tool = host_compiler
    check_output = subprocess.check_output

    def fail_version(command, **kwargs):
        if command == [tool.path, "--version"]:
            raise subprocess.CalledProcessError(1, command)
        return check_output(command, **kwargs)

    with monkeypatch.context() as patch:
        patch.setattr(subprocess, "check_output", fail_version)
        with pytest.raises(subprocess.CalledProcessError):
            _query_host_compiler(tool.path)
    assert list((tool.root / "cache").rglob("compiler-version.json")) == []
    assert _query_host_compiler(tool.path) == "test-compiler-v1"


def test_compiler_version_cache_roots_deletion_and_write_failure(host_compiler, monkeypatch):
    from triton.runtime.cache import FileCacheManager
    tool = host_compiler
    assert _query_host_compiler(tool.path) == "test-compiler-v1"
    record, = (tool.root / "cache").rglob("compiler-version.json")
    record.unlink()
    utils._npu_compiler_version.cache_clear()
    assert _query_host_compiler(tool.path) == "test-compiler-v1"
    assert record.exists()
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tool.root / "cache-two"))
    utils._npu_compiler_version.cache_clear()
    with monkeypatch.context() as patch:

        def fail_put(*args, **kwargs):
            raise PermissionError("read-only metadata storage")

        patch.setattr(FileCacheManager, "put", fail_put)
        assert _query_host_compiler(tool.path) == "test-compiler-v1"
    assert not list((tool.root / "cache-two").rglob("compiler-version.json"))
    utils._npu_compiler_version.cache_clear()
    assert _query_host_compiler(tool.path) == "test-compiler-v1"
    assert len(tool.log.read_text().splitlines()) == 4


def test_compiler_version_custom_cache_does_not_add_manager_queries(host_compiler, monkeypatch):
    from triton import knobs
    from triton.runtime.cache import FileCacheManager

    class CustomCache(FileCacheManager):

        def __init__(self, *args, **kwargs):
            pytest.fail("compiler identity must not instantiate a custom cache manager")

    knobs.cache.manager_class = CustomCache
    monkeypatch.setattr(utils, "_npu_compiler_identity", lambda path: pytest.fail("unexpected identity query"))
    assert _query_host_compiler(host_compiler.path) == "test-compiler-v1"


@pytest.mark.parametrize(
    "environment",
    [{"LD_PRELOAD": "/injected.so"}, {"LD_AUDIT": "/audit.so"}, {"LANGUAGE": "de"}, {"CCC_OVERRIDE_OPTIONS": "custom"}])
def test_compiler_version_unknown_environment_queries_live(host_compiler, monkeypatch, environment):
    for name, value in environment.items():
        monkeypatch.setenv(name, value)
    calls = []
    monkeypatch.setattr(subprocess, "check_output", lambda command, **kwargs: calls.append(command) or "live version\n")
    assert _query_host_compiler(host_compiler.path) == "live version"
    assert calls == [[host_compiler.path, "--version"]]


def test_compiler_version_script_wrapper_does_not_persist(tmp_path, monkeypatch):
    compiler = tmp_path / "clang"
    compiler.write_text("#!/bin/sh\nprintf 'wrapper-version\\n'\n")
    compiler.chmod(0o755)
    monkeypatch.setenv("TRITON_CACHE_DIR", str(tmp_path / "cache"))
    assert _query_host_compiler(str(compiler)) == "wrapper-version"
    assert not (tmp_path / "cache").exists()


def test_compiler_version_unstable_installation_does_not_publish(host_compiler, monkeypatch):
    tool = host_compiler
    identity = utils._npu_compiler_identity(tool.path)
    calls = iter((identity, None))
    with monkeypatch.context() as patch:
        patch.setattr(utils, "_npu_compiler_identity", lambda path: next(calls))
        assert _query_host_compiler(tool.path) == "test-compiler-v1"
    assert not list((tool.root / "cache").rglob("compiler-version.json"))
    utils._npu_compiler_version.cache_clear()
    assert _query_host_compiler(tool.path) == "test-compiler-v1"
    assert len(tool.log.read_text().splitlines()) == 2


@pytest.mark.parametrize("loader_result", ["", "unknown output", "libmissing.so => not found"])
def test_compiler_version_unresolved_dependencies_query_live(host_compiler, monkeypatch, loader_result):
    tool = host_compiler
    check_output = subprocess.check_output

    def output(command, **kwargs):
        if "--list" in command:
            return loader_result
        return check_output(command, **kwargs)

    monkeypatch.setattr(subprocess, "check_output", output)
    assert _query_host_compiler(tool.path) == "test-compiler-v1"
    assert not list((tool.root / "cache").rglob("compiler-version.json"))
