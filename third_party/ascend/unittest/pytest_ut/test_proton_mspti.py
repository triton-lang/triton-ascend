"""Run the MSPTI regressions in a fresh interpreter with CANN's library preloaded."""
import ctypes
import json
import math
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import warnings

import pytest
import torch

pytest.importorskip("torch_npu")
pytest.importorskip("triton._C.libproton")
import triton
import triton.language as tl
import triton.profiler as proton


def _default_mspti_library(home):
    if home:
        active = Path(home)
        # The A3 CI image uses CANN 8.5. Try a side-by-side 9.1 MSPTI when available.
        roots = (active.with_name("cann-9.1.0"), active) if active.name == "cann-8.5.0" else (active, )
        for root in roots:
            for directory in ("lib64", "tools/mspti/lib64"):
                library = root / directory / "libmspti.so"
                if library.is_file():
                    if root != active:
                        warnings.warn(f"Testing {library} with active CANN at {active}", RuntimeWarning, stacklevel=2)
                    return str(library)
    return "libmspti.so"


def test_mspti(tmp_path):
    env = os.environ.copy()
    preloads = env.get("LD_PRELOAD", "").replace(":", " ").split()
    library = next((p for p in preloads if Path(p).name == "libmspti.so" or Path(p).name.startswith("libmspti.so.")),
                   None)
    if not library:
        library = env.get("PROTON_MSPTI_LIBRARY_PATH") or _default_mspti_library(env.get("ASCEND_HOME_PATH"))
        env["LD_PRELOAD"] = " ".join([library, *preloads])
    # Do not inherit e.g. -n auto: the parent fixture already assigned a device.
    env.pop("PYTEST_ADDOPTS", None)
    result = subprocess.run(
        [sys.executable, __file__, str(torch.npu.current_device()),
         str(tmp_path), library], env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
        errors="replace", timeout=600)
    selected = f"MSPTI: {library} (ASCEND_HOME_PATH={env.get('ASCEND_HOME_PATH')})"
    if result.returncode:
        details = ""
        if Path(env.get("ASCEND_HOME_PATH", "")).name == "cann-8.5.0":
            details = _probe_failure(env, library, tmp_path)
        pytest.fail(f"{selected}; child exited with {result.returncode}\n{details}\n{result.stdout}", pytrace=False)
    print(selected)
    print(result.stdout, end="")


def _probe_failure(env, library, tmp):
    home = Path(env["ASCEND_HOME_PATH"])
    so = tmp / "mspti_probe.so"
    includes = [home / "tools/mspti/include", home / "include"]
    if env.get("MSPTI_INCLUDE_DIR"):
        includes.insert(0, Path(env["MSPTI_INCLUDE_DIR"]))
    build = [
        "c++", "-std=c++17", "-shared", "-fPIC", "-pthread", *(f"-I{d}" for d in includes),
        str(Path(__file__).with_name("proton_mspti_probe.cpp")), "-o",
        str(so)
    ]
    cmd = [sys.executable, __file__, str(torch.npu.current_device()), str(tmp), library, "--raw"]
    lines = []
    try:
        r = subprocess.run(build + ([library] if Path(library).is_absolute() else ["-lmspti"]), capture_output=True,
                           text=True, timeout=45)
        if r.returncode:
            return f"MSPTI-PROBE compile failed: {r.stderr[-2000:]}"
        for mode in ("late-triton", "late-torch", "early-triton"):
            logdir = tmp / mode
            logdir.mkdir()
            probe_env = dict(env, ASCEND_PROCESS_LOG_PATH=str(logdir), ASCEND_GLOBAL_LOG_LEVEL="0",
                             ASCEND_SLOG_PRINT_TO_STDOUT="0")
            r = subprocess.run([*cmd, mode], env=probe_env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               errors="replace", timeout=90)
            lines.append(f"MSPTI-PROBE mode={mode} rc={r.returncode}")
            lines.extend(line for line in r.stdout.splitlines() if line.startswith("MSPTI-PROBE"))
            if r.returncode:
                lines.extend(r.stdout.splitlines()[-3:])
            # Summarize only failure locations, not raw machine logs.
            failures = set()
            for path in logdir.rglob("*.log"):
                with path.open(errors="replace") as f:
                    for line in f:
                        if ("[ERROR]" in line or "[WARN]" in line) and re.search(r"profil|mspti|prof_task|prof_drv",
                                                                                 line, re.I):
                            failures.update(re.findall(r"\[([\w./-]+\.(?:cpp|cc|c|h):\d+)\]", line))
            lines.append(f"MSPTI-PROBE warning/error sites: {sorted(failures)[:24]}")
    except (OSError, subprocess.TimeoutExpired) as exc:
        lines.append(f"MSPTI-PROBE diagnostic did not finish: {exc}")
    return "\n".join(lines)


def _run_raw(device, tmp, mode):
    probe = ctypes.CDLL(str(tmp / "mspti_probe.so"))
    print(f"MSPTI-PROBE mode={mode} initialized_at_entry={torch.npu.is_initialized()}", flush=True)
    if mode.startswith("early"):
        assert probe.probe_start() == 0
    torch.npu.set_device(device)
    x = torch.arange(1024, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)

    def add():
        if mode.endswith("torch"):
            torch.add(x, 1, out=y)
        else:
            lifecycle_add_kernel[(4, )](x, y, n=1024, block=256)
        torch.npu.synchronize()

    add()
    if mode.startswith("late"):
        assert probe.probe_start() == 0
    probe.probe_mark(1)
    for _ in range(3):
        add()
    probe.probe_mark(0)
    assert probe.probe_stop() == 0
    torch.testing.assert_close(y.cpu(), x.cpu() + 1)


@triton.jit
def lifecycle_add_kernel(x_ptr, y_ptr, n: tl.constexpr, block: tl.constexpr):
    offsets = tl.program_id(0) * block + tl.arange(0, block)
    mask = offsets < n
    tl.store(
        y_ptr + offsets,
        tl.load(x_ptr + offsets, mask=mask) + 1,
        mask=mask,
    )


def walk_profile(nodes):
    for node in nodes:
        yield node
        yield from walk_profile(node.get("children", []))


def launch(x, y, count, synchronize=True):
    for _ in range(count):
        lifecycle_add_kernel[(4, )](x, y, n=1024, block=256)
    if synchronize:
        torch.npu.synchronize()


def kernel_count(path):
    profile = json.loads(path.read_text(encoding="utf-8"))
    return sum(
        node.get("metrics", {}).get("count", 0)
        for node in walk_profile(profile)
        if node.get("frame", {}).get("name") == "lifecycle_add_kernel")


@pytest.fixture(scope="module")
def npu_tensors():
    x = torch.arange(1024, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    launch(x, y, 1)
    return x, y


def case_deactivate_reactivate(tmp_path, npu_tensors):
    x, y = npu_tensors
    path = tmp_path / "reactivate.hatchet"
    session = proton.start(str(path.with_suffix("")), backend="mspti", hook="triton")
    try:
        with proton.scope("enabled_before"):
            launch(x, y, 2)
        proton.deactivate(session)
        launch(x, y, 3)
        proton.activate(session)
        with proton.scope("enabled_after"):
            launch(x, y, 4)
    finally:
        proton.finalize(session)

    profile = json.loads(path.read_text(encoding="utf-8"))
    scopes = {node.get("frame", {}).get("name") for node in walk_profile(profile)}
    assert kernel_count(path) == 6
    assert {"enabled_before", "enabled_after"}.issubset(scopes)


def case_multiple_sessions(tmp_path, npu_tensors):
    x, y = npu_tensors
    path0 = tmp_path / "session0.hatchet"
    path1 = tmp_path / "session1.hatchet"
    session0 = proton.start(str(path0.with_suffix("")), backend="mspti", hook="triton")
    session1 = proton.start(str(path1.with_suffix("")), backend="mspti", hook="triton")
    try:
        with proton.scope("both_sessions"):
            launch(x, y, 2)
        proton.deactivate(session0)
        with proton.scope("second_session_only"):
            launch(x, y, 3)
    finally:
        proton.finalize(session0)
        proton.finalize(session1)

    assert kernel_count(path0) == 2
    assert kernel_count(path1) == 5


def case_restart_after_finalize(tmp_path, npu_tensors):
    x, y = npu_tensors
    for index, expected in enumerate((1, 2)):
        path = tmp_path / f"restart{index}.hatchet"
        session = proton.start(str(path.with_suffix("")), hook="triton")
        try:
            launch(x, y, expected)
        finally:
            proton.finalize(session)
        assert kernel_count(path) == expected


def case_state_without_hook(tmp_path, npu_tensors):
    x, y = npu_tensors
    path = tmp_path / "state_no_hook.hatchet"
    session = proton.start(str(path.with_suffix("")), backend="mspti")
    try:
        with proton.scope("state_scope"):
            with proton.state("warmup"):
                launch(x, y, 1)
            with proton.state("steady"):
                launch(x, y, 1)
    finally:
        proton.finalize(session)

    profile = json.loads(path.read_text(encoding="utf-8"))
    root = profile[0]
    scope = next(child for child in root["children"] if child.get("frame", {}).get("name") == "state_scope")
    states = {child.get("frame", {}).get("name"): child for child in scope.get("children", [])}
    assert states.keys() == {"warmup", "steady"}
    for state in states.values():
        kernels = [
            node for node in walk_profile(state.get("children", []))
            if node.get("frame", {}).get("name") == "lifecycle_add_kernel"
        ]
        assert len(kernels) == 1
        assert kernels[0].get("metrics", {}).get("count") == 1


def case_multiple_streams(tmp_path, npu_tensors):
    x, _ = npu_tensors
    path = tmp_path / "streams.hatchet"
    session = proton.start(str(path.with_suffix("")), backend="mspti", hook="triton")
    try:
        streams = [torch.npu.Stream(), torch.npu.Stream()]
        outputs = [torch.empty_like(x), torch.empty_like(x)]
        for stream, output in zip(streams, outputs):
            with torch.npu.stream(stream):
                launch(x, output, 4, synchronize=False)
        torch.npu.synchronize()
    finally:
        proton.finalize(session)

    assert kernel_count(path) == 8


def case_multiple_threads(tmp_path, npu_tensors):
    x, _ = npu_tensors
    path = tmp_path / "threads.hatchet"
    session = proton.start(str(path.with_suffix("")), backend="mspti", hook="triton")
    errors = []

    def worker():
        try:
            torch.npu.set_device(x.device)
            stream = torch.npu.Stream()
            output = torch.empty_like(x)
            with torch.npu.stream(stream):
                launch(x, output, 3, synchronize=False)
            stream.synchronize()
        except Exception as error:
            errors.append(error)

    try:
        threads = [threading.Thread(target=worker) for _ in range(2)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert not errors
        torch.npu.synchronize()
    finally:
        proton.finalize(session)

    assert kernel_count(path) == 6


def case_chrome_trace(tmp_path, npu_tensors):
    x, y = npu_tensors
    path = tmp_path / "trace.chrome_trace"
    session = proton.start(
        str(path.with_suffix("")),
        data="trace",
        backend="mspti",
        hook="triton",
    )
    try:
        with proton.scope("trace_scope"):
            launch(x, y, 2)
    finally:
        proton.finalize(session)

    trace = json.loads(path.read_text(encoding="utf-8"))
    assert sum(event.get("name") == "lifecycle_add_kernel" for event in trace["traceEvents"]) == 2


def case_smoke(tmp_path, npu_tensors):
    x, y = npu_tensors
    path = tmp_path / "smoke.hatchet"
    session = proton.start(str(path.with_suffix("")), backend="mspti", hook="triton")
    try:
        with proton.scope("vector_add"):
            launch(x, y, 5)
    finally:
        proton.finalize(session)
    torch.testing.assert_close(y, x + 1)
    root, devices = json.loads(path.read_text())
    scope, = root["children"]
    kernel, = scope["children"]
    assert scope["frame"]["name"] == "vector_add"
    assert kernel["frame"]["name"] == "lifecycle_add_kernel"
    metrics = kernel["metrics"]
    assert metrics["count"] == 5 and metrics["time (ns)"] > 0
    info = devices["NPU"][str(metrics["device_id"])]
    assert info["arch"] not in ("", "ascend") and info["bus_width"] >= 0
    assert all(info[key] > 0 for key in ("num_sms", "clock_rate", "memory_clock_rate"))


def launch_metadata(grid, metadata, args):
    return {"name": "named_add", "flops32": args["n"], "bytes": args["n"] * 12}


@triton.jit(launch_metadata=launch_metadata)
def metadata_add(x, y, z, n: tl.constexpr, BLOCK: tl.constexpr):
    offsets = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    tl.store(z + offsets,
             tl.load(x + offsets, offsets < n, other=0) + tl.load(y + offsets, offsets < n, other=0), offsets < n)


@pytest.mark.parametrize("hook", [None, "triton"])
@pytest.mark.parametrize("limit_cores", [False, True])
def case_mspti_metadata_and_activation(tmp_path, monkeypatch, hook, limit_cores):
    n = 1024
    x = torch.arange(n, device="npu", dtype=torch.float32)
    y = torch.ones_like(x)
    z = torch.empty_like(x)

    def launch():
        metadata_add[(4, )](x, y, z, n, BLOCK=256)
        torch.npu.synchronize()

    launch()
    utils = triton.runtime.driver.active.utils
    hardware_aic, hardware_aiv = utils.get_device_core()
    divisor = math.gcd(hardware_aic, hardware_aiv)
    if limit_cores:
        monkeypatch.setenv("NPU_DEVICE_LIMIT", f"{hardware_aic // divisor},{hardware_aiv // divisor}")
    else:
        monkeypatch.delenv("NPU_DEVICE_LIMIT", raising=False)
    expected_aiv = hardware_aiv // divisor if limit_cores else hardware_aiv
    base = tmp_path / "mspti"
    # Also exercises default NPU -> MSPTI backend selection.
    session = proton.start(str(base), hook=hook)
    try:
        with proton.scope("before"):
            launch()
        proton.deactivate(session)
        launch()
        proton.activate(session)
        with proton.scope("after"):
            launch()
            launch()
    finally:
        proton.finalize(session)
    torch.testing.assert_close(z, x + y)
    profile = json.loads(base.with_suffix(".hatchet").read_text())
    scopes = {child["frame"]["name"]: child for child in profile[0]["children"]}
    assert scopes.keys() == {"before", "after"}
    for scope, count in (("before", 1), ("after", 2)):
        kernels = [node for node in walk_profile([scopes[scope]]) if node.get("metrics", {}).get("count", 0)]
        assert len(kernels) == 1
        kernel = kernels[0]
        metrics = kernel["metrics"]
        assert metrics["count"] == count
        assert metrics["time (ns)"] > 0
        assert metrics["device_type"] == "NPU"
        assert str(metrics["device_id"]) in profile[1]["NPU"]
        device_info = profile[1]["NPU"][str(metrics["device_id"])]
        assert device_info["num_sms"] == expected_aiv
        assert device_info["arch"] == utils.get_arch()
        logical_device = int(metrics["device_id"])
        assert device_info["clock_rate"] == utils.get_vector_clock_rate(logical_device)
        assert device_info["memory_clock_rate"] == utils.get_hbm_clock_rate(logical_device)
        if hook == "triton":
            assert kernel["frame"]["name"] == "named_add"
            assert metrics["flops32"] == n * count
            assert metrics["bytes"] == n * 12 * count
        else:
            assert kernel["frame"]["name"] == "metadata_add"
            assert "flops32" not in metrics


def case_mspti_unavailable_device_clocks(tmp_path, monkeypatch):
    n = 1024
    x = torch.arange(n, device="npu", dtype=torch.float32)
    y = torch.ones_like(x)
    z = torch.empty_like(x)
    metadata_add[(4, )](x, y, z, n, BLOCK=256)
    torch.npu.synchronize()
    utils = triton.runtime.driver.active.utils
    native = utils._load_mod()
    monkeypatch.setattr(utils, "get_vector_clock_rate", lambda device: native.get_vector_clock_rate(999999))
    monkeypatch.setattr(utils, "get_hbm_clock_rate", lambda device: native.get_hbm_clock_rate(999999))
    base = tmp_path / "unavailable_clocks"
    session = proton.start(str(base), backend="mspti", hook="triton")
    try:
        metadata_add[(4, )](x, y, z, n, BLOCK=256)
        torch.npu.synchronize()
    finally:
        with pytest.warns(RuntimeWarning, match="clock rate is unavailable"):
            proton.finalize(session)
    profile = json.loads(base.with_suffix(".hatchet").read_text())
    kernels = [node for node in walk_profile([profile[0]]) if node.get("metrics", {}).get("count", 0)]
    assert len(kernels) == 1
    assert kernels[0]["metrics"]["count"] == 1
    assert kernels[0]["metrics"]["time (ns)"] > 0
    info = profile[1]["NPU"][str(kernels[0]["metrics"]["device_id"])]
    assert info["clock_rate"] == info["memory_clock_rate"] == 0


if __name__ == "__main__":
    # The ELF loader ignores missing LD_PRELOAD entries; do not silently test dlopen instead.
    ctypes.CDLL(sys.argv[3], mode=os.RTLD_NOLOAD)
    if sys.argv[4:5] == ["--raw"]:
        _run_raw(int(sys.argv[1]), Path(sys.argv[2]), sys.argv[5])
        sys.exit(0)
    torch.npu.set_device(int(sys.argv[1]))
    sys.exit(
        pytest.main([
            __file__, "--noconftest", "-o", "python_functions=case_*", "-o", "addopts=", "--basetemp",
            str(Path(sys.argv[2]) / "mspti"), "-v"
        ]))
