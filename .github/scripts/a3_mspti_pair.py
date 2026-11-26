"""Temporary #3401 diagnostic; run on one A3 runner before xdist starts."""
import csv
import ctypes
import hashlib
import os
from pathlib import Path
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
from collections import Counter

HERE = Path(__file__).resolve()
TEST = HERE.parents[2] / "third_party/ascend/unittest/pytest_ut"


def workload(mode, out):
    import torch
    sys.path.insert(0, str(TEST))
    from test_proton_mspti import lifecycle_add_kernel
    torch.npu.set_device(0)
    physical = ctypes.c_uint32()
    assert ctypes.CDLL("libruntime.so").rtGetDevicePhyIdByIndex(0, ctypes.byref(physical)) == 0
    print(f"PAIR device={physical.value} mode={mode} soc={torch.npu.get_device_name(0)}", flush=True)
    x = torch.arange(1024, device="npu", dtype=torch.float32)
    y = torch.empty_like(x)
    torch.npu.synchronize()
    probe = None
    if mode in ("stock", "shared"):
        probe = ctypes.CDLL(str(out / "probe.so"))
        assert probe.probe_start() == 0
        probe.probe_mark(1)
    for _ in range(3):
        lifecycle_add_kernel[(4, )](x, y, n=1024, block=256)
        torch.npu.synchronize()
    if probe is not None:
        probe.probe_mark(0)
        if mode == "shared":
            assert probe.probe_shared_drain() == 0
        assert probe.probe_stop() == 0
    torch.testing.assert_close(y.cpu(), x.cpu() + 1)
    (out / (mode + ".device")).write_text(str(physical.value))
    print(f"PAIR verified mode={mode} launches=3", flush=True)


def execute(cmd, env, log, timeout=180):
    with log.open("w") as f:
        try:
            rc = subprocess.run(cmd, env=env, stdout=f, stderr=subprocess.STDOUT, timeout=timeout).returncode
        except (OSError, subprocess.TimeoutExpired) as exc:
            print(f"PAIR command unavailable: {exc}")
            rc = -1
    lines = log.read_text(errors="replace").splitlines()
    for line in lines:
        if line.startswith(("PAIR ", "MSPTI-PROBE", "STAGE ")):
            print(line)
    print(f"PAIR {log.name} rc={rc}")
    if rc:
        print(f"PAIR failure: {lines[-5:]}")
    return rc


def summary(output):
    rows = []
    headers = []
    for path in sorted(output.rglob("*.csv")):
        if not path.name.lower().startswith(("op_summary", "task_time")):
            continue
        with path.open(encoding="utf-8-sig", errors="replace") as f:
            reader = csv.DictReader(f)
            headers.append((path.name, reader.fieldnames))
            n = 0
            examples = []
            for row in reader:
                n += 1
                names = [v for k, v in row.items() if k and v and ("name" in k.lower() or "type" in k.lower())]
                if len(examples) < 5:
                    examples.append(names)
                if not any("lifecycle_add_kernel" in v for v in names):
                    continue
                rows.append(
                    {k: v
                     for k, v in row.items()
                     if k and re.search("name|device|stream|task|duration", k, re.I)})
            print(f"PAIR table={path.name} total_rows={n} sample_names={examples}")
    print(f"PAIR msprof_tables={headers[:6]} matching_rows={rows[:12]}")
    return rows


def main(out):
    home = Path(os.environ["ASCEND_HOME_PATH"])
    lib = (home / "lib64/libmspti.so").resolve(strict=True)
    env = os.environ.copy()
    env.pop("LD_PRELOAD", None)
    env["TRITON_CACHE_DIR"] = str(out / "cache")
    env["PYTHONUNBUFFERED"] = "1"
    cmd = [sys.executable, str(HERE)]
    print(f"PAIR library={lib} sha256={hashlib.sha256(lib.read_bytes()).hexdigest()}")
    subprocess.run([
        "c++", "-std=c++17", "-shared", "-fPIC", "-pthread", "-I" + str(home / "tools/mspti/include"),
        "-I" + str(home / "include"),
        str(TEST / "proton_mspti_probe.cpp"),
        str(lib), "-o",
        str(out / "probe.so")
    ], check=True)
    assert execute([*cmd, "warm", str(out)], env, out / "warm.log") == 0
    output = out / "msprof"
    app = shlex.join([*cmd, "msprof", str(out)])
    execute(["msprof", "--task-time=on", f"--output={output}", f"--application={app}"], env, out / "msprof.log", 300)
    if not list(output.rglob("op_summary*.csv")):
        for prof in output.glob("PROF_*"):
            execute(["msprof", "--export=on", f"--output={prof}"], env, out / "export.log")
    matched = summary(output)
    for mode in ("stock", "shared"):
        plog = out / ("plog-" + mode)
        plog.mkdir()
        e = dict(env, LD_PRELOAD=str(lib), ASCEND_GLOBAL_LOG_LEVEL="0", ASCEND_SLOG_PRINT_TO_STDOUT="1",
                 ASCEND_PROCESS_LOG_PATH=str(plog))
        log = out / (mode + ".log")
        execute([*cmd, mode, str(out)], e, log)
        sites = Counter()
        for path in [log, *plog.rglob("*.log")]:
            with path.open(errors="replace") as f:
                for line in f:
                    m = re.search(
                        r"\[((?:channel_reader|channel_pool|kernel_parser|activity_manager|dev_prof_task)[\w.]*:\d+)\].*",
                        line, re.I)
                    if m:
                        sites[m[1]] += 1
                        if sites[m[1]] <= 3:
                            print(f"PAIR {mode} {m[0][:320]}")
        print(f"PAIR {mode} plog_sites={dict(sites)}")
    files = [out / (mode + ".device") for mode in ("msprof", "stock", "shared")]
    ids = [f.read_text() if f.is_file() else None for f in files]
    print(
        f"PAIR msprof/stock/shared verified physical IDs: {ids}, same_device={None not in ids and len(set(ids)) == 1}")
    if shutil.which("gdb"):
        e = dict(env, PAIR_MSPTI=str(lib))
        execute([
            "gdb", "-nx", "-q", "-batch", "-x",
            str(HERE.with_name("a3_mspti_stages.py")), "--args", *cmd, "stock",
            str(out)
        ], e, out / "stages.log", 150)
    else:
        print(f"PAIR internal debugger not run: gdb={shutil.which('gdb')} matched_rows={len(matched)}")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        workload(sys.argv[1], Path(sys.argv[2]))
    else:
        with tempfile.TemporaryDirectory() as work:
            main(Path(work))
