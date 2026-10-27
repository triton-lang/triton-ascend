"""Observe the upstream tests without replacing fixtures, kernels or assertions."""

import importlib
import json
import os
from pathlib import Path
import platform
import sys
import time


def emit(kind, **data):
    path = os.environ.get("ASCEND_COMMUNITY_EVENTS")
    if path:
        with open(path, "a", encoding="utf-8") as stream:
            stream.write(json.dumps({"event": kind, "time": time.time(), **data}, ensure_ascii=False) + "\n")


def initialize_npu():
    # torch_npu's explicit import can leave autoload disabled for child processes.
    name = "TORCH_DEVICE_BACKEND_AUTOLOAD"
    previous = os.environ.get(name)
    try:
        importlib.import_module("torch_npu")
    finally:
        if previous is None:
            os.environ.pop(name, None)
        else:
            os.environ[name] = previous
    torch = importlib.import_module("torch")
    torch.npu.set_device(0)  # ASCEND_RT_VISIBLE_DEVICES selects the physical device.
    triton = importlib.import_module("triton")
    target = triton.runtime.driver.active.get_current_target()
    if target.backend != "npu":
        raise RuntimeError(f"Expected the NPU backend, got {target!r}")
    emit("environment", python=sys.version, executable=sys.executable, machine=platform.machine(),
         torch_version=torch.__version__, triton_version=triton.__version__, triton_path=triton.__file__,
         target=repr(target))


def pytest_configure(config):
    if config.getoption("device", default=None) == "npu":
        initialize_npu()


def pytest_collection_modifyitems(items):
    nodes = [{
        "nodeid": item.nodeid, "parameters":
        {key: repr(value)
         for key, value in getattr(getattr(item, "callspec", None), "params", {}).items()}
    }
             for item in items]
    path = os.environ.get("ASCEND_COMMUNITY_NODES")
    if path:
        Path(path).write_text(json.dumps(nodes, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    emit("collection", nodes=[item.nodeid for item in items])


def pytest_collectreport(report):
    emit("collect_report", nodeid=report.nodeid, outcome=report.outcome, message=report.longreprtext)


def pytest_runtest_logstart(nodeid, location):
    emit("start", nodeid=nodeid)


def pytest_runtest_logreport(report):
    emit("report", nodeid=report.nodeid, when=report.when, outcome=report.outcome,
         wasxfail=getattr(report, "wasxfail", None), duration=report.duration, message=report.longreprtext)


def pytest_runtest_logfinish(nodeid, location):
    emit("finish", nodeid=nodeid)


def pytest_sessionfinish(session, exitstatus):
    emit("session_finish", exitstatus=int(exitstatus))
