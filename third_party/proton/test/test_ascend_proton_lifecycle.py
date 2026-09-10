import json
import threading

import pytest
import torch
import torch_npu  # noqa: F401 - registers the NPU backend with PyTorch
import triton
import triton.language as tl
import triton.profiler as proton


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


def test_deactivate_reactivate(tmp_path, npu_tensors):
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


def test_multiple_sessions(tmp_path, npu_tensors):
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


def test_restart_after_finalize(tmp_path, npu_tensors):
    x, y = npu_tensors
    for index, expected in enumerate((1, 2)):
        path = tmp_path / f"restart{index}.hatchet"
        session = proton.start(str(path.with_suffix("")), hook="triton")
        try:
            launch(x, y, expected)
        finally:
            proton.finalize(session)
        assert kernel_count(path) == expected


def test_multiple_streams(tmp_path, npu_tensors):
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


def test_multiple_threads(tmp_path, npu_tensors):
    x, _ = npu_tensors
    path = tmp_path / "threads.hatchet"
    session = proton.start(str(path.with_suffix("")), backend="mspti", hook="triton")
    errors = []

    def worker():
        try:
            torch.npu.set_device(0)
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


def test_chrome_trace(tmp_path, npu_tensors):
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
