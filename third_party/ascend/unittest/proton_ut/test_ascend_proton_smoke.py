import json
from pathlib import Path

import torch
import torch_npu  # noqa: F401 - registers the NPU backend with PyTorch
import triton
import triton.language as tl
import triton.profiler as proton


@triton.jit
def vector_add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    offsets = tl.program_id(axis=0) * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def walk_profile(nodes):
    for node in nodes:
        yield node
        yield from walk_profile(node.get("children", []))


def test_ascend_proton_smoke():
    n_elements = 4096
    iterations = 5
    block_size = 256
    grid = (triton.cdiv(n_elements, block_size), )

    x = torch.arange(n_elements, device="npu", dtype=torch.float32)
    y = torch.full((n_elements, ), 2.0, device="npu", dtype=torch.float32)
    output = torch.empty_like(x)

    # Compile before profiling so the profile contains execution rather than JIT setup.
    vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
    torch.npu.synchronize()

    profile_base = Path(__file__).with_name("ascend_proton_smoke")
    profile_path = profile_base.with_suffix(".hatchet")
    if profile_path.exists():
        profile_path.unlink()

    session = proton.start(str(profile_base), backend="mspti", hook="triton")
    try:
        with proton.scope("ascend_vector_add"):
            for _ in range(iterations):
                vector_add_kernel[grid](x, y, output, n_elements, BLOCK_SIZE=block_size)
        torch.npu.synchronize()
    finally:
        proton.finalize(session)

    torch.testing.assert_close(output.cpu(), x.cpu() + y.cpu())
    if not profile_path.exists():
        raise AssertionError(f"Proton did not create {profile_path}")

    with profile_path.open(encoding="utf-8") as profile_file:
        profile = json.load(profile_file)

    kernel_nodes = [node for node in walk_profile(profile) if node.get("frame", {}).get("name") == "vector_add_kernel"]
    if len(kernel_nodes) != 1:
        raise AssertionError(f"Unexpected kernel nodes: {kernel_nodes}")

    metrics = kernel_nodes[0].get("metrics", {})
    if metrics.get("count") != iterations:
        raise AssertionError(f"Unexpected kernel count: {metrics}")
    if metrics.get("time (ns)", 0) <= 0:
        raise AssertionError(f"Proton profile contains no kernel timing: {metrics}")

    device_id = str(metrics.get("device_id"))
    device_info = profile[1].get("NPU", {}).get(device_id)
    if device_info is None:
        raise AssertionError(f"Proton profile contains no NPU {device_id} information")

    expected_device_fields = {
        "arch",
        "bus_width",
        "clock_rate",
        "memory_clock_rate",
        "num_sms",
    }
    missing_fields = expected_device_fields.difference(device_info)
    if missing_fields:
        raise AssertionError(f"Missing device information: {sorted(missing_fields)}")
    if device_info["arch"] in ("", "ascend"):
        raise AssertionError(f"Unexpected NPU architecture: {device_info}")
    for field in ("clock_rate", "memory_clock_rate", "num_sms"):
        if device_info[field] <= 0:
            raise AssertionError(f"Invalid {field}: {device_info}")

    print(f"target={triton.runtime.driver.active.get_current_target()}")
    print(f"result=ok, profile={profile_path}")
    print(f"kernel=vector_add_kernel, count={metrics['count']}, "
          f"time_ns={metrics['time (ns)']}")
    print(f"device={device_id}, info={device_info}")


if __name__ == "__main__":
    test_ascend_proton_smoke()
