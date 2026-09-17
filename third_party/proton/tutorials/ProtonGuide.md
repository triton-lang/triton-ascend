# Proton Profiling Guide for Ascend NPU (via Triton-Ascend)

> `README.md` style documentation for using **Proton** (`triton.profiler`) on Huawei Ascend NPU.

---

## Overview

**Proton** is Triton's lightweight, hierarchical profiler. It supports call-path profiling, custom software metrics (FLOPs, bytes, etc.), and scopes.

On **Ascend NPU** (using `triton-ascend` + `torch_npu`):

- Only **`context="shadow"`** works reliably.
- **`context="python"`** may work but is less useful on NPU.
- **`mode="npu"`** or **`mode="ascend"`** are **not supported** (no CUPTI/ROCtracer equivalent).
- **Time metrics** (`time`, `duration`, auto-calculated TFLOPS/GB/s) are **not automatically captured** in the output `.hatchet` file on Ascend (only your custom metrics appear).

---

### Installation

```bash
# Install Triton-Ascend (Proton is built-in)
pip install triton-ascend

# Make sure torch_npu is installed
pip install torch-npu
```

> Proton build is enabled by default in recent `triton-ascend` wheels. If building from source, avoid `TRITON_BUILD_PROTON=OFF`.

---

### Supported Commands / API

#### 1. `proton.start()` – Start a profiling session

```python
import triton.profiler as proton

# Recommended for Ascend
session_id = proton.start(
    "ascend_proton_fat_test"      # profile name (becomes prefix of output file)
)
```

#### 2. `proton.activate()` / `proton.deactivate()` – Pause/resume profiling

Useful when you want to skip some regions but keep the same session:

```python
proton.deactivate(session_id)   # kernels still execute, but not profiled
# ... code you don't want profiled ...
proton.activate(session_id)     # resume profiling
```

#### 3. `proton.scope()` – Define a profiled region with custom metrics (most important!)

```python
   with proton.scope(f"vector_add [N={N:,} x {ITERATIONS} iters]", {
        "flops": float(N) * ITERATIONS,
        "bytes_loaded": 2 * N * x.element_size() * ITERATIONS,
        "bytes_stored": 1 * N * x.element_size() * ITERATIONS,
        "total_bytes": 3 * N * x.element_size() * ITERATIONS,
        "block_size": BLOCK_SIZE,
        "duration_ms": elapsed_ms,
        "tflops": tflops,
        "gbps": gbps
    }):

    # Your kernel launches here
    for _ in range(ITERATIONS):
        vector_add_kernel[grid](x, y, out, N, BLOCK_SIZE)
```

#### 4. `proton.finalize()` – Write profile and shutdown

```python
proton.finalize()   # writes proton.hatchet (or <name>.hatchet)
```

#### 5. Other useful helpers

```python
# Decorator version
@proton.profile(name="my_func", context="shadow")
def my_function(...):
    ...

# CPU-timed scope (adds cpu_time metric)
with proton.cpu_timed_scope("my_cpu_region"):
    ...
```

---

### Complete Working Example (Ascend NPU)

```python
import torch
import torch_npu
import triton
import triton.language as tl
import triton.profiler as proton

@triton.jit
def vector_add_kernel(
    x_ptr, y_ptr, out_ptr,
    n: tl.constexpr,
    BLOCK_SIZE: tl.constexpr
):
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(out_ptr + offsets, x + y, mask=mask)


def run_proton_test():
    proton.start("ascend_proton_fat_test")
    N = 64 * 1024 * 1024
    ITERATIONS = 200
    BLOCK_SIZE = 2048
    x = torch.ones(N, device="npu", dtype=torch.float32)
    y = torch.ones(N, device="npu", dtype=torch.float32)
    out = torch.empty_like(x)
    grid = (int(triton.cdiv(N, BLOCK_SIZE)),)
    for _ in range(20):
        vector_add_kernel[grid](x, y, out, N, BLOCK_SIZE)
        torch.npu.synchronize()
        start_event = torch.npu.Event(enable_timing=True)
        end_event = torch.npu.Event(enable_timing=True)
        start_event.record()
        for _ in range(ITERATIONS):
            vector_add_kernel[grid](x, y, out, N, BLOCK_SIZE)
        end_event.record()
        torch.npu.synchronize()
        elapsed_ms = start_event.elapsed_time(end_event)
        total_flops = float(N) * ITERATIONS
        total_bytes_gb = 3 * N * x.element_size() * ITERATIONS / 1e9
        tflops = (total_flops / elapsed_ms / 1e6) if elapsed_ms > 0 else 0
        gbps = (total_bytes_gb / (elapsed_ms / 1000)) if elapsed_ms > 0 else 0
        print(f"Measured: {elapsed_ms:.3f} ms | {tflops:.2f} TFLOPS | {gbps:.2f} GB/s")
        with proton.scope(f"vector_add [N={N:,} x {ITERATIONS} iters]", {
            "flops": float(N) * ITERATIONS,
            "bytes_loaded": 2 * N * x.element_size() * ITERATIONS,
            "bytes_stored": 1 * N * x.element_size() * ITERATIONS,
            "total_bytes": 3 * N * x.element_size() * ITERATIONS,
            "block_size": BLOCK_SIZE,
            "duration_ms": elapsed_ms,
            "tflops": tflops,
            "gbps": gbps
        }):
            pass
    proton.finalize()
    print("Proton test is over!")


if __name__ == "__main__":
    run_proton_test()
```

---

### Output File (`proton.hatchet`)

After `finalize()`, you get a JSON file (Hatchet format):

```json
[
  {
     "children": [
      {
        "children": [],
        "frame": { "name": "vector_add [N=67,108,864 x 200 iters]", "type": "function" },
                 "metrics": {
                    "block_size": 40960,
                    "bytes_loaded": 2147483648000,
                    "bytes_stored": 1073741824000,
                    "duration_ms": 5561.438415527344,
                    "flops": 268435456000.0,
                    "gbps": 11584.155901710412,
                    "tflops": 965.3463251425343,
                    "total_bytes": 3221225472000
                }
        }
    ],
    "frame": { "name": "ROOT", "type": "function" },
    "metrics": { ... }
  }
]
```

**Important limitation on Ascend**:

- No automatic `time` / `duration` metrics.
- Only the metrics you pass to `proton.scope(...)` appear.
- Use manual `torch.npu.Event` (as in the example) for wall-time measurement.

---

### Viewing the Profile

```bash
# Built-in viewer (shows tree + metrics)
proton-viewer proton.hatchet

# Or with specific metrics
proton-viewer -m flops,bytes_loaded,total_bytes proton.hatchet

# Print sorted kernel list
proton-viewer --print-sorted proton.hatchet
```

You can also load the `.hatchet` file with **Hatchet** (Python library) for custom analysis.

---

### Command-Line Usage

```bash
# Run entire script under Proton (shadow context by default)
python -m proton your_script.py
```

---

### Best Practices for Ascend

1. Always use `context="shadow"`.
2. Wrap **only** the regions you care about with `proton.scope()`.
3. Provide meaningful custom metrics (FLOPs, bytes, etc.) — these are the only numbers you’ll see.
4. Use `torch.npu.Event` or `torch_npu.profiler` for real timing.
5. Call `torch.npu.synchronize()` before/after scopes when needed.
6. Deactivate/activate when profiling only parts of a large loop.

---

### Known Limitations on Ascend (as of 2026)

- No hardware counter / instruction sampling support.
- No automatic time collection in Proton (unlike NVIDIA/AMD).
- Only custom metrics from `scope()` are recorded.

For full NPU profiling (operators, CANN stack, AI Core metrics) use Huawei tools:

- `msprof`
- `torch_npu.profiler.profile`

---

**Happy profiling on Ascend!**  
If you have more examples or discover new features in newer `triton-ascend` releases, feel free to update this guide.
