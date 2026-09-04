# Triton-Ascend `do_bench_npu` Guide

## Positioning

This guide is intended for users who already know how to write Triton kernels and already understand the basic concepts of community `triton.testing.do_bench`. It focuses on the recommended Triton-Ascend usage for performance benchmarking on NPU:

- the recommended `do_bench_npu` usage on Triton-Ascend;
- what "pure Device-side execution time" means on the Ascend backend;
- when to use the default `mspti` path, and when you should fall back to the `torch_npu.profiler` path.

## Recommended Usage

On Triton-Ascend, the recommended usage is to import and use `do_bench_npu` when you need to build an operator performance baseline or measure the pure device time of a kernel:

```python
import torch
import torch_npu
from triton.backends.ascend.testing import do_bench_npu

@triton.jit
def kernel(
    x_ptr,
    y_ptr,
    out_ptr
):

def fn():
    kernel[grid](x, y, out)

ms = do_bench_npu(fn)
print(f"Kernel execution time: {ms} ms")
```

This means:

- `fn` has the same semantics as in community `do_bench` and represents the callable function to be benchmarked;
- `do_bench_npu` on Triton-Ascend means “let the NPU profiling tools capture pure Device-side kernel execution time”, effectively eliminating Host-side overhead from the measurement.

## Prerequisites

### 1. Understand the difference between Host-side and Device-side timing

The community `triton.testing.do_bench` uses `Event` to measure, and the recorded start and end times will include Host-side launch overhead.

`do_bench_npu` leverages NPU profiling tools (`MSPTI` or `torch_npu.profiler`), guaranteeing that the returned time is strictly the Device-side kernel execution times.

### 2. Environment requirements for the fast path (`mspti`)

To achieve a runtime speed comparable to community `do_bench`, `do_bench_npu` attempts to use the lightweight `mspti` library by default. To enable this:

- The `mspti` package must be importable.
- If the CANN version is earlier than 9.1.0, `libmspti.so` must be set in the `LD_PRELOAD` environment variable. If not, the system will raise a `RuntimeError` guiding you to set it.
- If `mspti` is unavailable, `do_bench_npu` will print a warning and automatically fall back to the `torch_npu.profiler` path, which is slower but functionally complete.

## Practical Notes

### 1. Passing a single function vs. multiple functions

You can pass either a single callable or a list of callables:

```python
# Single function: returns a float
time_ms = do_bench_npu(fn_A)

# Multiple functions: returns a list of floats
times_ms = do_bench_npu([fn_A, fn_B, fn_C])
```

Constraint: When passing a list of functions, `do_bench_npu` can accurately return the per-function Device-side time only if each function contains exactly one kernel. If a function contains multiple kernels, `do_bench_npu` can not correctly attribute which kernel belongs to which function. Avoid testing lists of complex functions.

### 2. Clearing the L2 Cache

To measure the worst-case performance or simulate cold-start memory scenarios, you can clear the L2 cache before every iteration:

```python
ms = do_bench_npu(fn, clear_l2_cache=True)
```

When enabled, `do_bench_npu` will execute a dummy buffer read operation before each fn call. `do_bench_npu` will automatically filter out the time consumed by this cache-clearing operation.

### 3. Benchmarking a specific kernel by name

If your `fn` contains multiple kernels, but you only want to measure the execution time of a specific one, use the `target_kernel_name` parameter:

```python
ms = do_bench_npu(fn, target_kernel_name="triton_matmul_kernel")
```

When `target_kernel_name` is specified, do_bench_npu forces the fallback to the `torch_npu.profiler` path. The profiler will parse the detailed CSV and strictly filter for the exact kernel name.
If the number of captured kernels does not match the expected `(warmup + active)` iterations, it will raise a `ProfilerResultMismatchError` to ensure data integrity.

### 4. Keeping raw profiler results

If you are debugging a performance anomaly and need to inspect the raw profiler output files (such as `task_time*.csv` or `kernel_details.csv`), use:

```python
ms = do_bench_npu(fn, prof_dir="./my_prof_results", keep_res=True)
```

Setting either `prof_dir` or `keep_res=True` will force the `torch_npu.profiler` path, as `mspti` does not write to these CSV files.

### 5. Set warmup and active times

Can autonomously input preheating and actual running times:

```python
ms = do_bench_npu(fn, warmup=1, active=1)
```

`warmup` represents the number of times fn is run before actual timing, used for preheating. `active` represents the actual number of runs of fn, and the final result is the average of these active iterations.

## Scope and Behavior of the Timing Collection

The core extensions of `do_bench_npu` over community `do_bench` are focused on accuracy and hardware affinity.

### 1. Pure Device-side time collection

The collection mechanism focuses on on-chip execution time. This capability does not record Host-side overhead or launch latency. It guarantees that the returned time is strictly the sum of the Device-side kernel execution times.

### 2. Automatic filtering of cache-clearing operations

When `clear_l2_cache=True` is used, the profiler will automatically recognize and filter out the dummy kernels (e.g., operations containing “ReduceSum” or “zero”) used for cache clearing, ensuring they do not pollute the final timing calculation.

### 3. Time collection with multiple kernels in a single function

In real NPU execution, a single Triton kernel launch might be fragmented into multiple hardware tasks by the underlying scheduler. `do_bench_npu` automatically detects if the actual number of captured rows (tasks) is a multiple of the expected iterations. If so, it calculates the multiplier and averages the time correctly, returning the accurate total time for the logical function.

## Advanced Usage: Integrating with Triton Autotune

Triton-Ascend’s Autotune extension supports switching the performance-collection method used during autotune benchmark. By setting the environment variable `TRITON_BENCH_METHOD="npu"`, the Autotune mechanism will internally call `do_bench_npu` to evaluate candidate configurations.

```python
export TRITON_BENCH_METHOD="npu"
```

This is highly recommended for short-running kernels where Host-side launch latency would otherwise dominate the community `do_bench` measurement, leading to the selection of configuration is not optimal on the device side.

## Summary

The key extension of `do_bench_npu` over community `do_bench` is the transition from event-based timing to profiler-based timing. For most users, the recommended usage is:

- call `do_bench_npu(fn)` to get the pure Device-side execution time;
- use `target_kernel_name` if you need to isolate a specific kernel in a complex function;
- enable `clear_l2_cache=True` if you need stable worst-case performance baselines;
- ensure `mspti` is installed to maintain runtime parity with community `do_bench`.
