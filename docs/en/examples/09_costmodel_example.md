# Pruning Autotune Configurations with Costmodel

Costmodel predicts candidate performance before real benchmarking, so that only
the candidates predicted to be faster are compiled and measured. This reduces
the first-run autotune cost. The recommended interface is
`triton.autotune` with `prune_configs_by["costmodel"]`; users do not need to
generate TTIR manually.

## Recommended Usage

The vector-add example below uses Costmodel to prune four candidates to two,
then benchmarks the remaining candidates on the NPU to select the final
configuration.

```python
import torch
import torch_npu
import triton
import triton.language as tl
import triton.backends.ascend.runtime


@triton.autotune(
    configs=[
        triton.Config({"BLOCK_SIZE": 128, "multibuffer": False}),
        triton.Config({"BLOCK_SIZE": 256, "multibuffer": False}),
        triton.Config({"BLOCK_SIZE": 512, "multibuffer": False}),
        triton.Config({"BLOCK_SIZE": 1024, "multibuffer": False}),
    ],
    key=["n_elements"],
    prune_configs_by={
        "costmodel": {
            "top_k": 2,
        },
    },
)
@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements,
               BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


size = 98432
x = torch.rand(size, device="npu")
y = torch.rand(size, device="npu")
output = torch.empty_like(x)

grid = lambda meta: (triton.cdiv(size, meta["BLOCK_SIZE"]),)
add_kernel[grid](x, y, output, size)

torch.testing.assert_close(output, x + y)
print(add_kernel.best_config)
```

Importing `triton.backends.ascend.runtime` is required to enable the
Triton-Ascend autotune extension. If the launch grid depends on a tunable
parameter, use a lambda that receives `meta`, so every candidate is launched
with the correct grid.

### Options

Set `costmodel` to `True` to use the defaults, or set it to a dictionary:

| Option | Default | Description |
|---|---:|---|
| `top_k` | `0.25` | A positive integer keeps that many candidates; a float in `(0, 1]` keeps that fraction, rounded up. |
| `hardware_config` | Selected from the current target | Optional hardware configuration path; an explicit path overrides automatic selection. |

When `hardware_config` is not specified, Costmodel uses the Triton backend's
`target.arch`. This value comes from `TRITON_ASCEND_ARCH` when set, or from
runtime `rtGetSocVersion()` otherwise. Multiple SoC names map to each calibrated
model:

- `Ascend910B*` and `Ascend910_93*` (for example `Ascend910_9362`) use
  `ascend_910b.json`;
- `Ascend910_95*`, `Ascend950*`, and David-family targets use
  `ascend_davidv100.json`.

An unknown SoC is not silently evaluated with the 910B model. Costmodel pruning
fails open to real benchmarking instead. Explicit paths are normalized and the
JSON is validated before pruning.

For example, the following keeps the fastest predicted 25 percent:

```python
prune_configs_by={"costmodel": True}
```

### Pruning Order and Failure Fallback

Configurations are processed in this order:

```text
original candidates
  -> early_config_prune (if configured)
  -> costmodel or perf_model (choose one)
  -> real NPU benchmark
  -> final best_config
```

- `costmodel` and `perf_model` are mutually exclusive performance-prediction
  pruners. Configuring both raises `ValueError`.
- Failure to generate TTIR or predict one candidate does not affect the other
  candidates.
- If no candidate receives a prediction, pruning fails open and benchmarks the
  candidates from before Costmodel pruning.
- If every primary candidate selected by Costmodel fails to compile or measure,
  autotune continues with lower-ranked fallback candidates.
- Configurations with identical Costmodel-visible TTIR and runtime bindings form
  an equivalence group. The whole group is retained to avoid discarding backend
  differences that Costmodel cannot see, so the retained count may exceed
  `top_k`.
- Costmodel is only a prefilter. Real hardware benchmarking still determines
  the final configuration.

## Advanced Usage: Calling `costmodel_bench` Directly

Most users should use the autotune integration above. Call `costmodel_bench`
directly only when you need to manage TTIR, runtime argument bindings, and
candidate ranking yourself.

The following example demonstrates the low-level Costmodel flow:

- generate TTIR from a Triton frontend kernel;
- build `costmodel_bench` inputs for multiple candidate configs;
- call `costmodel_bench` and get the predicted latency for each config.

### Complete Example

Save the following code as `costmodel_example.py` and run it:

```python
from __future__ import annotations

import triton
import triton.language as tl
from triton.backends.ascend.runtime.costmodel_runtime import costmodel_bench
from triton.backends.compiler import GPUTarget
from triton.compiler import ASTSource
from triton.compiler.code_generator import ast_to_ttir
from triton.compiler.compiler import make_backend
from triton._C.libtriton import ir
from triton._C.libtriton.ascend import ir as ascend_ir


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


def make_ttir(kernel, signature, constants):
    source = ASTSource(kernel, signature, constants, attrs=None)
    target = GPUTarget("npu", "", 32)
    backend = make_backend(target)

    options = backend.parse_options(
        {
            "num_warps": 8,
            "num_stages": 2,
            "debug": False,
            "multibuffer": False,
            "compile_mode": "simd",
            "enable_costmodel_backend": True,
            **source.parse_options(),
        }
    )

    context = ir.context()
    ir.load_dialects(context)
    ascend_ir.load_dialects(context)
    return str(ast_to_ttir(kernel, source, context, options, {}, {}))


signature = {
    "x_ptr": "*fp32",
    "y_ptr": "*fp32",
    "output_ptr": "*fp32",
    "n_elements": "i32",
}
n_elements = 98432
configs = [
    {"name": "block256", "BLOCK_SIZE": 256},
    {"name": "block1024", "BLOCK_SIZE": 1024},
    {"name": "block2048", "BLOCK_SIZE": 2048},
]

items = []
for cfg in configs:
    ttir = make_ttir(add_kernel, signature, {"BLOCK_SIZE": cfg["BLOCK_SIZE"]})
    items.append(
        {
            "config": cfg["name"],
            "ttir": ttir,
            # n_elements is the fourth argument in the signature, so it maps
            # to %arg3 in TTIR. pid_x gives tl.program_id(0) a static value.
            "arg_bindings": f"arg3={n_elements},pid_x=0",
        }
    )

latencies = costmodel_bench(items)
for config, latency_us in sorted(latencies.items(), key=lambda item: item[1]):
    print(f"{config}: {latency_us:.3f} us")
```

### Example Output

The exact numbers may vary with costmodel parameters, but the output shape should look like this:

```text
block256: 0.098 us
block1024: 0.110 us
block2048: 0.126 us
```

`costmodel_bench` returns a dictionary whose keys are the `config` values passed in and whose values are predicted latencies in microseconds. An autotuning layer can sort by the returned values and keep the configs predicted to be faster.

### Key Points

1. `ASTSource + ast_to_ttir` only generates TTIR. It does not compile or launch the kernel.
2. `config` affects `tl.constexpr` values such as `BLOCK_SIZE`, so each candidate config needs its own TTIR.
3. Each item passed to `costmodel_bench` should contain at least `config` and `ttir`, and may also include `arg_bindings`.
4. `arg_bindings` binds runtime integer values to TTIR `%argN` arguments. In this example, `n_elements=98432` maps to `arg3=98432`.
5. If the kernel uses `tl.program_id(0)`, usually pass `pid_x=0`. If it also uses `tl.num_programs(0)`, pass `num_programs_x=...` as well.
6. Low-level callers may specify the SoC with `target_arch`; when omitted, the current Triton target is used when available. An explicit `hardware_config` path takes precedence.
