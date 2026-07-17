# End-to-End Costmodel Example

This example shows the basic costmodel backend flow:

- generate TTIR from a Triton frontend kernel;
- build `costmodel_bench` inputs for multiple candidate configs;
- call `costmodel_bench` and get the predicted latency for each config.

This flow is useful for filtering out slow configs before autotuning. The example uses only a vector add kernel so that the focus stays on the costmodel input and return value.

## Complete Example

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

## Example Output

The exact numbers may vary with costmodel parameters, but the output shape should look like this:

```text
block256: 0.098 us
block1024: 0.110 us
block2048: 0.126 us
```

`costmodel_bench` returns a dictionary whose keys are the `config` values passed in and whose values are predicted latencies in microseconds. An autotuning layer can sort by the returned values and keep the configs predicted to be faster.

## Key Points

1. `ASTSource + ast_to_ttir` only generates TTIR. It does not compile or launch the kernel.
2. `config` affects `tl.constexpr` values such as `BLOCK_SIZE`, so each candidate config needs its own TTIR.
3. Each item passed to `costmodel_bench` should contain at least `config` and `ttir`, and may also include `arg_bindings`.
4. `arg_bindings` binds runtime integer values to TTIR `%argN` arguments. In this example, `n_elements=98432` maps to `arg3=98432`.
5. If the kernel uses `tl.program_id(0)`, usually pass `pid_x=0`. If it also uses `tl.num_programs(0)`, pass `num_programs_x=...` as well.
