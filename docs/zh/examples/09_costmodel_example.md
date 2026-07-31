# Costmodel 端到端示例

本示例展示 costmodel 后端的基本调用流程：

- 使用 Triton 前端算子生成 TTIR；
- 为多个候选 config 构造 `costmodel_bench` 输入；
- 调用 `costmodel_bench` 得到每个 config 的预测耗时。

这个流程适合在 autotune 前快速筛掉预计性能较差的 config。示例只使用向量加法 kernel，便于聚焦 costmodel 的输入和返回值。

## 完整示例

将下面的代码保存为 `costmodel_example.py` 后运行：

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
            # n_elements 是 signature 中的第 4 个参数，对应 TTIR 里的 %arg3。
            # pid_x 给 tl.program_id(0) 一个静态估算值。
            "arg_bindings": f"arg3={n_elements},pid_x=0",
        }
    )

latencies = costmodel_bench(items)
for config, latency_us in sorted(latencies.items(), key=lambda item: item[1]):
    print(f"{config}: {latency_us:.3f} us")
```

## 示例输出

不同版本的 costmodel 参数可能会使具体数值略有不同，但输出结构类似：

```text
block256: 0.098 us
block1024: 0.110 us
block2048: 0.126 us
```

`costmodel_bench` 的返回值是一个字典，key 为传入的 `config`，value 为预测耗时，单位是微秒。上层 autotune 逻辑可以按 value 排序，优先保留预测更快的 config。

## 关键点说明

1. `ASTSource + ast_to_ttir` 只生成 TTIR，不会真实编译或启动 kernel。
2. `config` 会影响 `tl.constexpr`，例如 `BLOCK_SIZE`，因此每个候选 config 都需要生成各自的 TTIR。
3. `costmodel_bench` 接收的每个元素至少包含 `config` 和 `ttir`，也可以附带 `arg_bindings`。
4. `arg_bindings` 用于把运行时整数参数绑定到 TTIR 中的 `%argN`。例如本例中 `n_elements=98432` 对应 `arg3=98432`。
5. 如果 kernel 中使用 `tl.program_id(0)`，通常需要传入 `pid_x=0`。如果还使用 `tl.num_programs(0)`，可额外传入 `num_programs_x=...`。
