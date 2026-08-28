# 使用 Costmodel 剪枝 Autotune 配置

Costmodel 可以在真实 benchmark 前预测候选配置的性能，仅让预测较快的配置进入实测，从而降低首次 autotune 的编译和测量成本。推荐通过 `triton.autotune` 的 `prune_configs_by["costmodel"]` 启用，不需要用户手工生成 TTIR。

## 推荐用法

下面的向量加示例先用 Costmodel 将 4 个候选配置裁剪为 2 个，再在 NPU 上 benchmark 剩余配置并选择最终配置。

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

必须导入 `triton.backends.ascend.runtime` 才会启用 Triton-Ascend 的 autotune 扩展。grid 依赖被调优参数时，应使用接收 `meta` 的 lambda，使每个候选配置使用正确的发射网格。

### 配置项

`costmodel` 可以设置为 `True`，此时使用默认选项；也可以设置为字典：

| 配置项 | 默认值 | 说明 |
|---|---:|---|
| `top_k` | `0.25` | 正整数表示保留数量；`(0, 1]` 的浮点数表示按候选总数保留的比例，比例计算向上取整。 |
| `hardware_config` | 根据当前 target 自动选择 | 可选的硬件配置文件路径；显式指定时覆盖自动选择结果。 |

未显式设置 `hardware_config` 时，Costmodel 使用 Triton backend 的 `target.arch`。该字段优先来自 `TRITON_ASCEND_ARCH`，否则由运行时 `rtGetSocVersion()` 获取。多个 SoC 名称会映射到同一个已校准模型：

- `Ascend910B*`、`Ascend910_93*`（例如 `Ascend910_9362`）使用 `ascend_910b.json`；
- `Ascend910_95*`、`Ascend950*`、David 系列使用 `ascend_davidv100.json`。

未知 SoC 不会静默套用 910B 模型，而是跳过 Costmodel 剪枝并回退到真实 benchmark。显式配置路径会在剪枝前完成路径规范化和 JSON 校验。

例如，下面的写法会保留预测排名前 25% 的候选配置：

```python
prune_configs_by={"costmodel": True}
```

### 剪枝顺序和失败回退

配置处理顺序为：

```text
原始候选配置
  -> early_config_prune（如果配置）
  -> costmodel 或 perf_model（二选一）
  -> 真实 NPU benchmark
  -> 最终 best_config
```

- `costmodel` 和 `perf_model` 是互斥的性能预测剪枝器，同时配置会抛出 `ValueError`。
- 单个配置无法生成 TTIR 或预测失败时，不影响其他配置。
- 所有配置都无法预测时会 fail-open，继续 benchmark 剪枝前的配置。
- Costmodel 选出的 primary 配置全部编译或测量失败时，会继续尝试预测排名靠后的 fallback 配置。
- 对 Costmodel 来说 TTIR 和运行时绑定完全相同的配置属于等价组。为避免误删 Costmodel 看不到的后端差异，等价组会整体保留，因此实际保留数量可能超过 `top_k`。
- Costmodel 只负责预筛选，最终配置仍由真实硬件 benchmark 决定。

## 高级用法：直接调用 `costmodel_bench`

一般用户应优先使用上面的 autotune 集成接口。只有在需要自行管理 TTIR、运行时参数绑定和候选排序时，才直接调用 `costmodel_bench`。

下面展示底层 Costmodel 调用流程：

- 使用 Triton 前端算子生成 TTIR；
- 为多个候选 config 构造 `costmodel_bench` 输入；
- 调用 `costmodel_bench` 得到每个 config 的预测耗时。

### 完整示例

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

### 示例输出

不同版本的 costmodel 参数可能会使具体数值略有不同，但输出结构类似：

```text
block256: 0.098 us
block1024: 0.110 us
block2048: 0.126 us
```

`costmodel_bench` 的返回值是一个字典，key 为传入的 `config`，value 为预测耗时，单位是微秒。上层 autotune 逻辑可以按 value 排序，优先保留预测更快的 config。

### 关键点说明

1. `ASTSource + ast_to_ttir` 只生成 TTIR，不会真实编译或启动 kernel。
2. `config` 会影响 `tl.constexpr`，例如 `BLOCK_SIZE`，因此每个候选 config 都需要生成各自的 TTIR。
3. `costmodel_bench` 接收的每个元素至少包含 `config` 和 `ttir`，也可以附带 `arg_bindings`。
4. `arg_bindings` 用于把运行时整数参数绑定到 TTIR 中的 `%argN`。例如本例中 `n_elements=98432` 对应 `arg3=98432`。
5. 如果 kernel 中使用 `tl.program_id(0)`，通常需要传入 `pid_x=0`。如果还使用 `tl.num_programs(0)`，可额外传入 `num_programs_x=...`。
6. 底层调用可以通过 `target_arch` 指定 SoC；未指定时会尝试从当前 Triton target 获取。`hardware_config` 显式路径的优先级更高。
