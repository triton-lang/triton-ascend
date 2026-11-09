# Warmup使用指南

## 概述

`kernel.warmup(*args, grid=...)` 只编译 kernel，不执行。编译产物写入 Triton 缓存，后续真实 launch 可以直接命中，因此适合在服务启动或 CI 阶段提前承担编译开销。

以下场景推荐使用：

- **降低首次推理时延**：真实请求不再等待编译，冷启动无卡顿；
- **提前暴露编译错误**：编译问题在启动或 CI 阶段即可发现，避免推迟到运行时；
- **只验证编译、不执行**：可不分配输出、不实际运行，仅确认 kernel 能否编译。

## 普通 JIT kernel使用

以一个向量加法 kernel 为例：

```python
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


n = 1024
grid = (triton.cdiv(n, 256),)
```

### 场景一：用 dtype 占位

`warmup()` 只依赖参数类型和指针属性，不读取张量内容，因此无需分配和初始化设备内存，给出 dtype 即可；在设备内存紧张，或启动脚本、CI 中尚未准备输入数据时尤其方便：

```python
compiled = add_kernel.warmup(
    torch.float32, torch.float32, torch.float32, n,
    BLOCK_SIZE=256, grid=grid,
)
print(compiled.metadata)
```

调用时有两个要点：

- `grid` 必须显式传入；
- `warmup()` 返回 `CompiledKernel`，可以读取 `metadata`、`asm` 等编译结果。

### 场景二：传真实张量

占位符按“指针 16 字节对齐”的普通张量处理。如果真实输入是切片、偏移视图等可能未对齐的指针，应传真实张量，否则编译特化可能不同：

```python
x = torch.randn(n, device="npu", dtype=torch.float32)
y = torch.randn(n, device="npu", dtype=torch.float32)
out = torch.empty_like(x)
compiled = add_kernel.warmup(x, y, out, n, BLOCK_SIZE=256, grid=grid)
```

## Autotune kernel使用

如果 kernel 使用了 `@triton.autotune`，warmup 会编译每个候选配置，写法与普通 kernel 相同：

```python
import torch
import torch_npu
import triton
import triton.language as tl


configs = [
    triton.Config({"BLOCK_SIZE": 256}, num_warps=4),
    triton.Config({"BLOCK_SIZE": 512}, num_warps=8),
]


@triton.autotune(configs=configs, key=["n_elements"])
@triton.jit
def add_kernel_autotune(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, x + y, mask=mask)


n = 1024
grid = (1,)

handles = add_kernel_autotune.warmup(
    torch.float32, torch.float32, torch.float32, n, grid=grid,
)
kernels = [h.result() if hasattr(h, "result") else h for h in handles]
```

与普通 kernel 相比，有三点不同：

- 返回的是列表，每个候选配置对应一项；Ascend 后端默认并行编译，元素可能是 `FutureKernel`，用 `.result()` 取到 `CompiledKernel` 后即可统一处理；
- `configs=[]` 的自动 Tiling 同样支持；并行编译的异常会在 `warmup()` 返回前抛出；
- warmup **不做 benchmark、不选最优配置**，选优仍发生在第一次真实 launch，不要把 warmup 当作“选优预热”。

## 注意事项

- warmup 不会读写输入输出，也不会发现越界、数值错误、UB 溢出等运行时问题；
- warmup 的收益依赖缓存命中：真实 launch 应与 warmup 使用相同的 Triton 缓存目录（`TRITON_CACHE_DIR`）和编译选项；
- 可以开启 `TRITON_DEBUG=1` 确认是否命中：真实 launch 命中缓存后，不会再出现该 kernel 的编译日志。
