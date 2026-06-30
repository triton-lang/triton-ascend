# max_autotune 使用示例

`max_autotune` 是 Triton-Ascend 提供的增强自动调优装饰器，旨在简化多参数调优的代码编写。与社区版 `autotune` 要求用户手动枚举所有 `triton.Config` 不同，`max_autotune` 允许用户仅提供少量基础配置（如分块大小），并自动将相关的编译器选项（如 `num_stages`、`enable_hivm_auto_cv_balance` 等）纳入最优组合的搜索空间。用户也可以通过参数列表显式控制搜索范围。

**适用场景**：Ascend NPU 上的 `cube`、`mix`、`vector` 算子，尤其适合需要同时调整多个硬件相关参数的场景。

## 基本使用示例

以下示例演示了使用 `max_autotune` 对一个简单的向量加法 kernel 进行自动调优。与社区版 `autotune` 相比，`max_autotune` 还会自动将不同的编译器选项纳入调优空间，无需用户手动指定。

```Python
import torch
import torch_npu
import triton
import triton.language as tl
from triton.backends.ascend.runtime import max_autotune

def test_max_autotune():

    # 基础配置：只需提供分块大小，其他调优参数由装饰器自动生成
    base_configs = [
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
    ]

    @max_autotune(
        configs=base_configs,
        key=["numel"],
        kernel_type="vector",          # 算子类型：cube / mix / vector, 默认为mix
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        BLOCK_SIZE: tl.constexpr
    ):
        pid = tl.program_id(0)
        idx = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = idx < numel

        # 模拟计算负载
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=mask, other=0.0)
            tmp1 = tl.load(in_ptr1 + idx, mask=mask, other=0.0)
            tmp2 = tl.math.exp(tmp0) + tmp1 + i
            tl.store(out_ptr0 + idx, tmp2, mask=mask)

    # 封装调用函数
    def triton_calc_func(x0, x1):
        n = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]),)
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0

    # 与 PyTorch 参考结果对比
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000 - 1

    DEV = "npu"
    DTYPE = torch.float32
    N = 192 * 1024
    x0 = torch.randn((N,), dtype=DTYPE, device=DEV)
    x1 = torch.randn((N,), dtype=DTYPE, device=DEV)
    torch_ref = torch_calc_func(x0, x1)
    triton_cal = triton_calc_func(x0, x1)
    torch.testing.assert_close(triton_cal, torch_ref)

if __name__ == "__main__":
    test_max_autotune()
    print("success: test_max_autotune")
```

## 进阶使用：精确控制调优参数

用户可以通过 **tuning_params 显式指定需要调优的编译器选项及其取值列表；未指定的参数会使用内置默认值。以下示例展示了如何对多个参数进行组合搜索。

```python
from triton.backends.ascend.runtime import max_autotune

def test_max_autotune():

    # 基础配置：只需提供分块大小，其他调优参数由装饰器自动生成
    base_configs = [
        triton.Config({'BLOCK_SIZE': 128}),
        triton.Config({'BLOCK_SIZE': 256}),
    ]

    @max_autotune(
        configs=base_configs,          # 基础配置列表
        key=["numel"],                 # 当 numel 变化时触发重新调优
        kernel_type="vector",          # 算子类型：cube / mix / vector
        # 以下参数为可选的调优列表，不提供时使用内置默认值
        num_stages=[1, 2],
        enable_ubuf_saving=[True, False]
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        BLOCK_SIZE: tl.constexpr,
        **META
    ):
        pass
```
