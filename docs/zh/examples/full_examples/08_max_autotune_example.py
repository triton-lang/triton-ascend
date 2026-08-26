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
        kernel_type="vector",  # 算子类型：cube / mix / vector, 默认为mix
        num_stages=[1, 2],
    )
    @triton.jit
    def triton_calc_kernel(out_ptr0, in_ptr0, in_ptr1, numel, BLOCK_SIZE: tl.constexpr):
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
        grid = lambda meta: (triton.cdiv(n, meta["BLOCK_SIZE"]), )
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0

    # 与 PyTorch 参考结果对比
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000 - 1

    DEV = "npu"
    DTYPE = torch.float32
    N = 192 * 1024
    x0 = torch.randn((N, ), dtype=DTYPE, device=DEV)
    x1 = torch.randn((N, ), dtype=DTYPE, device=DEV)
    torch_ref = torch_calc_func(x0, x1)
    triton_cal = triton_calc_func(x0, x1)
    torch.testing.assert_close(triton_cal, torch_ref)


if __name__ == "__main__":
    test_max_autotune()
    print("success: test_max_autotune")
