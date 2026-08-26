import torch, torch_npu
import triton
import triton.language as tl


def test_triton_autotune():

    # 返回一组不同的 kernel 配置，用于 autotune 测试
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': False}),
            triton.Config({'XS': 8 * 1024, 'multibuffer': True}),
        ]

    @triton.autotune(configs=get_autotune_config(),  # 配置列表
                     key=["numel"],  # 当numel大小发生变化时会触发autotune
                     )
    @triton.jit
    def triton_calc_kernel(out_ptr0, in_ptr0, in_ptr1, numel, XS: tl.constexpr  # 块大小，用于控制每个线程块处理多少数据
                           ):
        pid = tl.program_id(0)  # 获取当前 program 的 ID
        idx = pid * XS + tl.arange(0, XS)  # 当前线程块处理的 index 范围
        msk = idx < numel  # 避免越界的掩码

        # 重复执行一些计算以模拟负载（并测试性能）/ Repeat computation to simulate load (for perf test)
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=msk, other=0.0)  # 加载 x0
            tmp1 = tl.load(in_ptr1 + idx, mask=msk, other=0.0)  # 加载 x1
            tmp2 = tl.math.exp(tmp0) + tmp1 + i  # 计算
            tl.store(out_ptr0 + idx, tmp2, mask=msk)  # 存储到输出

    # Triton 调用函数，自动使用 autotuned kernel
    def triton_calc_func(x0, x1):
        n = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(n, meta["XS"]), 1, 1)  # 计算 grid 大小
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0

    # 使用 PyTorch 作为参考实现进行对比
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000 - 1

    DEV = "npu"  # 使用 NPU 作为设备
    DTYPE = torch.float32
    N = 192 * 1024  # 输入长度
    x0 = torch.randn((N, ), dtype=DTYPE, device=DEV)  # 随机输入 x0
    x1 = torch.randn((N, ), dtype=DTYPE, device=DEV)  # 随机输入 x1
    torch_ref = torch_calc_func(x0, x1)  # 得到参考结果
    triton_cal = triton_calc_func(x0, x1)  # 运行 Triton kernel
    torch.testing.assert_close(triton_cal, torch_ref)  # 验证输出是否一致


if __name__ == "__main__":
    test_triton_autotune()
    print("success: test_triton_autotune")  # 输出成功标志 / Print success message
