# 精度比对 （Accuracy Comparison）

在本节中，我们将使用 Triton 编写一个简单的精度比对的程序。
在此过程中，用户会学习到：

- Triton 每种数据类型的精度比对方法。
- 参考示例代码：triton-ascend/ascend/examples/tutorials/14-accuracy-comparison.py

计算内核:

```Python
def test_add(x0, x1):
    """
    测试 Triton 实现的向量加法与 PyTorch 的结果,精度比对是否一致。
    
    步骤：
    1. 使用 PyTorch 计算参考结果（torch_ref）
    2. 使用 Triton 编写 kernel 并计算结果（triton_cal）
    3. 调用 accuracy_comparison 进行精度比对
    """

    # 1. 使用 PyTorch 作为参考实现（golden truth）
    def torch_func(x0, x1):
        res = x0 + x1
        return res

    # 2. 定义 Triton kernel（在 NPU/GPU 上执行）
    @triton.jit
    def triton_kernel_add(
        out_ptr0,   # 输出指针：结果存储位置
        in_ptr0,    # 输入指针0：x0 的起始地址
        in_ptr1,    # 输入指针1：x1 的起始地址
        XS: tl.constexpr  # constexpr 参数：向量长度，在编译时确定
    ):
        # 生成 [0, 1, 2, ..., XS-1] 的索引数组
        idx = tl.arange(0, XS)
        # 从 in_ptr0 + idx 处加载 x0 的值
        tmp0 = tl.load(in_ptr0 + idx)
        # 从 in_ptr1 + idx 处加载 x1 的值
        tmp1 = tl.load(in_ptr1 + idx)
        # 执行加法
        tmp2 = tmp0 + tmp1
        # 将结果写入 out_ptr0 + idx
        tl.store(out_ptr0 + idx, tmp2)

    # 3. Triton 封装函数：调用 kernel 并返回结果
    def triton_func(x0, x1):
        y0 = torch.empty_like(x0)  # 创建与输入形状、dtype 相同的输出张量
        # 启动 kernel：grid = [1, 1, 1] 表示仅使用一个 block
        # 注意：XS 必须作为参数传入，因为它是 tl.constexpr 类型
        triton_kernel_add[1, 1, 1](y0, x0, x1, XS=x0.numel())
        return y0

    # 4. 获取参考结果和 Triton 计算结果
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1)

    # 5. 精度比对
    accuracy_comparison(triton_cal, torch_ref)

    # 6. 打印成功信息
    print(f"== dtype:{triton_cal.dtype} == The accuracy comparison between triton_cal and torch_ref was successful.")


```

创建一个精度比对函数，适应每一种dtype，采用对应的精度比对方法。

```Python

def accuracy_comparison(y_cal, y_ref):
    """
    精度比对函数：根据数据类型选择合适的比对策略。

    不同数据类型的处理策略：
    - 浮点类型（float16/32, bfloat16）：使用 torch.testing.assert_close，设置相对/绝对误差容限
    - 整数类型（int8/16/32/64）：要求完全相等（torch.equal）
    - 布尔类型（bool）：CPU 上严格比较（避免设备差异）
    """
    # 检查输出数据类型是否一致
    assert y_cal.dtype == y_ref.dtype, f"dtype mismatch: {y_cal.dtype} vs {y_ref.dtype}"
    tensor_dtype = y_cal.dtype

    # 将张量移动到 NPU（假设测试在 NPU 上进行）
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()

    # 根据数据类型选择不同的比对方式
    if tensor_dtype == torch.float16:
        # float16 精度较低，允许稍大误差
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif tensor_dtype == torch.bfloat16:
        # bfloat16 精度更低，建议转为 float32 再比较
        torch.testing.assert_close(
            y_ref.to(torch.float32),
            y_cal.to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True
        )
    elif tensor_dtype == torch.float32:
        # float32 精度较高，使用更严格的容差
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-4, atol=1e-4, equal_nan=True)
    elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        # 整数类型应完全相等
        assert torch.equal(y_cal, y_ref), f"Integer tensors are not equal for dtype {tensor_dtype}"
    elif tensor_dtype == torch.bool:
        # 布尔类型建议在 CPU 上比较，避免设备间布尔表示差异
        assert torch.equal(y_cal.cpu(), y_ref.cpu()), "Boolean tensors are not equal"
    else:
        raise ValueError(f'Invalid or unsupported tensor dtype: {tensor_dtype}')


```

可以使用下面指令，运行参考示例代码：tutorials/14-accuracy-comparison.py

```Python
python triton-ascend/ascend/examples/tutorials/14-accuracy-comparison.py
```
