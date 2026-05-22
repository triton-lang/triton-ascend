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

    # 1. Use PyTorch as the reference implementation (golden truth)
    def torch_func(x0, x1):
        res = x0 + x1
        return res

    # 2. Define the Triton kernel (executes on NPU/GPU)
    @triton.jit
    def triton_kernel_add(
        out_ptr0,   # Output pointer: result storage location
        in_ptr0,    # Input pointer 0: starting address of x0
        in_ptr1,    # Input pointer 1: starting address of x1
        XS: tl.constexpr  # constexpr parameter: vector length, determined at compile time
    ):
        # Generate index array [0, 1, 2, ..., XS-1]
        idx = tl.arange(0, XS)
        # Load the value of x0 from in_ptr0 + idx
        tmp0 = tl.load(in_ptr0 + idx)
        # Load the value of x1 from in_ptr1 + idx
        tmp1 = tl.load(in_ptr1 + idx)
        # Perform addition
        tmp2 = tmp0 + tmp1
        # Write the result to out_ptr0 + idx
        tl.store(out_ptr0 + idx, tmp2)

    # 3. Triton wrapper function: call kernel and return result
    def triton_func(x0, x1):
        y0 = torch.empty_like(x0)  # Create output tensor with same shape and dtype as input
        # Launch kernel: grid = [1, 1, 1] means using only one block
        # Note: XS must be passed as a parameter because it is of tl.constexpr type
        triton_kernel_add[1, 1, 1](y0, x0, x1, XS=x0.numel())
        return y0

    # 4. Get the reference result and the Triton computation result
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1)

    # 5. Accuracy comparison
    accuracy_comparison(triton_cal, torch_ref)

    # 6. Print success message
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
    # Check that output data types match
    assert y_cal.dtype == y_ref.dtype, f"dtype mismatch: {y_cal.dtype} vs {y_ref.dtype}"
    tensor_dtype = y_cal.dtype

    # Move tensors to NPU (assuming the test runs on NPU)
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()

    # Choose comparison method based on data type
    if tensor_dtype == torch.float16:
        # float16 has lower precision, allow slightly larger error
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif tensor_dtype == torch.bfloat16:
        # bfloat16 has even lower precision, convert to float32 for comparison
        torch.testing.assert_close(
            y_ref.to(torch.float32),
            y_cal.to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True
        )
    elif tensor_dtype == torch.float32:
        # float32 has higher precision, use tighter tolerance
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-4, atol=1e-4, equal_nan=True)
    elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        # Integer types should be exactly equal
        assert torch.equal(y_cal, y_ref), f"Integer tensors are not equal for dtype {tensor_dtype}"
    elif tensor_dtype == torch.bool:
        # Boolean types should be compared on CPU to avoid device-specific boolean representation differences
        assert torch.equal(y_cal.cpu(), y_ref.cpu()), "Boolean tensors are not equal"
    else:
        raise ValueError(f'Invalid or unsupported tensor dtype: {tensor_dtype}')


```

可以使用下面指令，运行参考示例代码：tutorials/14-accuracy-comparison.py

```Python
python triton-ascend/ascend/examples/tutorials/14-accuracy-comparison.py
```
