# 精度比对 （Accuracy Comparison）

在本节中，我们将使用 Triton 编写一个简单的精度比对的程序。
在此过程中，用户会学习到：

- Triton 每种数据类型的精度比对方法。

计算内核:

```Python
def test_add(x0, x1):
    """
    Tests whether the vector addition implemented in Triton produces results consistent with PyTorch within the accuracy comparison.

    Steps:
    1. Use PyTorch to compute the reference result (torch_ref)
    2. Use Triton to write the kernel and compute the result (triton_cal)
    3. Call accuracy_comparison to compare precision
    """

    # 1. Use PyTorch as the reference implementation (golden truth)
    def torch_func(x0, x1):
        res = x0 + x1
        return res

    # 2. Define the Triton kernel (executed on NPU/GPU)
    @triton.jit
    def triton_kernel_add(
        out_ptr0,   # Output pointer: where the result is stored
        in_ptr0,    # Input pointer 0: starting address of x0
        in_ptr1,    # Input pointer 1: starting address of x1
        XS: tl.constexpr  # constexpr parameter: vector length, determined at compile time
    ):
        # Generate an index array [0, 1, 2, ..., XS-1]
        idx = tl.arange(0, XS)
        # Load the value of x0 from in_ptr0 + idx
        tmp0 = tl.load(in_ptr0 + idx)
        # Load the value of x1 from in_ptr1 + idx
        tmp1 = tl.load(in_ptr1 + idx)
        # Perform addition
        tmp2 = tmp0 + tmp1
        # Write the result to out_ptr0 + idx
        tl.store(out_ptr0 + idx, tmp2)

    # 3. Triton wrapper function: invokes the kernel and returns the result
    def triton_func(x0, x1):
        y0 = torch.empty_like(x0)  # Create an output tensor with the same shape and dtype as the input
        # Launch the kernel: grid = [1, 1, 1] means only one block is used
        # Note: XS must be passed as an argument because it is a tl.constexpr type
        triton_kernel_add[1, 1, 1](y0, x0, x1, XS=x0.numel())
        return y0

    # 4. Get the reference result and the Triton computation result
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1)

    # 5. Compare precision
    accuracy_comparison(triton_cal, torch_ref)

    # 6. Print the success message
    print(f"== dtype:{triton_cal.dtype} == The accuracy comparison between triton_cal and torch_ref was successful.")


```

创建一个精度比对函数，适应每一种dtype，采用对应的精度比对方法。

```Python

def accuracy_comparison(y_cal, y_ref):
    """
    Accuracy comparison function: selects an appropriate comparison strategy based on the data type.

    Handling strategies for different data types:
    - Floating-point types (float16/32, bfloat16): use torch.testing.assert_close with relative/absolute error tolerances
    - Integer types (int8/16/32/64): require exact equality (torch.equal)
    - Boolean type (bool): strictly compare on CPU (to avoid device differences)
    """
    # Check if the output data types match
    assert y_cal.dtype == y_ref.dtype, f"dtype mismatch: {y_cal.dtype} vs {y_ref.dtype}"
    tensor_dtype = y_cal.dtype

    # Move the tensors to the NPU (assuming the test is performed on the NPU)
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()

    # Choose different comparison methods based on the data type
    if tensor_dtype == torch.float16:
        # float16 has lower precision, so a slightly larger tolerance is allowed
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif tensor_dtype == torch.bfloat16:
        # bfloat16 has even lower precision; it is recommended to convert to float32 before comparing
        torch.testing.assert_close(
            y_ref.to(torch.float32),
            y_cal.to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True
        )
    elif tensor_dtype == torch.float32:
        # float32 has high precision, so stricter tolerances are used
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-4, atol=1e-4, equal_nan=True)
    elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        # Integer types must be exactly equal
        assert torch.equal(y_cal, y_ref), f"Integer tensors are not equal for dtype {tensor_dtype}"
    elif tensor_dtype == torch.bool:
        # Boolean types are recommended to be compared on CPU to avoid device representation differences
        assert torch.equal(y_cal.cpu(), y_ref.cpu()), "Boolean tensors are not equal"
    else:
        raise ValueError(f'Invalid or unsupported tensor dtype: {tensor_dtype}')


```
