# Accuracy Comparison

In this section, you will use Triton to write a simple accuracy comparison program.
During this process, you will learn:

- The method of comparing the accuracy of each data type in Triton.
- Reference code: triton-ascend/ascend/examples/tutorials/14-accuracy-comparison.py

Compute kernel:

```Python
def test_add(x0, x1):
    """
    Test the vector addition implemented by Triton and compare its accuracy with that of PyTorch.

    Procedure:
    1. Use PyTorch to compute the reference result (torch_ref).
    2. Use Triton to compile the kernel and compute the result (triton_cal).
    3. Call accuracy_comparison to compare the accuracy.
    """

    # 1. Use PyTorch as the reference implementation (golden truth).
    def torch_func(x0, x1):
        res = x0 + x1
        return res

    # 2. Define the Triton kernel (executed on the NPU or GPU).
    @triton.jit
    def triton_kernel_add(
        out_ptr0,   # Pointer to the output: location where the result is stored
        in_ptr0,    # Pointer 0 to the input: start address of x0
        in_ptr1,    # Pointer 1 to the input: start address of x1
        XS: tl.constexpr  # constexpr parameter: vector length, which is determined at compile time
    ):
        # Generate an index array of [0, 1, 2,..., XS-1].
        idx = tl.arange(0, XS)
        # Load the value of x0 from in_ptr0 + idx.
        tmp0 = tl.load(in_ptr0 + idx)
        # Load the value of x1 from in_ptr1 + idx.
        tmp1 = tl.load(in_ptr1 + idx)
        # Perform addition.
        tmp2 = tmp0 + tmp1
        # Write the result to out_ptr0 + idx.
        tl.store(out_ptr0 + idx, tmp2)

    # 3. Triton encapsulation function: Call the kernel and return the result.
    def triton_func(x0, x1):
        y0 = torch.empty_like(x0)  # Create an output tensor with the same shape and dtype as the input.
        # Start the kernel. grid = [1, 1, 1] indicates that only one block is used.
        # Note: XS must be passed as a parameter because it is of the tl.constexpr type.
        triton_kernel_add[1, 1, 1](y0, x0, x1, XS=x0.numel())
        return y0

    # 4. Obtain the reference result and Triton computation result.
    torch_ref = torch_func(x0, x1)
    triton_cal = triton_func(x0, x1)

    # 5. Compare the accuracy.
    accuracy_comparison(triton_cal, torch_ref)

    # 6. Print success information.
    print(f"== dtype:{triton_cal.dtype} == The accuracy comparison between triton_cal and torch_ref was successful.")


```

Create an accuracy comparison function that adapts to each dtype and uses the corresponding accuracy comparison method.

```Python

def accuracy_comparison(y_cal, y_ref):
    """
    Accuracy comparison function: Select a proper comparison policy based on the data type.

    Processing policies for different data types:
    - Floating-point types (float16/32, bfloat16): Use torch.testing.assert_close and set the relative/absolute error tolerance.
    - Integer types (int8/16/32/64): The results must be equal (torch.equal).
    - Boolean type (bool): Strict comparison is performed on the CPU (to avoid device differences).
    """
    # Check whether the output data types are consistent.
    assert y_cal.dtype == y_ref.dtype, f"dtype mismatch: {y_cal.dtype} vs {y_ref.dtype}"
    tensor_dtype = y_cal.dtype

    # Move the tensor to the NPU (assuming that the test is performed on the NPU).
    y_cal = y_cal.npu()
    y_ref = y_ref.npu()

    # Select different comparison methods based on the data types.
    if tensor_dtype == torch.float16:
        # For the float16 type, the accuracy is low, and a slightly larger error is allowed.
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-3, atol=1e-3, equal_nan=True)
    elif tensor_dtype == torch.bfloat16:
        # For bfloat16, the accuracy is lower. You are advised to convert it to float32 before comparison.
        torch.testing.assert_close(
            y_ref.to(torch.float32),
            y_cal.to(torch.float32),
            rtol=1e-3,
            atol=1e-3,
            equal_nan=True
        )
    elif tensor_dtype == torch.float32:
        # For the float32 type, the accuracy is high. A stricter tolerance is recommended.
        torch.testing.assert_close(y_ref, y_cal, rtol=1e-4, atol=1e-4, equal_nan=True)
    elif tensor_dtype in [torch.int64, torch.int32, torch.int16, torch.int8]:
        # For the integer type, the results must be equal.
        assert torch.equal(y_cal, y_ref), f"Integer tensors are not equal for dtype {tensor_dtype}"
    elif tensor_dtype == torch.bool:
        # For the Boolean type, comparison on the CPU is recommended to avoid differences in Boolean representation between devices.
        assert torch.equal(y_cal.cpu(), y_ref.cpu()), "Boolean tensors are not equal"
    else:
        raise ValueError(f'Invalid or unsupported tensor dtype: {tensor_dtype}')


```

You can run the following command to execute the sample code: tutorials/14-accuracy-comparison.py.
```Python
python triton-ascend/ascend/examples/tutorials/14-accuracy-comparison.py
```
