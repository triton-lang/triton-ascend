# Matrix Multiplication

This section describes how to implement a matrix multiplication kernel using Triton.

## Compute Kernel

The following Triton kernel implements batched matrix multiplication with bias: 
The formula is as follows: 
$$ \text{output}[b, i, j] = \sum_k \text{x}[b, i, k] \cdot \text{y}[k, j] + \text{z}[b, i, j] $$  
Specifically:

- The shape of `x` is `(A, B)`.
- The shape of `y` is `(B, C)`.
- The shape of `z` (bias) is `(A, C)`.
- The shape of `output` is `(A, C)`.

This kernel assumes that a single block is responsible for computing the entire output matrix. It is applicable to small-scale matrices (A, B, and C are small and can be fully covered by the current program block).

```python
import pytest
import torch
import torch_npu
import triton
import triton.language as tl


@triton.jit
def triton_dot_2_Bias(
    output_ptr,   # Pointer to the output tensor, with shape (A, C)
    x_ptr,        # Pointer to the input tensor x, with shape (A, B)
    y_ptr,        # Pointer to the input tensor y, with shape (B, C)
    z_ptr,        # Pointer to the bias tensor z, with shape (A, C)
    A: tl.constexpr,  # Size of the first dimension (batch/number of rows)
    B: tl.constexpr,  # Shared dimension (number of columns in x and number of rows in y)
    C: tl.constexpr   # Size of the second dimension (number of columns)
):
    # Create an index vector.
    bidx = tl.arange(0, A)  # [0, 1,..., A-1], used for the row dimension.
    cidx = tl.arange(0, B)  # [0, 1,..., B-1], used for the columns of x or rows of y.
    didx = tl.arange(0, C)  # [0, 1,..., C-1], used for the column dimension.

    # Construct the linear index of x: (A, B) -> flattened to A*B
    Xidx = bidx[:, None] * B + cidx[None, :]  # Broadcast to form an (A, B) index grid

    # Construct the linear index of y: (B, C) -> flattened to B*C
    Yidx = cidx[:, None] * C + didx[None, :]  # (B, C) index grid

    # Construct the linear index of z and output: (A, C).
    Zidx = bidx[:, None] * C + didx[None, :]  # (A, C) index grid

    # Load data from global memory.
    X = tl.load(x_ptr + Xidx)  # Load the (A, B) sub-block.
    Y = tl.load(y_ptr + Yidx)  # Load the (B, C) sub-block.
    Z = tl.load(z_ptr + Zidx)  # Load the bias (A, C).

    # Perform matrix multiplication and add the bias.
    ret = tl.dot(X, Y) + Z #  tl.dot performs (A, B) × (B, C) → (A, C).

    # Write the result back to global memory.
    oidx = bidx[:, None] * C + didx[None, :]  # Same as Zidx, which can be reused.
    tl.store(output_ptr + oidx, ret)
```

## Tools and Methods

The following helper functions are used to support the testing and verification of Triton kernels, including PyTorch reference implementation, data type mapping, random tensor generation, and result verification.

```Python
def torch_dot_Bias(x0, x1, bias):
    """PyTorch reference implementation: Perform matrix multiplication and add the bias."""
    res = torch.matmul(x0, x1) + bias
    return res

def get_torch_typename(dtype):
    """Map the data type in string format to the corresponding torch.dtype."""
    if dtype == 'float32':
        tyname = torch.float32
    elif dtype == 'int32':
        tyname = torch.int32
    elif dtype == 'int64':
        tyname = torch.int64
    elif dtype == 'float16':
        tyname = torch.float16
    elif dtype == 'int16':
        tyname = torch.int16
    elif dtype == 'int8':
        tyname = torch.int8
    elif dtype == 'bool':
        tyname = torch.bool
    elif dtype == 'bfloat16':
        tyname = torch.bfloat16
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
    return tyname

def generate_tensor(shape, dtype):
     """Generates a random tensor based on the specified shape and data type, and adapts to the value ranges of different data types."""
    if dtype == 'float32' or dtype == 'float16' or dtype == 'bfloat16':
        return torch.randn(size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16':
        return torch.randint(low=0, high=2000, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'int8':
        return torch.randint(low=0, high=127, size=shape, dtype=eval('torch.' + dtype))
    elif dtype == 'bool':
        return torch.randint(low=0, high=2, size=shape).bool()
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))

def validate_cmp(dtype, y_cal, y_ref):
    """Compare the Triton compute result with the PyTorch reference result on the NPU, and set the tolerance or strict equality based on the data type."""
    y_cal=y_cal.npu()
    y_ref=y_ref.npu()
    if dtype == 'float16': 
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'bfloat16':
        torch.testing.assert_close(y_ref.to(torch.float32), y_cal.to(torch.float32),  rtol=1e-03, atol=1e-03, equal_nan=True)
    elif dtype == 'float32':
        torch.testing.assert_close(y_ref, y_cal,  rtol=1e-04, atol=1e-04, equal_nan=True)
    elif dtype == 'int32' or dtype == 'int64' or dtype == 'int16' or dtype == 'int8':
        assert torch.equal(y_cal, y_ref)
    elif dtype == 'bool':
        assert torch.equal(y_cal, y_ref)
    else:
        raise ValueError('Invalid parameter \"dtype\" is found : {}'.format(dtype))
```

## Parameterized Test

Use `pytest` to verify the parameterization function of the `triton_dot_2_Bias` kernel, covering different combinations of matrix dimensions and data types.

```python
# Test case configuration: (A, B, C) indicates that matrix x is (A, B), y is (B, C), and bias/output is (A, C).
testlist = [   
    (16, 16, 16), 
]

# Supported data types (only float16 is supported currently)
typelist = ['float16',]

@pytest.mark.parametrize('A, B, C', testlist)
@pytest.mark.parametrize('sigtype', typelist)
def test_dot_2_Bias(sigtype, A, B, C):
    """Perform an end-to-end function test on the triton_dot_2_Bias kernel."""
    dtype = get_torch_typename(sigtype)
    
    # Generate the input tensor and move it to the NPU.
    x0 = generate_tensor(shape=(A, B), dtype=sigtype).npu()
    x1 = generate_tensor(shape=(B, C), dtype=sigtype).npu()
    
    # The bias items are generated using float32 (to avoid accuracy issues caused by integer bias).
    if 'int' in sigtype:
        bias = generate_tensor(shape=(A, C), dtype='int32').npu()
        # The integer input needs to be converted to float32 for computation and then converted back to the target type.
        ans = torch_dot_Bias(x0.to(torch.float32), x1.to(torch.float32), bias.to(torch.float32)).to(dtype)
    else:
        bias = generate_tensor(shape=(A, C), dtype='float32').npu()
        ans = torch_dot_Bias(x0, x1, bias).to(eval(f"torch.{dtype}"))
    
    # Initialize the output tensor.
    output = torch.zeros((A, C), dtype=dtype).npu()
    
    # Start the Triton kernel (grid=(1,1,1), single-block execution).
    triton_dot_2_Bias[1, 1, 1](output, x0, x1, bias, A, B, C, debug=True)
    
    # Verify the result correctness.
    validate_cmp(sigtype, output, ans)
    print(f"Test matmul with dtype={sigtype}, shape=({A},{B},{C}) PASSED!")


if __name__ == "__main__":
    # Running a single test case directly (for debugging) is supported.
    test_dot_2_Bias("float16", 16, 16, 16)
```

**Output example:**

```python
Test matmul with dtype=float16, shape=(16,16,16) PASSED!
```

The preceding logs indicate that the output on Triton is the same as that on PyTorch.
