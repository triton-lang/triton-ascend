# Autotune

In this section, we will demonstrate how to use the autotune method of Triton to automatically select the optimal kernel configuration parameters. Currently, Triton-Ascend autotune is fully compatible with the usage of the autotune in the community (visit https://triton-lang.org/main/python-api/generated/triton.autotune.html). That is, users need to manually pass some defined **triton.Config** to autotune, and then autotune selects the optimal kernel configuration through benchmarking. In addition, Triton-Ascend provides the **advanced autotune** usage. Users need to provide information such as the split and tiling axes of the current Triton kernel. In this case, autotune automatically generates some possible optimal kernel configurations based on the actual input size, and then selects the optimal configuration through benchmarking or profiling.

Note:
Currently, Triton-Ascend autotune supports block size and multibuffer (compiler optimization). However, the **num_warps** and **num_stages** parameters are not supported due to hardware architecture differences. In the future, more adjustable autotune options will be added.

## Community Autotune Usage Example
```Python
import torch, torch_npu
import triton
import triton.language as tl

def test_triton_autotune():

    # Return a group of different kernel configurations for autotune testing.
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': False}),
            triton.Config({'XS': 8 * 1024, 'multibuffer': True}),
        ]

    @triton.autotune(
        configs=get_autotune_config(),      # Configuration list
        key=["numel"],                      # Autotune is triggered when the numel size changes.
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        XS: tl.constexpr                  # Block size, which is used to control the amount of data processed by each thread block.
    ):
        pid = tl.program_id(0)            # Obtain the ID of the current program.
        idx = pid * XS + tl.arange(0, XS) # Index range processed by the current thread block.
        msk = idx < numel                 # Mask to avoid out-of-bounds access.

        # Repeat computation to simulate load (for perf test).
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=msk, other=0.0)  # Load x0.
            tmp1 = tl.load(in_ptr1 + idx, mask=msk, other=0.0)  # Load x1.
            tmp2 = tl.math.exp(tmp0) + tmp1 + i                # Compute.
            tl.store(out_ptr0 + idx, tmp2, mask=msk)           # Store and output the result.

    # Triton calls a function and automatically uses the autotuned kernel.
    def triton_calc_func(x0, x1):
        n = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(n, meta["XS"]), 1, 1)  # Compute the grid size.
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0

    # Use PyTorch as the reference implementation for comparison.
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000 - 1

    DEV = "npu"                         # Use the NPU as the device.
    DTYPE = torch.float32
    N = 192 * 1024                      # Input length.
    x0 = torch.randn((N,), dtype=DTYPE, device=DEV)  # Randomly input x0.
    x1 = torch.randn((N,), dtype=DTYPE, device=DEV)  # Randomly input x1.
    torch_ref = torch_calc_func(x0, x1)              # Obtain the reference result.
    triton_cal = triton_calc_func(x0, x1)            # Run the Triton kernel.
    torch.testing.assert_close(triton_cal, torch_ref)  # Verify whether the outputs are consistent.

if __name__ == "__main__":
    test_triton_autotune()
    print("success: test_triton_autotune")  # Print success message.
```

## Advanced Autotune Usage Example
```Python
# The following are parameters added or modified compared with the community autotune.
# Note: When either split_params or tiling_params is not empty, the advanced autotune method is automatically triggered.

# In the dictionary consisting of "key (Dict[str, str]): axis name: argument name", the change of the argument triggers the regeneration and evaluation of candidate configurations.
#     The axis name belongs to the set {'x', 'y', 'z', 'w', 'v', 't', 'rx', 'ry', 'rz', 'rw', 'rv', 'rt'}. The prefix 'r' indicates the reduction axis.
#     The prefix 'r' should be added only when the axis name in this parameter is used as the reduction axis.
# In the dictionary consisting of "split_params (Dict[str, str]): axis name: argument name", the argument is the tunable parameter of the split axis, for example, 'XBLOCK'.
#     The axis name must be in the axis name set of the parameter `key`. Do not prefix the axis name with 'r'.
#     This parameter can be left empty. If both split_params and tiling_params are empty, autotune is not performed.
#     The split axis can be determined based on the kernel splitting statement `tl.program_id()`.
# In the dictionary consisting of "tiling_params (Dict[str, str]): axis name: argument name", the argument is an tunable parameter of the tiling axis, for example, 'XBLOCK_SUB'.
#     The axis name must be in the axis name set of the parameter `key`. Do not prefix the axis name with 'r'.
#     This parameter can be left empty. If both split_params and tiling_params are empty, autotune is not performed.
#     The tiling axis can be determined based on the `tl.arange()` expression.
# low_dims (List[str]): list of axis names of all low-dimensional axes. The axis name must be in the axis name set of the parameter `key`. Do not prefix the axis name with 'r'.
# dual_reduction (bool): specifies whether to perform reduction on multiple axes, which affects the tiling generation policy.
# persistent_reduction (bool): specifies whether to perform tiling on the reduction axis, which affects the tiling generation policy.
# For details, see the cases in ascend\examples\autotune_cases.
@triton.autotune(
    configs=[],
    key={"x": "n_elements"},           # Size of the split axis x.
    split_params={"x": "BLOCK_SIZE"},  # Size of BLOCK_SIZE to be adjusted for the split axis x.
    tiling_params={},                  # The tiling axis is the split axis.
    low_dims=["x"],                    # Low-dimensional axis.
    persistent_reduction=False,
    dual_reduction=False,
)
@triton.jit
def add_kernel(
    x_ptr,  # Pointer to the first input vector.
    y_ptr,  # Pointer to the second input vector.
    output_ptr,  # Pointer to the output vector.
    n_elements, # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements that should be processed by each kernel.
    # Note: `constexpr` indicates that it can be determined at compile time and therefore can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # A one-dimensional grid is used, so the axis is 0.
    # Offset of the data to be processed by the current kernel in the memory relative to the start address.
    # For example, if there is a vector of length 256 and block sizes 64, each program
    # will access the elements [0:64, 64:128, 128:192, 192:256] respectively.
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent out-of-bounds memory access.
    mask = offsets < n_elements
    # Load x and y, and use the mask to mask out the redundant elements to prevent the length of the input vector from not being an integer multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back.
    tl.store(output_ptr + offsets, output, mask=mask)
```

Note:
1. By default, Triton-Ascend uses the benchmark mode to obtain the on-chip computation time. After the environment variable is set by running `export TRITON_BENCH_METHOD="npu"`, the on-chip computation time of each kernel is obtained by using `torch_npu.profiler.profile`. For some Triton kernels that compute fast, such as small-shape operators, this method can obtain more accurate computation time than the default method. However, this will significantly increase the overall autotune time. Therefore, exercise caution when enabling this method.
2. Currently, this advanced usage is mainly used for vector operators and is not supported by cube operators. For more advanced usage examples, see [Advanced Autotune Cases](https://gitcode.com/Ascend/triton-ascend/tree/master/ascend/examples/autotune_cases).

### Automatic Parameter Parsing

Before automatically parsing parameters, the system obtains the parameters that are not passed during the `kernel` function call. **The parameters that are not passed are used as the candidate parameters for the split axis and tiling axis.**

```Python
@triton.jit
def kernel_func(
    outputptr,
    input_ptr,
    n_rows,
    n_cols,
    BLOCK_SIZE: tl.constexpr,
    XBLOCK: tl.constexpr,
    XBLOCK_SUB: tl.constexpr,
):
    # kernel implementation
    ...

# If XBLOCK and XBLOCK_SUB are not passed, they are used as candidate parameters for the split axis and tiling axis.
# BLOCK_SIZE is passed as a keyword argument and is not used as a candidate parameter. Therefore, it will not be identified.
kernel_func[grid](y, x, n_rows, n_cols, BLOCK_SIZE=block_size)
```

#### Split Axis Parameter Parsing

The split axis parameters are parsed based on the kernel splitting statement `tl.program_id()`. The system analyzes the usage of the `tl.program_id()` variable in the program and the multiplication operation between the variable and other variables to identify potential split axis parameters (currently, direct or indirect multiplication through intermediate variables is supported) and filters the parameters based on the candidate parameter list (parameters not provided by users).

Finally, the split axis corresponding to the current parameters is identified through mask comparison and the `key` passed in `autotune`.

Notes: 1. The split axis parameter must be multiplied by `tl.program_id()`. 2. The mask comparison must be performed, and the `key` corresponding to the split axis or the min function with the `key` as the parameter must be used as the right value. Otherwise, the axis cannot be identified and the parameter parsing will fail.3. The identified axis parameters are limited to the candidate parameter list. This ensures that only the parameters that can be dynamically tuned by autotune are considered.

```Python
@triton.autotune(
    key={"n_elements"} # It needs to be specified.
    ...
)
@triton.jit
def triton_func(...):
    # case1:
    pid = tl.program_id(0)
    block_start = pid * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)

    # case2:
    block_start = tl.program_id(0) * XBLOCK
    offsets = block_start + tl.arange(0, XBLOCK)

    # case3:
    offsets = tl.program_id(0) * XBLOCK + tl.arange(0, XBLOCK)

    # mask compare
    mask = offsets < n_elements # 1
    mask = offsets < min(..., n_elements) # 2

# The split axis parameter split_params is parsed as {"x": "XBLOCK"}.
```

#### Tiling Axis Parameter Parsing

The tiling axis parameter is determined based on the `tl.arange()`, `tl.range()`, and `range()` tiling statements. The potential tiling axis parameters are identified by analyzing the usage of `tl.range()`, `tl.arange()`, and `range()` in the `for` loop in the program, and the variables computed based on the usage. The common parameters of `tl.range()` or `range()` and `tl.arange()` are extracted and filtered based on the candidate parameter list (parameters not provided by users).

Finally, the split axis corresponding to the current parameter is identified through mask comparison with the `key` passed in `autotune`.

Notes: 1. The tiling axis parameters must be used in the call of `tl.arange()` and be involved in the computation of the loop range in the `for` loop through `tl.range()`, `range()`, or integer division (`//`). 2. The mask comparison must be performed, and the key corresponding to the tiling axis or the min function with the key as the parameter must be used as the right value. Otherwise, the axis cannot be identified and the parameter parsing will fail.3. The identified tiling parameters are limited to the candidate parameter list. This ensures that only the parameters that can be dynamically tuned by autotune are considered.

```Python
@triton.autotune(
    key={"n_rows", "n_cols"} # It needs to be specified.
    ...
)
@triton.jit
def triton_func(...):
    ...
    # case 1
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        row_offsets = row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]

    # case 2
    loops = (XBLOCK + XBLOCK_SUB - 1) // XBLOCK_SUB
    for loop in range(loops):
        row_offsets = loop * XBLOCK_SUB + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]

        ...
        xmask = row_offsets < n_rows # 1
        xmask = row_offsets < min(..., n_rows) # 2
        ymask = col_offsets < n_cols

# The tiling axis parameter tiling_params is parsed as {"x": "XBLOCK_SUB"}.
# Although the BLOCK_SIZE parameter is also in tl.arange and is compared with n_cols to compute the mask, it is not a tiling axis parameter.
```

#### Low-Dimensional Axis Parameter Parsing

The low-dimensional axis parameters are parsed based on the tiling statement `tl.arange()`. The potential low-dimensional axis parameters are identified by analyzing the usage of `tl.arange()` in the program and the variables computed by it. `tl.arange()` and the variables involved in the computation are extracted. The dimension is expanded based on whether slicing is performed, and the filtering is performed based on the expansion of the dimension.

Finally, the low-dimensional axis of the current kernel is determined by comparing the mask with the `key` passed in `autotune`.

Notes: 1. The low-dimensional axis must be computed using `tl.arange()` and sliced. It will be identified only when expansion is perform on or slicing is not involved in the non-lowest dimension. 2. If mask comparison is not performed, the specific low-dimensional axis cannot be identified, resulting in parameter parsing failure.

```Python
@triton.autotune(
    key={"n_rows", "n_cols"} # Automatically allocated in the order of {"x": "n_rows", "y": "n_cols"}
    ...
)
@triton.jit
def triton_func(...):
    ...
    for row_idx in tl.range(0, XBLOCK, XBLOCK_SUB):
        row_offsets = row_idx + tl.arange(0, XBLOCK_SUB)[:, None]
        col_offsets = tl.arange(0, BLOCK_SIZE)[None, :]

        xmask = row_offsets < n_rows
        ymask = col_offsets < n_cols

# The low-dimensional axis low_dims is parsed as {"y"}.
# Although row_offsets is also computed using tl.arange and compared with n_rows to compute the mask, slices are expanded in a low dimension. Therefore, x is not a low-dimensional axis.
```

#### Parameter Pointer Parsing

The pointer-type parameters are parsed based on whether the parameters are involved in the memory access statements of `tl.load()` and `tl.store()`.

First, all parameters in the kernel function are parsed, and then all variables involved in the computation of each parameter are recursively searched.

If a parameter is directly or indirectly (via the intermediate variable obtained by the parameter through computation) involved in the computation of the first parameter of `tl.load()` and `tl.store()`, the parameter is considered as a pointer-type parameter.

Notes: 1. Variables modified by `tl.constexpr` are not pointer-type variables and will not be parsed subsequently. 2. Only memory access statements with directly or indirectly (via the intermediate variable obtained by parameters through computation) involved parameters are counted. If the intermediate variables obtained by these parameters are involved in the computation for more than two times, the intermediate variables are not counted.

```Python
@triton.autotune(...)
@triton.jit
def triton_func(input_ptr, output_ptr, ...):
    ...
    # case1
    input = tl.load(input_ptr + offsets, mask=mask)
    tl.store(output_ptr + offsets, input, mask=mask)

    # case2
    inputs_ptr = input_ptr + offsets
    input = tl.load(inputs_ptr, mask=mask)
    outputs_ptr = output_ptr + offsets
    tl.store(outputs_ptr, input, mask=mask)

# The parsed pointer parameters are input_ptr and output_ptr.
```

## More Functions
### Automatically Generating the Profiling Result of the Optimal Configuration
```Python
# Automatically generate the profiling result of the optimal kernel configuration of the current autotune in the `auto_profile_dir` directory, that is, the performance data collected by `torch_npu.profiler.profile`.
# This takes effect in both the community autotune usage and advanced autotune usage.
@triton.autotune(
    auto_profile_dir="./profile_result",
    ...
)
```
