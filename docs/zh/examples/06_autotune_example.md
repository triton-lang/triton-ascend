# 自动调优 （Autotune）

如果你希望先了解 Triton-Ascend autotune 的推荐用法、`configs=[]` 的含义，以及自动 Tiling 的适用边界，建议先阅读 [Triton-Ascend autotune 使用指南](../autotune_guide.md)。

在本节中，我们将展示使用 Triton 的 autotune 方法自动选择最优的 kernel 配置参数。当前 Triton-Ascend autotune 完全兼容社区 autotune 的使用方法（参考[社区文档](https://triton-lang.org/main/python-api/generated/triton.autotune.html)），即需要用户手动传入一些定义好的 triton.Config，然后 autotune 会通过 benchmark 的方式选择其中的最优 kernel 配置；此外 Triton-Ascend 提供了**进阶的 autotune** 用法，用户无需提供triton kernel 的切分轴、tiling 轴等信息，autotune 会根据triton kernel语义自动解析切分轴、tiling轴等信息，并自动生成一些可能最优的 kernel 配置，然后通过 benchmark 或者 profiling 的方式选择其中的最优配置。

说明：
当前Triton-Ascend autotune支持block size、multibuffer（编译器的优化），因为硬件架构差异不支持num_warps、num_stages参数，未来还会持续增加autotune可调项。

## 社区 autotune 使用示例

```Python
import torch, torch_npu
import triton
import triton.language as tl

def test_triton_autotune():

    # Return a set of different kernel configurations for autotune testing
    def get_autotune_config():
        return [
            triton.Config({'XS': 1 * 128, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': True}),
            triton.Config({'XS': 12 * 1024, 'multibuffer': False}),
            triton.Config({'XS': 8 * 1024, 'multibuffer': True}),
        ]

    @triton.autotune(
        configs=get_autotune_config(),      # Configuration list
        key=["numel"],                      # Autotune is triggered when the numel size changes
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        XS: tl.constexpr                  # Block size, controls how much data each thread block processes
    ):
        pid = tl.program_id(0)            # Get the ID of the current program
        idx = pid * XS + tl.arange(0, XS) # Index range processed by the current thread block
        msk = idx < numel                 # Mask to avoid out-of-bounds access

        # Repeat some computation to simulate load (and test performance) / Repeat computation to simulate load (for perf test)
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=msk, other=0.0)  # Load x0
            tmp1 = tl.load(in_ptr1 + idx, mask=msk, other=0.0)  # Load x1
            tmp2 = tl.math.exp(tmp0) + tmp1 + i                # Compute
            tl.store(out_ptr0 + idx, tmp2, mask=msk)           # Store to output

    # Triton invocation function that automatically uses the autotuned kernel
    def triton_calc_func(x0, x1):
        n = x0.numel()
        y0 = torch.empty_like(x0)
        grid = lambda meta: (triton.cdiv(n, meta["XS"]), 1, 1)  # Compute the grid size
        triton_calc_kernel[grid](y0, x0, x1, n)
        return y0

    # Use PyTorch as the reference implementation for comparison
    def torch_calc_func(x0, x1):
        return torch.exp(x0) + x1 + 10000 - 1

    DEV = "npu"                         # Use NPU as the device
    DTYPE = torch.float32
    N = 192 * 1024                      # Input length
    x0 = torch.randn((N,), dtype=DTYPE, device=DEV)  # Random input x0
    x1 = torch.randn((N,), dtype=DTYPE, device=DEV)  # Random input x1
    torch_ref = torch_calc_func(x0, x1)              # Get the reference result
    triton_cal = triton_calc_func(x0, x1)            # Run the Triton kernel
    torch.testing.assert_close(triton_cal, torch_ref)  # Verify the outputs match

if __name__ == "__main__":
    test_triton_autotune()
    print("success: test_triton_autotune")  # Print success flag / Print success message
```

## 进阶 autotune 使用示例

```Python
# The following explains the key parameter usage differences between the advanced autotune and the community version
#
# configs:
# - Community autotune (default) requires explicitly passing a set of triton.Config; the framework compiles and benchmarks each config one by one to select the best one
# - Advanced autotune: the framework automatically generates candidate tiling configs based on the kernel, and compiles and benchmarks each config to select the best one
# * Note: 1. To enable advanced mode, users must manually import triton.backends.ascend.runtime;
#        2. If configs=[], the framework automatically generates candidate tiling configs based on the kernel; note that the @triton.autotune decorator must be applied directly on top of @triton.jit,
#           and no other decorators (e.g. libentry) may be inserted in between;
#        3. If configs is not empty, the framework will not automatically generate candidate tiling configs;
#        4. If configs is not empty and hints.auto_gen_config=True, the framework auto-generates Configs and merges them with user-defined Configs for selection;
#        5. The advanced version supports setting the performance measurement method via os.environ["TRITON_BENCH_METHOD"] = ("npu").
#
# hints(Dict[str, str]):
# Note: 1. hints is optional; if not provided, the framework automatically parses related parameters such as split_params and tiling_params
#      2. Users can pass hints to generate tiling; this involves split_params, tiling_params, low_dim_axes, and reduction_axes, and all four parameters must be provided together

# split_params (Dict[str, str]): A dict of axis name: argument name; argument is the tunable parameter of the split axis, e.g. 'XBLOCK'
#     The axis name must be in the set of axis names of the parameter keys. Do not prefix the axis name with r
#     This parameter can be empty; when both split_params and tiling_params are empty, no automatic optimization is performed
#     The split axis can usually be determined by the `tl.program_id()` kernel-splitting statement
# tiling_params (Dict[str, str]): A dict of axis name: argument name; argument is the tunable parameter of the tiling axis, e.g. 'XBLOCK_SUB'
#     The axis name must be in the set of axis names of the parameter keys. Do not prefix the axis name with r
#     This parameter can be empty; when both split_params and tiling_params are empty, no automatic optimization is performed
#     The tiling axis can usually be determined by the `tl.arange()` tiling expression
# low_dim_axes (List[str]): List of axis names of all low-dimension axes; the axis name must be in the set of axis names of the parameter keys
# reduction_axes (List[str]): List of axis names of all reduction axes; the axis name must be in the set of axis names of the parameter keys, with the prefix r added before the axis name
# auto_gen_config (bool): Defaults to False; involves the following scenarios
#     1. If the user has not defined Config, the framework auto-generates Config by default regardless of the auto_gen_config setting;
#     2. If the user defines Config and auto_gen_config=False, the framework does not auto-generate Config and only uses the user-defined Config;
#     3. If the user defines Config and auto_gen_config=True, the framework auto-generates Config and merges it with the user-defined Config for selection;
#
# key (list[str]/Dict[str,str]):
# - A list of runtime parameter names; a change in any parameter value triggers regeneration and re-evaluation of candidate configs
# Note: 1. If hints passes split_params, tiling_params, low_dim_axes, reduction_axes, the key type must be Dict[str,str], as in Example 1:
#      2. If hints does not pass split_params, tiling_params, low_dim_axes, reduction_axes, the key type must be list[str], and axis information is assigned according to the parameter order, as in Example 2:

示例1:
@triton.autotune(
    configs=[],
    key={"x":"n_elements"},
    hints={
        "split_params":{"x":"BLOCK_SIZE"},
        "tiling_params":{},
        "low_dim_axes":["x"],
        "reduction_axes":[],
    }
)
示例2:
@triton.autotune(
    configs=[],
    key=["n_elements"],
)
@triton.jit
def add_kernel(
    x_ptr,  # *Pointer* to the first input vector.
    y_ptr,  # *Pointer* to the second input vector.
    output_ptr,  # *Pointer* to the output vector.
    n_elements,  # Size of the vector.
    BLOCK_SIZE: tl.constexpr,  # Number of elements each kernel should process.
    # Note: `constexpr` means it can be determined at compile time, so it can be used as a shape value.
):
    pid = tl.program_id(axis=0)  # We use a 1D grid, so the axis is 0.
    # The offset of the data this kernel will process in memory relative to the starting address.
    # For example, if you have a vector of length 256 and a block size of 64, each program
    # will access elements [0:64, 64:128, 128:192, 192:256] respectively.
    # Note that offsets is a list of pointers:
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # Create a mask to prevent memory operations from accessing out of bounds.
    mask = offsets < n_elements
    # Load x and y, masking out extra elements in case the input vector length is not a multiple of the block size.
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # Write x + y back.
    tl.store(output_ptr + offsets, output, mask=mask)
```

说明：

1. Triton-Ascend默认采取benchmark的方式取片上计算时间，当设置环境变量`export TRITON_BENCH_METHOD="npu"`后，会通过`torch_npu.profiler.profile`的方式获取每个kernel配置下的片上计算时间，对于一些triton kernel计算快速的情况，例如小shape算子，相较于默认方式能够获取更准确的计算时间，但是会显著增加整体autotune的时间，请谨慎开启
2. 目前该进阶用法针对的是 Vector 类算子，不支持 Cube 类算子。更多进阶使用示例可以参考[autotune进阶使用示例](https://gitcode.com/Ascend/triton-ascend/tree/main/third_party/ascend/unittest/autotune_ut/)。

### 参数自动解析

执行参数自动解析前首先会获取`kernel`函数调用时未传入的参数，**将未传入的参数作为切分轴和分块轴参数的候选项**。

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

# XBLOCK and XBLOCK_SUB are not passed, so they are candidates for split axis and tiling axis parameters
# BLOCK_SIZE is passed as a keyword argument, so it is not a parameter candidate and will not be recognized
kernel_func[grid](y, x, n_rows, n_cols, BLOCK_SIZE=block_size)
```

#### 切分轴参数解析

切分轴参数解析依据 `tl.program_id()`分核语句来确定 ，系统通过分析程序中 `tl.program_id()` 变量的使用情况及其与其他变量的乘法运算识别潜在的切分轴参数（当前支持直接相乘或通过中间变量间接相乘的场景），并根据候选参数列表（用户未提供的参数）进行过滤。

最后通过掩码比较和 `autotune` 中传入的 `key` 确认当前参数对应的切分轴。

注意：1. 分割轴参数必须要与 `tl.program_id()` 相乘。 2. 必须要进行掩码比较，且该轴对应的key需要直接作为右值或以key为参数的min函数作为右值，才能对应到具体的切分轴，否则会导致参数解析失败。3. 识别出的分割轴参数仅限于候选参数列表，确保只有那些可以通过自动调优动态调整的参数才会被考虑。

```Python
@triton.autotune(
    configs=[],
    key={"n_elements"} # Must be specified
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

# The split axis parameter is resolved as split_params = {"x": "XBLOCK"}
```

#### 分块轴参数解析

分块轴参数依据 `tl.arange()` ，`tl.range()`，`range()` 分块语句来确定。通过分析程序中`for` 循环里的 `tl.range()`，`tl.arange()`以及`range()` 的使用情况及其计算得到的变量来识别潜在的分块轴参数，提取 `tl.range()` 或 `range()` 中和 `tl.arange()` 的共同参数，并根据候选参数列表（用户未提供的参数）进行过滤。

最后通过掩码比较和 `autotune` 中传入的 `key` 确认当前参数对应的分块轴。

注意：1. 分块轴参数必须出现在 `tl.arange()` 的调用中，并且需在 `for` 循环中通过 `tl.range()`、`range()` 或整除运算（`//`）参与循环范围的计算。 2. 必须要进行掩码比较，且该轴对应的key需要直接作为右值或以key为参数的min函数作为右值，才能对应到具体的分块轴，否则会导致参数解析失败。3. 识别出的分块轴参数仅限于候选参数列表，确保只有那些可以通过自动调优动态调整的参数才会被考虑。

```Python
@triton.autotune(
    key={"n_rows", "n_cols"} # Must be specified
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

# The tiling axis parameter is resolved as tiling_params = {"x": "XBLOCK_SUB"}
# Although BLOCK_SIZE is also used in tl.arange and compared with n_cols to compute the mask, it is not a tiling axis parameter
```

#### 低维轴参数解析

低维轴参数解析依据 `tl.arange()` 分块语句来确定，通过分析程序中 `tl.arange()` 的使用情况及其计算得到的变量来识别潜在的低维轴参数，提取 `tl.arange()` 本身以及它参与计算的变量，通过是否进行切片操作来进行增维，以及通过判断增维维度来进行过滤。

最后通过掩码比较和 `autotune` 中传入的 `key` 确认当前kernel的低维轴。

注意：1. 低维轴必须要通过`tl.arange()`进行计算，并进行切片。并在非最低维进行维度扩充或不参与切片，才会被识别。 2. 若不进行掩码比较则无法对应到具体的低维轴，会导致参数解析失败。

```Python
@triton.autotune(
    key={"n_rows", "n_cols"} # Will be automatically assigned in order as {"x": "n_rows", "y": "n_cols"}
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

# The low-dimension axis is resolved as low_dim_axes = {"y"}
# Although row_offsets is also computed via tl.arange and compared with n_rows to compute the mask, the slicing expands along the low dimension, so x is not a low-dimension axis
```

#### 参数指针解析

指针类型的参数解析依据该参数是否参与 `tl.load()` 和 `tl.store()` 的访存类语句来确定。

首先解析出kernel函数中的所有参数，之后递归寻找每一个参数参与计算的所有变量。

如果该参数直接参与或该参数计算得到的中间变量间接参与 `tl.load()` 和 `tl.store()` 的第一个参数计算，则认为该参数是一个指针类型参数。

注意：1. 使用 `tl.constexpr` 修饰的变量不会是指针类型的变量，不进行后续解析 2. 只计算参数直接参与或参数经过一次计算得到的中间变量间接参与的访存类语句，若参数进行两次以上计算得到的中间变量不进行统计。

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

# The pointer type parameters are resolved as: input_ptr, output_ptr
```

## 更多功能

### 自动生成最优配置的 Profiling 结果

```Python
# Automatically generate profiling results of the current optimal autotune kernel configuration in the `auto_profile_dir` directory, i.e., performance data collected via `torch_npu.profiler.profile`
# This takes effect in both community autotune usage and advanced autotune usage
@triton.autotune(
    auto_profile_dir="./profile_result",
    ...
)
```
