# 自动调优 （Autotune）

在本节中，我们将展示使用 Triton 的 autotune 方法自动选择最优的 kernel 配置参数。当前 Triton-Ascend autotune 完全兼容社区 autotune 的使用方法（参考[社区文档](https://triton-lang.org/main/python-api/generated/triton.autotune.html)），即需要用户手动传入一些定义好的 triton.Config，然后 autotune 会通过 benchmark 的方式选择其中的最优 kernel 配置；此外 Triton-Ascend 提供了**进阶的 autotune** 用法，用户无需提供triton kernel 的切分轴、tiling 轴等信息，autotune 会根据triton kernel语义自动解析切分轴、tiling轴等信息，并自动生成一些可能最优的 kernel 配置，然后通过 benchmark 或者 profiling 的方式选择其中的最优配置。

说明：
当前Triton-Ascend autotune支持block size、multibuffer（编译器的优化），因为硬件架构差异不支持num_warps、num_stages参数，未来还会持续增加autotune可调项。

## 社区 autotune 使用示例
```Python
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

    @triton.autotune(
        configs=get_autotune_config(),      # 配置列表
        key=["numel"],                      # 当numel大小发生变化时会触发autotune
    )
    @triton.jit
    def triton_calc_kernel(
        out_ptr0, in_ptr0, in_ptr1, numel,
        XS: tl.constexpr                  # 块大小，用于控制每个线程块处理多少数据
    ):
        pid = tl.program_id(0)            # 获取当前 program 的 ID
        idx = pid * XS + tl.arange(0, XS) # 当前线程块处理的 index 范围
        msk = idx < numel                 # 避免越界的掩码

        # 重复执行一些计算以模拟负载（并测试性能）/ Repeat computation to simulate load (for perf test)
        for i in range(10000):
            tmp0 = tl.load(in_ptr0 + idx, mask=msk, other=0.0)  # 加载 x0
            tmp1 = tl.load(in_ptr1 + idx, mask=msk, other=0.0)  # 加载 x1
            tmp2 = tl.math.exp(tmp0) + tmp1 + i                # 计算
            tl.store(out_ptr0 + idx, tmp2, mask=msk)           # 存储到输出

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

    DEV = "npu"                         # 使用 NPU 作为设备
    DTYPE = torch.float32
    N = 192 * 1024                      # 输入长度
    x0 = torch.randn((N,), dtype=DTYPE, device=DEV)  # 随机输入 x0
    x1 = torch.randn((N,), dtype=DTYPE, device=DEV)  # 随机输入 x1
    torch_ref = torch_calc_func(x0, x1)              # 得到参考结果
    triton_cal = triton_calc_func(x0, x1)            # 运行 Triton kernel
    torch.testing.assert_close(triton_cal, torch_ref)  # 验证输出是否一致

if __name__ == "__main__":
    test_triton_autotune()
    print("success: test_triton_autotune")  # 输出成功标志 / Print success message
```

## 进阶 autotune 使用示例
```Python
# 下面说明进阶 autotune 与社区版的参数使用要点
#
# configs：
# - 社区版 autotune（默认）需要显式传入一组 triton.Config，框架会对这些配置逐一编译并基准测试以选择最优配置
# - 进阶版 autotune 框架基于 kernel 自动生成候选 tiling 配置，并对配置逐一编译并基准测试以选择最优配置
# * 注意：1. 进阶模式启动需用户手动 import triton.backends.ascend.runtime;
#        2. 若 configs=[]，框架基于 kernel 自动生成候选 tiling 配置，注意此时需要将@triton.autotune装饰器直接应用在@triton.jit之上，
#           中间不能插入其他装饰器，例如libentry;
#        3. 若 configs 不为空，则框架默认不会自动生成候选 tiling 配置;
#        4. 若 configs 不为空，且hints.auto_gen_config=True,则框架自动生成Config,并与用户定义Config合并进行配置择优；
#        5. 进阶版本支持通过设置os.environ["TRITON_BENCH_METHOD"] = ( "npu" ) 来设置性能采集方式。
# 
# hints(Dict[str, str])：
# 注意：1. hints可选，用户不填时框架会自动解析切分轴（split_params），分块轴（tiling_params）等相关参数
#      2. 用户可通过hints传参来生成tiling,涉及切分轴（split_params）、分块轴（tiling_params）、低维轴（low_dim_axes）、规约轴（reduction_axes），且四个参数需同时提供

# split_params (Dict[str, str]): axis name: argument name组成的字典, argument 是切分轴的可调参数, 例如 'XBLOCK'
#     axis name必须在参数key的轴名称集合里。 请勿在轴名称前添加前缀 r
#     此参数可以为空，当split_params 和 tiling_params 都为空的时候不会进行自动寻优
#     切分轴通常可以根据 `tl.program_id()` 分核语句来确定
# tiling_params (Dict[str, str]): axis name: argument name组成的字典， argument 是分块轴的可调参数, 例如 'XBLOCK_SUB'
#     axis name必须在参数key的轴名称集合里。请勿在轴名称前添加前缀 r
#     此参数可以为空，当split_params 和 tiling_params 都为空的时候不会进行自动寻优
#     分块轴通常可以根据 `tl.arange()` 分块表达式来确定
# low_dim_axes (List[str]): 所有低维轴的轴名称列表，axis name必须在参数key的轴名称集合里
# reduction_axes (List[str]): 所有规约轴的轴名称列表，axis name必须在参数key的轴名称集合里， 在轴名称前添加前缀 r
# auto_gen_config (bool): 默认为False,涉及如下场景组合
#     1. 用户未定义Config,无论是否设置auto_gen_config,框架默认自动生成Config；
#     2. 用户定义了Config,且auto_gen_config=False,则框架不自动生成Config,只使用用户定义的Config；
#     3. 用户定义了Config,且auto_gen_config=True,则框架自动生成Config,并与用户定义Config合并进行配置择优；
#
# key（list[str]/Dict[str,str]）：
# - 传入运行时参数名列表；列表中任一参数值变化会触发候选配置的重新生成与评估
# 注意：1.若hints传递切分轴（split_params）、分块轴（tiling_params）、低维轴（low_dim_axes）、规约轴（reduction_axes）参数信息，key类型需为Dict[str,str],如示例1：
#      2.若hints不传递切分轴（split_params）、分块轴（tiling_params）、低维轴（low_dim_axes）、规约轴（reduction_axes）参数信息，key类型需为list[str]，轴信息会按参数顺利进行分配，如示例2：

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
}
@triton.jit
def add_kernel(
    x_ptr,  # *指向*第一个输入向量的指针。
    y_ptr,  # *指向*第二个输入向量的指针。
    output_ptr,  # *指向*输出向量的指针。
    n_elements,  # 向量的大小。
    BLOCK_SIZE: tl.constexpr,  # 每个核应该处理的元素数量。
    # 注意：`constexpr` 表示它可以在编译时确定，因此可以作为形状（shape）值使用。
):
    pid = tl.program_id(axis=0)  # 我们使用一维的grid，因此轴为0。
    # 当前核将处理的数据在内存中相对于起始地址的偏移。
    # 例如，如果你有一个长度为256的向量，且块大小（block_size）为64，那么各个程序
    # 将分别访问元素 [0:64, 64:128, 128:192, 192:256]。
    # 注意，offsets 是一个指针列表：
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    # 创建一个掩码（mask），以防止内存操作访问越界。
    mask = offsets < n_elements
    # 加载x和y，并使用掩码屏蔽掉多余的元素，以防输入向量的长度不是块大小的整数倍。
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    # 将 x + y 写回。
    tl.store(output_ptr + offsets, output, mask=mask)
```

说明：
1. Triton-Ascend默认采取benchmark的方式取片上计算时间，当设置环境变量`export TRITON_BENCH_METHOD="npu"`后，会通过`torch_npu.profiler.profile`的方式获取每个kernel配置下的片上计算时间，对于一些triton kernel计算快速的情况，例如小shape算子，相较于默认方式能够获取更准确的计算时间，但是会显著增加整体autotune的时间，请谨慎开启
2. 目前该进阶用法针对的是 Vector 类算子，不支持 Cube 类算子。更多进阶使用示例可以参考[autotune进阶使用示例](https://gitcode.com/Ascend/triton-ascend/tree/main/third_party/ascend/unittest/autotune_ut)

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

# XBLOCK和XBLOCK_SUB未传入，则作为切分轴和分块轴参数的候选项
# BLOCK_SIZE以关键字参数传入，不作为参数候选项，不会被识别
kernel_func[grid](y, x, n_rows, n_cols, BLOCK_SIZE=block_size)
```

#### 切分轴参数解析

切分轴参数解析依据 `tl.program_id()`分核语句来确定 ，系统通过分析程序中 `tl.program_id()` 变量的使用情况及其与其他变量的乘法运算识别潜在的切分轴参数（当前支持直接相乘或通过中间变量间接相乘的场景），并根据候选参数列表（用户未提供的参数）进行过滤。

最后通过掩码比较和 `autotune` 中传入的 `key` 确认当前参数对应的切分轴。

注意：1. 分割轴参数必须要与 `tl.program_id()` 相乘。 2. 必须要进行掩码比较，且该轴对应的key需要直接作为右值或以key为参数的min函数作为右值，才能对应到具体的切分轴，否则会导致参数解析失败。3. 识别出的分割轴参数仅限于候选参数列表，确保只有那些可以通过自动调优动态调整的参数才会被考虑。

```Python
@triton.autotune(
    configs=[],
    key={"n_elements"} # 需要指定
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

# 解析得到切分轴参数 split_params = {"x": "XBLOCK"}
```

#### 分块轴参数解析

分块轴参数依据 `tl.arange()` ，`tl.range()`，`range()` 分块语句来确定。通过分析程序中`for` 循环里的 `tl.range()`，`tl.arange()`以及`range()` 的使用情况及其计算得到的变量来识别潜在的分块轴参数，提取 `tl.range()` 或 `range()` 中和 `tl.arange()` 的共同参数，并根据候选参数列表（用户未提供的参数）进行过滤。

最后通过掩码比较和 `autotune` 中传入的 `key` 确认当前参数对应的分块轴。

注意：1. 分块轴参数必须出现在 `tl.arange()` 的调用中，并且需在 `for` 循环中通过 `tl.range()`、`range()` 或整除运算（`//`）参与循环范围的计算。 2. 必须要进行掩码比较，且该轴对应的key需要直接作为右值或以key为参数的min函数作为右值，才能对应到具体的分块轴，否则会导致参数解析失败。3. 识别出的分块轴参数仅限于候选参数列表，确保只有那些可以通过自动调优动态调整的参数才会被考虑。

```Python
@triton.autotune(
    key={"n_rows", "n_cols"} # 需要指定
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

# 解析得到分块轴参数 tiling_params = {"x": "XBLOCK_SUB"}
# 参数BLOCK_SIZE虽然也在tl.arange中且与n_cols比较计算mask，但不是一个分块轴参数
```

#### 低维轴参数解析

低维轴参数解析依据 `tl.arange()` 分块语句来确定，通过分析程序中 `tl.arange()` 的使用情况及其计算得到的变量来识别潜在的低维轴参数，提取 `tl.arange()` 本身以及它参与计算的变量，通过是否进行切片操作来进行增维，以及通过判断增维维度来进行过滤。

最后通过掩码比较和 `autotune` 中传入的 `key` 确认当前kernel的低维轴。

注意：1. 低维轴必须要通过`tl.arange()`进行计算，并进行切片。并在非最低维进行维度扩充或不参与切片，才会被识别。 2. 若不进行掩码比较则无法对应到具体的低维轴，会导致参数解析失败。

```Python
@triton.autotune(
    key={"n_rows", "n_cols"} # 会按顺序自动分配成 {"x": "n_rows", "y": "n_cols"}
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

# 解析得到低维轴 low_dim_axes = {"y"}
# row_offsets虽然也通过tl.arange计算且与n_rows比较计算mask，但切片在低维进行扩充，所以x不是一个低维轴
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

# 解析得到指针类型参数为：input_ptr, output_ptr
```

## 更多功能
### 自动生成最优配置的 Profiling 结果
```Python
# 自动在`auto_profile_dir`目录中生成当前autotune最优kernel配置的profiling结果，即利用`torch_npu.profiler.profile`采集的性能数据
# 在社区autotune用法和进阶autotune用法中均可生效
@triton.autotune(
    auto_profile_dir="./profile_result",
    ...
)
```
