# triton_launch_kernel

## 1. 接口概述

`triton_launch_kernel` 是 Ascend 后端 launcher stub 动态库（`.so`）中导出的 C 语言运行时接口，用于在已通过 CANN runtime 注册 kernel function handle 后，直接发射（launch）Triton 算子到 NPU 上执行。

该接口以 `extern "C"` 方式导出，与标准 Python JIT 调用路径（`@triton.jit` → `kernel[grid](...)`）**并列独立**——普通用户通过 `@triton.jit` 调用时不经过此函数；它面向的是需要 C 级别 kernel 发射能力的高级场景，如自定义部署流水线、推理引擎集成、同签名 kernel 复用等。

每个 JIT 编译产物（launcher stub `.so`）中均包含一份由 `generate_npu_wrapper_src()` 动态生成的 `triton_launch_kernel`，其参数区内存布局受 kernel signature、metadata（workspace / syncBlockLock / coalesce）、device_print、ffts 等因素共同影响。调用方应注意不同编译产物之间的布局差异（详见[第 8 章](#8-限制与注意事项)）。

## 2. 函数签名

```c
extern "C" {
void triton_launch_kernel(
    const char* kernelName,
    const void* func,
    rtStream_t stream,
    int gridX,
    int gridY,
    int gridZ,
    const int64_t* shapes_data,
    const int* shape_dims,
    int num_tensors,
    const int* tensor_kinds,
    const void* const* kernel_args,
    const size_t* arg_sizes,
    int num_args
);
}
```

## 3. 参数说明

| 参数名 | 类型 | 输入/输出 | 说明 |
|--------|------|-----------|------|
| `kernelName` | `const char*` | 输入 | kernel 名称字符串，用于日志和 profiling 标识 |
| `func` | `const void*` | 输入 | **CANN runtime 注册后的 kernel function handle**。调用方需先通过 `rtDevBinaryRegister` / `rtFunctionRegister` 完成注册，不可直接传入裸 `.aocx` 文件数据 |
| `stream` | `rtStream_t` | 输入 | CANN 运行时 stream 句柄，kernel 将在此 stream 上执行 |
| `gridX` | `int` | 输入 | 启动网格 X 维度。`blockNum = gridX × gridY × gridZ` 决定总 block 发射数 |
| `gridY` | `int` | 输入 | 启动网格 Y 维度 |
| `gridZ` | `int` | 输入 | 启动网格 Z 维度 |
| `shapes_data` | `const int64_t*` | 输入 | 各 tensor shape 按顺序拼接的一维展平数组。长度等于 `sum(shape_dims[i]) for i in 0..num_tensors-1`。主要用于 msprof tensor 信息上报，非 kernel 执行必需参数，可为 `nullptr`（此时 `shape_dims` 也应置空） |
| `shape_dims` | `const int*` | 输入 | 每个 tensor 的维度数，数组长度 = `num_tensors`。与 `shapes_data` 配对使用；若 `shapes_data` 为 `nullptr` 则也应为 `nullptr` |
| `num_tensors` | `int` | 输入 | tensor 总数，同时决定 `shape_dims` 和 `tensor_kinds` 数组的有效长度 |
| `tensor_kinds` | `const int*` | 输入 | 各 tensor 的类型标记，数组长度 = `num_tensors`。取值：`0` = INPUT，`1` = OUTPUT，`2` = INPUT_OUTPUT。主要用于 msprof tensor 分类上报；若为 `nullptr` 则默认按 INPUT 处理 |
| `kernel_args` | `const void* const*` | 输入 | kernel 参数指针数组，每个元素指向一个 kernel 参数的起始地址，按 kernel 函数签名顺序一一对应。数组长度 = `num_args`。若 `num_args` 为 0 则可为 `nullptr` |
| `arg_sizes` | `const size_t*` | 输入 | 各 kernel 参数的大小（字节），与 `kernel_args` 一一对应，数组长度 = `num_args`。若 `num_args` 为 0 则可为 `nullptr` |
| `num_args` | `int` | 输入 | kernel 参数个数。若 `num_args > 0` 但 `kernel_args` 或 `arg_sizes` 为 `nullptr`，函数将静默返回，不发射 kernel |

## 4. 参数内存布局与生命周期

### 4.1 kernel_args 深拷贝

函数内部会遍历 `kernel_args` 和 `arg_sizes`，将每个参数值 `memcpy` 到 `std::vector<std::vector<char>> copied_kernel_args` 本地容器中。因此：

- **调用方在函数返回后可立即释放 `kernel_args` 中各指针指向的内存**，不影响发射
- 深拷贝带来额外内存开销，开销大小 = `sum(arg_sizes[i])`

### 4.2 launch_args 连续缓冲区布局

函数内部将所有发射参数组装到一段连续的 `std::vector<char> launch_args` 中，按以下顺序布局（各槽位按对齐要求偏移）：

```
[ffts_addr] → [syncBlockLock_ptr] → [workspace_addr_ptr] →
[kernel_arg_0] [kernel_arg_1] ... [kernel_arg_N-1] →
[gridX] [gridY] [gridZ] →
[DTData]   // 仅当 TRITON_DEVICE_PRINT="true" 时存在
```

> **注意：** 此布局由 `generate_npu_wrapper_src()` 根据当前 kernel 的 signature、metadata、编译选项动态生成，不同编译产物的布局可能不同。

### 4.3 workspace 与 syncBlockLock

- **workspace**：当 `metadata.workspace_size > 0` 时，函数内部通过 `allocate_memory` 分配 workspace，并以 `std::shared_ptr` 管理生命周期，函数返回时自动释放
- **syncBlockLock**：当 `metadata.lock_num > 0` 时，函数内部通过 `allocate_sync_block_lock` 分配同步锁内存，初始化值为 `metadata.lock_init_value`，同样以 `std::shared_ptr` 管理生命周期
- 分配失败时向 stderr 输出错误信息并直接 `return`，不抛出异常

### 4.4 stream 同步策略

取决于 `TRITON_ENABLE_TASKQUEUE` 环境变量（默认 `"true"`）：

| 模式 | 行为 | 函数返回时机 |
|------|------|-------------|
| TaskQueue 启用（默认） | 将 `rtKernelLaunch` 封装为 `std::function<rtError_t()>`，通过 `triton_async_launch` 提交到任务队列 | 提交后立即返回，不等待 kernel 执行完成 |
| TaskQueue 禁用 | 同步执行 `rtKernelLaunch`，随后调用 `rtStreamSynchronize(stream)` | 等待 kernel 执行完成后返回 |

## 5. 调用路径与适用场景

### 5.1 路径 A：标准 Python JIT 路径（普通用户）

此路径**不经过** `triton_launch_kernel`，是 Triton 的标准调用方式：

```
用户代码:  kernel[grid](args...)
  │
  ▼
NPULauncher.__call__()                    [driver.py:136]
  │  动态加载 .so launcher stub
  │  调用 self.launch(gridX, gridY, gridZ, stream, func, ...)
  ▼
Python C 扩展函数 launch()               [driver.py:1063 — 模板生成]
  │  PyArg_ParseTuple 解析 Python 参数
  │  提取 tensorShapes、tensorKinds
  │  调用 _launch(kernelName, func, stream, gridX, gridY, gridZ, ...)
  ▼
_launch() 内部函数                        [driver.py:946 — 模板生成]
  │  处理 workspace / syncBlockLock 分配
  │  组装 launch_args，调用 rtKernelLaunch()
  ▼
rtKernelLaunch() → NPU 硬件执行
```

### 5.2 路径 B：外部 C 接口路径（直接调用 triton_launch_kernel）

面向需要 C 级别 kernel 发射的场景：

```
第三方 C/C++ 代码
  │  dlopen / dlsym 获取 triton_launch_kernel 符号
  │  或直接链接 launcher stub .so
  ▼
triton_launch_kernel()                   [driver.py:800 — 模板生成]
  │  深拷贝 kernel_args
  │  处理 workspace / syncBlockLock 分配
  │  组装 launch_args，调用 rtKernelLaunch()
  ▼
rtKernelLaunch() → NPU 硬件执行
```

### 5.3 适用场景

| 场景 | 推荐路径 | 说明 |
|------|----------|------|
| 日常 Triton 算子开发 | 路径 A | 使用 `@triton.jit` 标准流程 |
| 自定义部署流水线 | 路径 B | 已有预注册的 kernel handle，绕过 JIT 前端 |
| 推理引擎集成 | 路径 B | 引擎需要 C 级别 kernel 发射接口 |
| 同签名 kernel 复用 | 路径 B | 用同一 launcher stub 发射不同编译产物 |

## 6. 最小调用示例

### 6.1 C/C++ 示例（已有 func 句柄后）

以下示例假设调用方已通过 CANN runtime 完成 kernel binary 的注册并持有可用的 `func` handle，仅展示调用 `triton_launch_kernel` 的参数组织方式。kernel 注册流程（`rtDevBinaryRegister` / `rtFunctionRegister`）属于 CANN runtime 标准流程，不在本示例范围内。

```c
#include <cstring>
#include <dlfcn.h>
#include <vector>
#include "rt.h"  // CANN runtime 头文件

// 函数指针类型定义（与 launcher stub 导出的签名一致）
typedef void (*triton_launch_kernel_t)(
    const char* kernelName,
    const void* func,
    rtStream_t stream,
    int gridX, int gridY, int gridZ,
    const int64_t* shapes_data,
    const int* shape_dims,
    int num_tensors,
    const int* tensor_kinds,
    const void* const* kernel_args,
    const size_t* arg_sizes,
    int num_args
);

void launch_kernel_via_stub(
    const char* stub_so_path,
    const char* kernel_name,
    const void* func,          // 已通过 CANN runtime 注册完成的 kernel function handle
    rtStream_t stream,
    int grid_x, int grid_y, int grid_z)
{
    // 1. 加载 launcher stub 并获取 triton_launch_kernel 符号
    void* handle = dlopen(stub_so_path, RTLD_LAZY);
    if (!handle) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return;
    }
    auto launch_fn = (triton_launch_kernel_t)dlsym(handle, "triton_launch_kernel");
    if (!launch_fn) {
        fprintf(stderr, "dlsym failed: %s\n", dlerror());
        dlclose(handle);
        return;
    }

    // 2. 组装 kernel 参数
    float alpha = 1.0f;
    int N = 1024;
    const void* arg_ptrs[] = { &alpha, &N };
    const size_t arg_sizes[] = { sizeof(float), sizeof(int) };

    // 3. shapes_data / tensor_kinds（仅用于 profiling，可为空）
    //    此处以 2 个 tensor 为例：input shape [1, 1024]、output shape [1, 1024]
    const int64_t shapes[] = {1, 1024, 1, 1024};
    const int dims[] = {2, 2};
    const int kinds[] = {0, 1};  // INPUT, OUTPUT

    // 4. 调用 triton_launch_kernel
    launch_fn(
        kernel_name, func, stream,
        grid_x, grid_y, grid_z,
        shapes, dims, 2, kinds,
        arg_ptrs, arg_sizes, 2
    );

    // 5. 非 TaskQueue 模式下可在此处同步；TaskQueue 模式默认启用，
    //    函数已提交任务到队列并返回，同步由调用方自行管理
    dlclose(handle);
}
```

> **说明：** 示例中的 `func` 参数是 CANN runtime 注册后的 kernel function handle，获取方式（`rtDevBinaryRegister` / `rtFunctionRegister`）属于 CANN runtime 标准流程。当前仓内 `npu_utils.cpp` 中的 `loadKernelBinary` / `registerKernel` 函数提供了 Python 侧的封装实现，具体 API 签名请以 CANN 头文件（`runtime/kernel.h`）为准。关于这些辅助函数是否可作为公开 API 文档化，请参见[待确认事项](#待确认事项)。

### 6.2 Python ctypes 示例

通过 ctypes 直接调用 launcher stub 中的 `triton_launch_kernel`：

```python
import ctypes
from pathlib import Path

# 假设 launcher stub .so 路径已知（可通过 NPULauncher.get_launcher_so_path() 获取）
stub_path = Path("/path/to/launcher_cxx11abi1.cpython-39-aarch64-linux-gnu.so")

lib = ctypes.CDLL(str(stub_path))

# 定义函数签名
lib.triton_launch_kernel.argtypes = [
    ctypes.c_char_p,           # kernelName
    ctypes.c_void_p,           # func
    ctypes.c_void_p,           # stream (rtStream_t)
    ctypes.c_int,              # gridX
    ctypes.c_int,              # gridY
    ctypes.c_int,              # gridZ
    ctypes.POINTER(ctypes.c_int64), # shapes_data
    ctypes.POINTER(ctypes.c_int),   # shape_dims
    ctypes.c_int,              # num_tensors
    ctypes.POINTER(ctypes.c_int),   # tensor_kinds
    ctypes.POINTER(ctypes.c_void_p),# kernel_args
    ctypes.POINTER(ctypes.c_size_t),# arg_sizes
    ctypes.c_int,              # num_args
]
lib.triton_launch_kernel.restype = None  # void

# 组装参数（示例省略 func handle 获取步骤，
# 实际需通过 CANN runtime Python 绑定获取）
# ...
# lib.triton_launch_kernel(kernel_name, func, stream, ...)
```

### 6.3 标准 Python JIT 间接调用（对照）

多数用户无需直接调用 `triton_launch_kernel`，使用标准 `@triton.jit` 即可：

```python
import triton
import torch

@triton.jit
def add_kernel(x_ptr, y_ptr, output_ptr, n_elements, BLOCK_SIZE: tl.constexpr):
    pid = tl.program_id(axis=0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    output = x + y
    tl.store(output_ptr + offsets, output, mask=mask)

x = torch.randn(1024, device='npu')
y = torch.randn(1024, device='npu')
output = torch.empty_like(x)
grid = lambda meta: (triton.cdiv(1024, meta['BLOCK_SIZE']),)
add_kernel[grid](x, y, output, 1024, BLOCK_SIZE=256)
# 此调用经过路径 A：NPULauncher → launch() → _launch() → rtKernelLaunch()
# 不经过 triton_launch_kernel()
```

## 7. 环境变量影响

以下环境变量影响 `triton_launch_kernel` 的运行时行为。默认值以 `release/3.2.2` 分支源码为准。

| 环境变量 | 默认值 | 源码位置 | 影响 |
|----------|--------|----------|------|
| `TRITON_COMPILE_ONLY` | `"false"` | `driver.py:109` | 为 `"true"` 时跳过 kernel 发射，仅编译 |
| `TRITON_DEVICE_PRINT` | `"false"` | `driver.py:452-453` | 为 `"true"` 时启用 device printf，影响 launch_args 中 DTData 槽位的有无和 CANN DebugTunnel 的开关 |
| `TRITON_ENABLE_TASKQUEUE` | `"true"` | `driver.py:454-455` | 为 `"true"` 时启用异步 TaskQueue 模式，函数提交任务后立即返回；为 `"false"` 时同步等待 kernel 执行完成 |
| `TRITON_GRID_WARN_PRINT` | `"false"` | `driver.py:456-457` | 为 `"true"` 时，若 `blockNum` 超过物理核数则向 stderr 输出性能警告 |

> **注意：** 以上默认值仅在 `release/3.2.2` 分支中验证。其他分支或版本可能有不同默认值，请以实际源码为准。

## 8. 限制与注意事项

### 平台限制

- 仅支持昇腾 NPU 平台（Atlas A2/A3 系列），不支持 GPU

### 返回值与错误处理

- 函数返回类型为 `void`，**不提供稳定错误码契约**
- 参数校验失败时静默返回（如 `num_args > 0` 但 `kernel_args` 为 `nullptr`）
- 资源分配失败时向 stderr 输出错误信息并返回
- `rtKernelLaunch` 的返回码仅在函数内部消费，不向调用方传播
- 调用方若需确认 kernel 执行状态，应：
  - TaskQueue 禁用模式下：函数返回后 kernel 已执行完毕，通过 `rtStreamSynchronize` 等价物确认
  - TaskQueue 启用模式（默认）：自行跟踪任务队列状态

### ABI 稳定性

- `triton_launch_kernel` 由 `generate_npu_wrapper_src()` 按 kernel 编译动态生成，每个 launcher stub `.so` 中的该函数其参数区内存布局可能不同
- 布局影响因素包括：kernel signature、`metadata.workspace_size`、`metadata.lock_num`、`metadata.coalesce_factor`、`metadata.coalesce_axis`、`metadata.force_simt_only`、`TRITON_DEVICE_PRINT`、ffts 支持状态
- **不同编译产物生成的 `triton_launch_kernel` 不可互换使用**；调用方应确保使用的 launcher stub `.so` 与目标 kernel 编译产物匹配

### grid 语义

- 后端以 `blockNum = gridX × gridY × gridZ` 计算总 block 数，以此数量调用 `rtKernelLaunch`
- 三个 grid 值均写入 kernel 参数区的 grid 槽位，kernel 内部可通过对应偏移读取
- 当 `coalesce_factor > 1` 且 `coalesce_axis` 合法（0/1/2）时，对应轴的 grid 维度在发射前执行整数除法（`gridX = gridX / H` 等）
- 若启用了 `auto_map_parallel_blocks`，`blockNum` 会被钳制到 `min(blockNum, 物理核数)`
- 当 `blockNum` 超过物理核数且 `TRITON_GRID_WARN_PRINT="true"` 时，向 stderr 输出性能警告

### 其他

- `kernel_args` 深拷贝增加 `sum(arg_sizes[i])` 字节的临时内存开销
- `shapes_data` / `shape_dims` / `tensor_kinds` 主要用于 msprof tensor 信息上报；不传（设为 `nullptr`）不影响 kernel 正确性，但 profiler 中 tensor 信息将缺失
- 该接口生成的代码依赖 CANN 运行时版本，升级 CANN 后需重新编译 launcher stub

---
