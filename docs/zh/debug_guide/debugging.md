# Triton-Ascend 调试指南

## 1 引言

本文档为 **Triton-Ascend 调试指南**，面向参与 Triton 与昇腾（Ascend）NPU 适配开发的工程师，系统性地介绍在 Triton-Ascend 编译与运行过程中常用的调试方法与工具。

全文内容概览如下：

| 章节 | 主要内容 |
|------|--------|
| **1. 概述** | 说明调试的核心目标（聚焦 `ttir.mlir` → `ttadapter.mlir` 转换），并对常见问题进行分类指引。 |
| **2. 编译流程概览** | 介绍 Triton-Ascend 端到端编译链的关键阶段，为后续调试提供上下文基础。 |
| **3. 临时文件指引** | 详解编译过程中生成的中间文件（如 `.mlir`、`.ll`、`.o` 等）的存储位置与用途，便于人工检查。 |
| **4. 解释器模式** | 介绍如何通过 `TRITON_INTERPRET=1` 在 CPU 上运行 kernel，作为 NPU 计算结果的精度基准。 |
| **5. 调试方法** | 提供多种实用调试手段：<br>• 静态/运行时打印<br>• 编译错误调试方法<br> |
| **附录 A** | 常用环境变量速查表，提升调试效率。 |

建议开发者结合具体问题，按需查阅对应章节，以高效定位并解决 Triton-Ascend 集成中的各类异常。

### 1.1 Triton-Ascend 常见问题分类与调试指引

在开发过程中，问题通常可归纳为以下几类。下表提供了快速的问题类型辨识与首选调试方法指引。

| 问题类型 | 典型表现/描述 | 推荐的首要调试方法 |
| :--- | :--- | :--- |
| **精度问题** | NPU运行结果与标杆参考结果（如PyTorch或Triton CPU解释器）存在差异。 | 4. 解释器模式 <br> 5.1 打印调试方法 |
| **编译错误 (MLIRCompileError)** | 在编译转换阶段失败，通常在Python端抛出 `MLIRCompileError`。 | 5.2 编译错误调试方法 |

## 2 Triton-Ascend 编译流程概览

理解完整的编译链是进行有效调试的基础。Triton-Ascend 的编译过程遵循以下主要阶段：

| 阶段 | 输入 | 输出 | 工具/组件 | 说明 |
| :--- | :--- | :--- | :--- | :--- |
| **Python Kernel编译** | `triton_kernel.py` (Python) | `ttir.mlir` (MLIR) | Triton JIT 编译器 | 将用户编写的Triton Python kernel编译为标准Triton IR (TTIR)。 |
| **Triton IR 适配转换** | `ttir.mlir` | `ttadapter.mlir` | 适配Ascend的Triton后端 | **关键调试阶段**。将TTIR转换为面向Ascend NPU后端的适配器IR。 |
| **MLIR 编译与代码生成** | `ttadapter.mlir` | `.o` (可执行对象文件) | 毕昇编译器 (`bishengir-compile`) | 将适配器IR进一步编译并优化，生成可在NPU上执行的二进制代码。 |

```bash
# Triton-Ascend 编译流程示意
[Python Kernel]
     ↓ (triton.compile)
[ttir.mlir]
     ↓        │ (TRITON_DEBUG=1 → ~/.triton/dump/)
[ttadapter.mlir]
     ↓ (bishengir-compile)
[NPU 可执行文件 .o]
```

**本指南的调试重点**集中在第二阶段：`ttir.mlir` → `ttadapter.mlir` 的转换过程，此阶段是 Triton-Ascend 的主要功能。

## 3 Triton-Ascend 临时文件指引

在 Triton-Ascend 的编译过程中，系统会生成多种临时文件用于缓存和调试。理解这些文件的位置和用途对于高效调试至关重要。

### 3.1 缓存文件（Cache）

Triton 使用缓存机制来加速重复编译过程。编译过程中生成的中间文件会被缓存在用户目录下，避免重复编译相同的 kernel。

缓存目录结构：

- 默认路径: ~/.triton/cache/

主要缓存内容:

- 输入文件缓存: 原始 Triton kernel 生成的 ttir.mlir 文件

- 输出文件缓存: 经过适配Ascend转换后的 ttadapter.mlir 文件

- 编译产物缓存: 最终编译生成的可执行文件

缓存文件命名约定：
缓存文件通常以 MD5 哈希值命名，确保相同的 kernel 代码对应相同的缓存文件。

**缓存管理建议：**

定期清理: 缓存可能占用较多磁盘空间，可定期清理：

```bash
rm -rf ~/.triton/cache
```

调试时禁用缓存: 在调试编译问题时，建议临时禁用缓存以确保每次都重新编译：

```bash
export TRITON_DISABLE_CACHE=1
```

缓存验证: 当怀疑缓存导致问题时，可删除相关缓存文件后重新测试。

### 3.2 调试转储文件（Dump Files）

通过设置环境变量 TRITON_DEBUG=1，可以在编译过程中将中间表示文件转储到磁盘，这些文件是调试编译问题的关键资源。

转储目录结构：

- 默认路径: ~/.triton/dump/

目录命名: 每个编译会话会生成一个以时间戳或唯一ID命名的子目录

主要转储文件:

- kernel.ttir.mlir: Triton IR 文件（编译输入）

- kernel.ttadapter.mlir: 适配器 IR 文件（转换输出）

启用调试转储：
即使启用缓存，只要设置了 TRITON_DEBUG=1，系统仍会在每次运行时重新生成转储文件（覆盖同名目录中的文件）。但若缓存命中且跳过编译，则可能不会触发 IR 转换，导致无新 dump 生成。因此调试时建议同时设置：

```bash
# 在运行 Triton 程序前设置环境变量
export TRITON_DEBUG=1
export TRITON_DISABLE_CACHE=1

# 运行 Triton kernel
python your_triton_program.py
```

### 3.3 文件生命周期管理

了解这些临时文件的生成时机和清理策略有助于更好地管理调试环境：

文件生成时机表：

| 文件类型 | 生成阶段 | 触发条件 | 清理建议 |
|----------|----------|----------|----------|
| 缓存文件 | 每次编译执行时 | 缓存未命中时生成 | 定期清理或问题排查时清除 |
| 转储文件 | 设置 TRITON_DEBUG=1 后 | 每次编译都会生成 | 调试结束后手动清理 |

- 生产环境中应禁用调试转储（不设置 TRITON_DEBUG=1）

- 缓存机制可以显著提升性能，不应轻易禁用

通过合理利用这些临时文件，开发者可以更高效地定位和解决 Triton-Ascend 在编译过程中遇到的问题。

### 3.4 IR文件解析

以示范测试用例  [01-vector-add.py](../../../third_party/ascend/tutorials/01-vector-add.py#) 举例说明编译流程：
这是一个简单的两个tensor的加法计算，计算逻辑请参考示范用例中的注解。
通过TRITON_DEBUG=1开启dump文件输出，设置TRITON_DISABLE_CACHE=1禁用缓存确保重新编译，可以获取到 kernel.ttir.mlir 和 kernel.ttadapter.mlir 

- 运行用例

```python
TRITON_DEBUG=1 TRITON_DISABLE_CACHE=1 python 01-vector-add.py
```

运行用例后会打印dump文件路径，默认是 ~/.triton/dump，显示如下：

```python
Dumping intermediate results to ~/.triton/dump/xxx 
# xxx是一串hash的唯一标识符
```

进入该dump路径，查看 kernel.ttir.mlir 和 kernel.ttadapter.mlir 

#### 3.4.1 TTIR（Triton Intermediate Representation）

- TTIR 样例
查看 kernel.ttir.mlir 如下：

```python
module {
  tt.func public @add_kernel(%arg0: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg1: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg2: !tt.ptr<f32> {tt.divisibility = 16 : i32} , %arg3: i32 {tt.divisibility = 16 : i32} ) attributes {noinline = false} {
    %cst = arith.constant dense<0.000000e+00> : tensor<1024xf32> loc(#loc1)
    %c1024_i32 = arith.constant 1024 : i32 loc(#loc1)
    %0 = tt.get_program_id x : i32 loc(#loc2)
    %1 = arith.muli %0, %c1024_i32 : i32 loc(#loc3)
    %2 = tt.make_range {end = 1024 : i32, start = 0 : i32} : tensor<1024xi32> loc(#loc4)
    %3 = tt.splat %1 : i32 -> tensor<1024xi32> loc(#loc5)
    %4 = arith.addi %3, %2 : tensor<1024xi32> loc(#loc5)
    %5 = tt.splat %arg3 : i32 -> tensor<1024xi32> loc(#loc6)
    %6 = arith.cmpi slt, %4, %5 : tensor<1024xi32> loc(#loc6)
    %7 = tt.splat %arg0 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc7)
    %8 = tt.addptr %7, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc7)
    %9 = tt.load %8, %6, %cst : tensor<1024x!tt.ptr<f32>> loc(#loc8)
    %10 = tt.splat %arg1 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc9)
    %11 = tt.addptr %10, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc9)
    %12 = tt.load %11, %6, %cst : tensor<1024x!tt.ptr<f32>> loc(#loc10)
    %13 = arith.addf %9, %12 : tensor<1024xf32> loc(#loc11)
    %14 = tt.splat %arg2 : !tt.ptr<f32> -> tensor<1024x!tt.ptr<f32>> loc(#loc12)
    %15 = tt.addptr %14, %4 : tensor<1024x!tt.ptr<f32>>, tensor<1024xi32> loc(#loc12)
    tt.store %15, %13, %6 : tensor<1024x!tt.ptr<f32>> loc(#loc13)
    tt.return loc(#loc14)}}
```

- TTIR 解析

TTIR 是 Triton 编译器前端生成的中间表示（Intermediate Representation），它以 MLIR（Multi-Level IR）格式表达，保留了原始 Triton Python 内核的语义结构。在 `kernel.ttir.mlir` 中：

- 函数 `@add_kernel` 接收三个指针参数（分别对应输入 A、B 和输出 C 的设备内存地址）以及一个整型参数 `n` 表示向量长度。
- 每个Triton program (向量化执行单元) 处理 1024 个元素（由 `%c1024_i32` 常量体现），通过 `tt.get_program_id x` 获取当前 block 的 ID，并计算其全局偏移。
- 使用 `tt.make_range` 与 `tt.splat` 构造 SIMD 风格的索引张量，配合 `arith.addi` 生成每个线程处理的全局地址偏移。
- 通过 `tt.addptr` 和 `tt.load` 实现向量化加载，使用掩码 `%6`（由 `arith.cmpi slt` 生成）防止越界访问。
- 执行逐元素浮点加法 `arith.addf`，再通过 `tt.store` 将结果写回全局内存。

TTIR 层面仍基于 Triton 原生抽象（如 `!tt.ptr<f32>`、`tt.load`/`tt.store` 等），尚未映射到底层硬件的具体内存模型或执行单元，是平台无关的高层次 IR。

#### 3.4.1 TTAdapter IR（Target-Specific Adapter Representation）

- TTAdapter IR 样例
查看 kernel.ttadapter.mlir 如下：

```python
module {
  func.func @add_kernel(%arg0: memref<?xi8>, %arg1: memref<?xi8>, %arg2: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg3: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 0 : i32}, %arg4: memref<?xf32> {tt.divisibility = 16 : i32, tt.tensor_kind = 1 : i32}, %arg5: i32 {tt.divisibility = 16 : i32}, %arg6: i32, %arg7: i32, %arg8: i32, %arg9: i32, %arg10: i32, %arg11: i32) attributes {SyncBlockLockArgIdx = 0 : i64, WorkspaceArgIdx = 1 : i64, global_kernel = "local", mix_mode = "aiv", parallel_mode = "simd"} {
    %cst = arith.constant 0.000000e+00 : f32
    %c1024 = arith.constant 1024 : index
    %c1024_i32 = arith.constant 1024 : i32
    %0 = arith.muli %arg9, %c1024_i32 : i32
    %1 = arith.index_cast %0 : i32 to index
    %reinterpret_cast = memref.reinterpret_cast %arg2 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc = memref.alloc() : memref<1024xf32>
    %2 = arith.addi %1, %c1024 : index
    %3 = arith.index_cast %arg5 : i32 to index
    %4 = arith.maxsi %1, %3 : index
    %5 = arith.minsi %2, %4 : index
    %6 = arith.subi %5, %1 : index
    %7 = arith.cmpi slt, %6, %c1024 : index
    scf.if %7 {
      linalg.fill ins(%cst : f32) outs(%alloc : memref<1024xf32>)
    } {hivm.unlikely_condition}
    %subview = memref.subview %reinterpret_cast[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_0 = memref.subview %alloc[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview, %subview_0 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %8 = bufferization.to_tensor %alloc restrict writable : memref<1024xf32>
    %reinterpret_cast_1 = memref.reinterpret_cast %arg3 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %alloc_2 = memref.alloc() : memref<1024xf32>
    scf.if %7 {
      linalg.fill ins(%cst : f32) outs(%alloc_2 : memref<1024xf32>)
    } {hivm.unlikely_condition}
    %subview_3 = memref.subview %reinterpret_cast_1[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    %subview_4 = memref.subview %alloc_2[0] [%6] [1] : memref<1024xf32> to memref<?xf32, strided<[1]>>
    memref.copy %subview_3, %subview_4 : memref<?xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1]>>
    %9 = bufferization.to_tensor %alloc_2 restrict writable : memref<1024xf32>
    %10 = arith.addf %8, %9 : tensor<1024xf32>
    %reinterpret_cast_5 = memref.reinterpret_cast %arg4 to offset: [%1], sizes: [1024], strides: [1] : memref<?xf32> to memref<1024xf32, strided<[1], offset: ?>>
    %extracted_slice = tensor.extract_slice %10[0] [%6] [1] : tensor<1024xf32> to tensor<?xf32>
    %subview_6 = memref.subview %reinterpret_cast_5[0] [%6] [1] : memref<1024xf32, strided<[1], offset: ?>> to memref<?xf32, strided<[1], offset: ?>>
    bufferization.materialize_in_destination %extracted_slice in writable %subview_6 : (tensor<?xf32>, memref<?xf32, strided<[1], offset: ?>>) -> ()
    return
  }
}
```

- TTAdapter IR 解析

TTAdapter IR 是 Triton-Ascend 编译流程中将 TTIR 转换为适配昇腾 NPU 架构的中间表示，采用标准 MLIR dialect（如 `memref`、`linalg`、`scf` 等），并引入 NPU 特定的约束和优化策略。在 `kernel.ttadapter.mlir` 中：

- 函数签名已从 Triton 指针类型转换为 `memref<?xi8>` 或带属性的 `memref<?xf32>`，其中 `tt.divisibility = 16` 表示内存对齐要求，`tt.tensor_kind` 区分输入（0）与输出（1）。
- 全局偏移通过 `memref.reinterpret_cast` 重构为固定大小（1024）的局部视图，便于后续向量化处理。
- 引入边界检查逻辑：计算有效元素数量 `%6`，并通过 `scf.if` 控制是否对尾部填充零（`linalg.fill`），确保 SIMD 宽度对齐且不越界。
- 使用 `memref.alloc` 分配本地暂存 buffer，通过 `memref.copy` 将全局内存数据安全拷贝至本地，再经 `bufferization.to_tensor` 转为 tensor 供算子使用。
- 加法操作由 `arith.addf` 在 tensor 上执行，结果通过 `tensor.extract_slice` 截取有效部分，并用 `bufferization.materialize_in_destination` 写回目标 memref。

TTAdapter IR 已完成从 Triton 抽象到适配昇腾 NPU 的格式。

## 4 解释器模式

解释器的核心价值在于**隔离硬件差异**。通过环境变量 `TRITON_INTERPRET=1` 强制Triton在CPU上执行kernel计算，其结果可作为判断NPU计算精度的基准。

**使用方法：**

1. 设置环境变量`TRITON_INTERPRET=1`并运行程序，使Triton kernel在CPU解释器上执行。
2. 在Triton kernel源码中需要检查的位置插入Python断点：

    ```python
    breakpoint()  # Python 内置断点函数
    ```

3. 程序执行到此处会暂停并进入Python调试器 (`Pdb`)。可以打印和检查任意中间变量的值：

    ```python
    (Pdb) p tmp0  # 打印变量 tmp0 的值
    ```

- 注：解释器模式会在 CPU 上执行所有计算，显著降低运行效率。因此，在完成调试或验证后，务必取消设置环境变量 TRITON_INTERPRET，或显式将其设为 0，确保系统性能不受影响：

```bash
# 取消该环境变量
unset TRITON_INTERPRET

# 显式将其设为 0
export TRITON_INTERPRET=0
```

## 5 调试方法

### 5.1 打印调试方法

### 5.1.1 静态打印调试方法

此方法使用 `tl.static_print` 在编译时打印常量表达式的值，适用于调试编译时已知的配置参数和常量。

设置环境变量 `TRITON_DEVICE_PRINT=1` 可启用 `tl.static_print` 功能。此函数允许在 kernel 编译阶段打印常量值，是验证配置参数和常量表达式的有效方法。

特性说明：

- tl.static_print 在 编译时 执行，而不是运行时

- 只能打印编译时常量（tl.constexpr 参数、常量表达式）

- 输出显示在编译器的标准输出中

使用方法：

1.在 Triton kernel 中，为需要调试的常量参数添加 tl.static_print 语句

```python
import triton.language as tl

@triton.jit
def triton_kernel(
    out_ptr0, 
    in_ptr0, 
    in_ptr1, 
    XBLOCK: tl.constexpr,  # 编译时常量参数
    USE_FP16: tl.constexpr  # 编译时常量参数
):
    # 打印编译时常量参数
    tl.static_print("XBLOCK = ", XBLOCK)
    tl.static_print("USE_FP16 = ", USE_FP16)
    
    idx = tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + idx)
    tmp1 = tl.load(in_ptr1 + idx)
    
    # 打印常量计算结果
    elements_per_thread = XBLOCK // 32
    tl.static_print("Elements per thread = ", elements_per_thread)
    
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + idx, tmp2)
```

2.设置环境变量并运行程序进行编译

```bash
# 启用 Triton 调试输出（包含 static_print）
export TRITON_DEVICE_PRINT=1

# 运行 Python 程序，会在编译阶段看到打印输出
python your_program.py
```

### 5.1.2 运行时调试方法

此方法的使用 `tl.device_print` 可以灵活打印需要观察的变量的值。
设置环境变量 `TRITON_DEVICE_PRINT=1` 可启用 `tl.device_print` 功能。此函数允许在kernel内部打印张量值，是分阶段验证计算精度的高效方法。
**使用方法：**

1. 在Triton kernel中，为需要打印的变量增加 `tl.device_print` 的语句。

```python
import triton.language as tl

@triton.jit
def triton_kernel(out_ptr0, in_ptr0, in_ptr1, XBLOCK: tl.constexpr):
    idx = tl.arange(0, XBLOCK)
    tmp0 = tl.load(in_ptr0 + idx)
    tmp1 = tl.load(in_ptr1 + idx)
    tmp2 = tmp0 + tmp1
    tl.device_print("tmp2 after addition = ", tmp2)  # 打印中间结果
    tl.store(out_ptr0 + idx, tmp2)
```

2.设置环境变量`TRITON_DEVICE_PRINT=1`并运行程序，窗口将打印出该变量的值。

```bash
# 启用 Triton 调试输出（包含 device_print）
export TRITON_DEVICE_PRINT=1

# 运行 Python 程序，会在编译阶段看到打印输出
python your_program.py
```

- 注：打印长度限制：
tl.device_print 在张量打印有长度限制，具体表现为：当张量长度超过一定阈值时，输出会被截断

### 5.1.3 对比两种打印方法

| 特性 | `tl.device_print` | `tl.static_print` |
|------|-------------------|-------------------|
| **执行时机** | 运行时（kernel 执行时） | 编译时（kernel 编译时） |
| **输出位置** | 运行时标准输出 | 编译器标准输出 |
| **可打印内容** | 运行时张量值、变量 | 编译时常量、常量表达式 |
| **性能影响** | 有运行时开销 | 无运行时开销 |
| **启用环境变量** | `TRITON_DEVICE_PRINT=1` | `TRITON_DEVICE_PRINT=1` |

环境变量说明：

TRITON_DEVICE_PRINT=1：启用运行时打印，同时也会启用编译时打印

TRITON_DEBUG=1：启用所有调试输出（包括编译时和运行时打印）

### 5.2 编译错误调试方法

当 `ttir.mlir` → `ttadapter.mlir` 的转换过程失败，无法生成`ttadapter.mlir`，报错`MLIRCompileError`.
需要进入 Triton-Ascend 代码层面定位问题。Triton-Ascend 包含 Python 和 C++ 代码层，需根据报错日志中的调用栈信息，定位到具体的报错代码片段，并采用相应的调试方法。

### 5.2.1 Python 代码调试方法

当调用栈信息显示错误源自 Triton-Ascend 的 Python 层代码时，可以使用 Python 内置的调试器 pdb 进行交互式调试。pdb 允许你设置断点、单步执行、检查变量状态，是定位 Python 代码逻辑错误的有效工具。

使用步骤：

定位问题入口
在报错日志中找到最接近用户代码的 Python 调用栈信息，通常位于栈顶附近，例如：

```text
File "/path/to/triton/ascend/compiler.py", line 123, in compile_fn
    result = lower_function(...)
```

插入调试断点
在怀疑出错的 Python 源文件中插入 pdb 断点。

```python
def compile_fn(ttir):
    import pdb; pdb.set_trace()  # 兼容所有Python版本
```

**示例:**
假设在 `compiler.py` 的第 123 行设置了断点，程序暂停后：

```python
python
(Pdb) l  # 查看当前代码上下文
118     def compile_fn(ttir):
120         import pdb; pdb.set_trace()
121         # 检查输入参数
122         print(f"ttir type: {type(ttir)}")
123         result = lower_function(ttir)  # <-- 当前暂停位置

(Pdb) p ttir  # 检查输入参数
(Pdb) n  # 单步执行到下一行
(Pdb) p result  # 查看结果
```

### 5.2.2 环境变量调试方法

在开发或调试 Triton 算子时，可通过设置以下环境变量启用不同阶段的中间表示（IR）打印，帮助定位问题。以下是两个关键调试开关的详细说明。

#### 5.2.2.1 `MLIR_ENABLE_DUMP=1`

**功能**
启用 **MLIR 高层 IR 的自动 dump**。在每个 MLIR Pass 执行前后，将当前函数的 IR 以可读文本形式输出到 `stderr`。

**特点**
日志量小：通常几十至几百行，易于阅读
聚焦高层逻辑：适用于调试算子转换、内存布局、并行策略等

**使用建议**
日常调试首选：90% 的 Triton 算子问题可通过此日志定位
配合 `TRITON_DEBUG=1` 可进一步增强信息

**启用方式**

```bash
export MLIR_ENABLE_DUMP=1
export TRITON_DEBUG=1
python your_triton_script.py
```

#### 5.2.2.2 `TRITON_ENABLE_LLVM_DEBUG=1`

**功能**
启用 LLVM 后端 CodeGen 阶段的全量调试日志，包括指令选择、寄存器分配、指令调度、机器码生成等底层过程。

**特点**
日志量极大：单个 kernel 可产生数万行输出
极底层细节：包含寄存器名、物理/虚拟寄存器映射、栈帧布局等
仅限 LLVM 专家：对普通 Triton 开发者通常为“噪声”

**使用建议**
仅在怀疑 LLVM 后端 bug 时启用（如生成非法指令、性能异常）
配合 LLVM_DEBUG_ONLY 限制输出范围

在启用 `TRITON_ENABLE_LLVM_DEBUG=1` 时，可通过 `LLVM_DEBUG_ONLY` 环境变量指定仅输出特定模块的调试日志。以下是常用 `DEBUG_TYPE` 的简要解释：

```bash
## `isel`（Instruction Selection）
- **作用**：将 LLVM IR 指令转换为目标架构的机器指令（MachineInstr）。
- **调试内容**：显示 IR → 机器指令的映射过程、模式匹配结果。
- **适用场景**：怀疑指令选择错误（如生成了非法指令或低效指令序列）。

## `regalloc`（Register Allocation）
- **作用**：为虚拟寄存器分配物理寄存器，并处理溢出（spilling）。
- **调试内容**：寄存器分配前后状态、冲突图、活跃区间分析。
- **适用场景**：寄存器压力大、性能下降、或出现意外的内存访问。

## `spiller`（Spiller）
- **作用**：当寄存器不足时，将部分值“溢出”到栈内存。
- **调试内容**：哪些虚拟寄存器被 spill、插入的 load/store 指令位置。
- **适用场景**：性能因频繁访存下降，需优化寄存器使用。

## `peephole`（Peephole Optimizer）
- **作用**：在机器码层面进行局部优化（如常量折叠、冗余指令消除）。
- **调试内容**：优化前后的指令对比。
- **适用场景**：生成代码存在明显冗余，但高层优化未覆盖。

## `asm-printer`（Assembly Printer）
- **作用**：将 MachineInstr 转换为最终汇编文本（如 PTX、AMDGCN、CCE）。
- **调试内容**：生成的汇编代码、符号引用、指令编码。
- **适用场景**：汇编语法错误、标签不匹配、或需要查看最终输出。
```

**启用方式**
以指定仅输出`isel`为例

```bash
export TRITON_ENABLE_LLVM_DEBUG=1
export LLVM_DEBUG_ONLY="isel"
python your_triton_script.py
```

**调试流程推荐**
先启用 `MLIR_ENABLE_DUMP=1`
→ 验证 MLIR 层转换是否正确（如 ReduceOp → scf.for）
若 MLIR 正常但结果错误
→ 怀疑 LLVM 问题，再启用 `TRITON_ENABLE_LLVM_DEBUG=1 + LLVM_DEBUG_ONLY`
避免直接开启 `TRITON_ENABLE_LLVM_DEBUG=1`
→ 日志过大易掩盖关键信息，且严重影响运行速度

## 附录 A：常用环境变量速查表

| 变量                      | 作用                             |
|--------------------------|----------------------------------|
| `TRITON_DEBUG=1`         | 启用中间 IR 转储                 |
| `TRITON_DISABLE_CACHE=1` | 禁用编译缓存                     |
| `TRITON_INTERPRET=1`     | 使用 CPU 解释器执行 kernel       |
| `TRITON_DEVICE_PRINT=1`  | 启用运行时打印输出，同时也会启用编译时打印输出      |
| `MLIR_ENABLE_DUMP=1`  | 启用 MLIR 高层 IR 的自动 dump。在每个 MLIR Pass 执行前后，将当前函数的 IR 以可读文本形式输出 |
| `TRITON_ENABLE_LLVM_DEBUG=1`  | 启用 LLVM 后端 CodeGen 阶段的全量调试日志，包括指令选择、寄存器分配、指令调度、机器码生成等底层过程 |
