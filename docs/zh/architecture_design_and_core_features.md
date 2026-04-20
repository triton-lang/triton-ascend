# 架构设计与核心特性

## 1.逻辑架构

![图像](./figures/architectural_diagram.png)
**Triton-Ascend 架构说明**

**核心组件：**

- **`Ascend language extension`**：适配 Ascend 的 Triton 语言扩展  
- **`compiler`**：适配 Ascend 的 Triton 编译器  
- **`driver`**：适配 Ascend 的设备驱动接口

**组件功能：**

- **`Ascend language extension`**  
  在标准 Triton 语言基础上，引入针对 Ascend NPU 架构的语法与语义扩展。

- **`compiler`**  
  接收来自上层 Triton compiler 生成的中间表示文件 `TTIR`（Triton IR），执行一系列适配昇腾硬件的转换。

  ```python
  Triton IR → Linalg IR → AscendNPU IR → triton_xxx_kernel.o
  ```

  Triton IR 转换为 Linalg IR，再经 BiSheng Compiler 生成面向 Ascend NPU 的可执行二进制文件  `triton_xxx_kernel.o`。

- **`driver`**  
  提供 Triton 运行时与 Ascend 软件栈（CANN）之间的对接能力， 加载由 BiSheng Compiler 生成的设备侧可执行内核 `triton_xxx_kernel.o` 。

## 2.代码结构

### 2.1 代码结构原则

本项目在标准 Triton 基础上，扩展支持华为 Ascend NPU（通过 CANN 软件栈）。整体设计遵循以下**代码原则**：

> - **若修改与目标硬件无关**（target independent），应保留在 **Triton core** 部分（如language、runtime的通用修改）；  
> - **若修改与 Ascend 硬件强相关**（target affinitive），应放在 **Triton-Ascend** 中。

### 2.2 目录结构与功能说明

**`include/` 和 `lib/`**

- **内容**：包含针对 Ascend NPU 的 **MLIR Passes**、**Dialects**（方言）以及相关工具。
- **作用**：用于在 MLIR 编译流程中表达和优化 Ascend 特定的计算图。

**`libdevice.py`**

- **内容**：适配 Ascend NPU 的 `libdevice` 接口。
- **作用**：提供适配 Ascend NPU 硬件的底层实现支持，供 Triton 算子调用。

**`backend/compiler.py`**

- **内容**：`triton-ascend` 编译器主入口。
- **作用**：将 Triton 高层 DSL 代码编译为可在 Ascend NPU 上执行的**可执行二进制文件**（如 `.o`文件）。

**`backend/driver.py`**

- **内容**：`triton-ascend` 驱动模块。
- **作用**：加载并启动已编译的可执行二进制。

## 3. Modules

### 3.1 Triton core Enhancement

#### 3.1.1 Language expansion

| 序号 | 算子名称      | 描述        |
| :--- | :--------------------------------------------- | :--------------------------------- |
| 1    | `tl.insert_slice(full, src, offsets, sizes, strides)` | 按照指定的偏移量（offsets）、尺寸（sizes）和步幅（strides）参数，将一个张量插入到另一个张量中。<br>**返回值**：目标张量。<br>**full**：目标张量，源张量将被插入到此张量中。<br>**src**：源张量。<br>**offsets**：目标张量上的偏移量（整数元组）。<br>**sizes**：源张量上的尺寸（整数元组）。<br>**strides**：目标张量上的步幅（整数元组）。 |
| 2    | `tl.extract_slice(full, offsets, sizes, strides)`     | 按照指定的偏移量（offsets）、尺寸（sizes）和步幅（strides）参数，从另一个张量中提取一个切片张量。<br>**返回值**：切片张量。<br>**full**：源张量，从此张量中提取切片。<br>**offsets**：源张量上的偏移量（整数元组）。<br>**sizes**：切片张量的尺寸（整数元组）。<br>**strides**：源张量上的步幅（整数元组）。                        |
| 3    | `tl.get_element(source, offset)`                      | 读取一个具有维度的张量，并返回指定偏移量处的单个元素。<br>**source**：源张量。<br>**offset**：元素提取位置的偏移量（整数元组）。   |

### 3.2 Triton-Ascend

#### 3.2.1 Compiler Options

|序号| NPUOptions                                    | 硬件平台     | 用途 |
| --- | --------------------------------------------- | ---------- | ----- |
| 1   | multibuffer                                   | NPU        | Autotune Option: Enable or disable ping-pong pipeline. |
| 2   | enable_auto_bind_sub_block                    | NPU        | Autotune option (CV-fused kernels only): Enable or disable auto-binding of sub-blocks. |
| 3   | enable_hivm_auto_cv_balance                   | NPU        | Autotune option (CV-fused kernels only): Enable or disable automatic CV balancing. |
| 4   | sync_solver                                   | NPU        | Autotune option (CV-fused kernels only): Enable or disable the synchronization solver. |
| 5   | unit_flag                                     | NPU        | Autotune option: Enable or disable the sync unit flag. |
| 6   | inject_barrier_all                            | NPU        | Autotune option: Enable or disable automatic injection of barriers for all operations. |
| 7   | inject_block_all                              | NPU        | Autotune option: Enable or disable automatic injection of blocks for all operations. |
| 8   | limit_auto_multi_buffer_only_for_local_buffer | NPU        | Autotune option: Restrict automatic multi-buffering only to local buffers. |
| 9   | limit_auto_multi_buffer_of_local_buffer       | NPU        | Autotune option: Enable or disable automatic multi-buffering for local buffers. |
| 10  | set_workspace_multibuffer                     | NPU        | Autotune option: Enable or disable multi-buffering for the workspace. |
| 11  | tile_mix_vector_loop                          | NPU        | Autotune option (CV-fused kernels only): Enable or disable tiling for vector loops. |
| 12  | tile_mix_cube_loop                            | NPU        | Autotune option (CV-fused kernels only): Enable or disable tiling for cube loops. |
| 13  | disable_auto_inject_block_sync                | NPU        | Autotune option (CV-fused kernels only): Enable or disable automatic injection of block synchronizations. |
| 14  | stream                                        | NPU        | Optional: Inform the compiler about the NPU stream to use. |
| 15  | enable_linearize                              | NPU        | Autotune option: Enable or disable the linearization pass. |
| 16  | enable_nd2nz_on_vector                        | NPU        | Autotune option (CV-fused kernels only): Enable or disable the ND (n-dimensional) to NZ (non-zero) layout transformation. |
| 17  | auto_blockify_size                            | NPU        | Autotune option: Enable or disable AutoBlockify pass. It is ignored when TRITON_ALL_BLOCKS_PARALLEL is not set |

#### 3.2.2 SIMD compiler

| 序号 | Pass                   | 目的                                                                   | IR 转换                 |
| ------ | ---------------------- |----------------------------------------------------------------------| ----------------------- |
| 1      | triton-to-structured   | linearize                                                             | ttir->ttir              |
| 2      | triton-to-unstructured | convert indirect axis to loop                                        | ttir->ttir              |
| 3      | triton-to-linalg       | memory/reduction/view/creation/math/arith/linear algebra to linalgir | ttir->linalgir          |
| 4      | triton-to-other        | ttir->hivm/hfusion/llvm                                              | ttir->hivm/hfusion/llvm |

##### 3.2.2.1 TritonToStructured

处理指针表达式和mask表达式中的整除取余，通过升维的方法，去除整除取余后重新生成load/store 等OP。

| Converter                | 功能  | 局限性 |
| ------------------------ | -------------------------- | ------------------------- |
| RewriteAddPtrOp          | 分析 `tl.load`, `tl.store`等操作中的指针表达式 (`AddPtrOp`)。将原始的指针偏移计算分解并建模为包含各维度（轴）具体偏移信息的 `PtrState` 对象。例如，对于形如 `ptr + x // 1024 * 4096 + x % 1024 * 4 + y` 的表达式，分析出 `x` 和 `y` 轴的贡献与关系。                   | 1. 所涉及的原始迭代轴（如`x`）必须能被分裂轴（如`1024`）整除。<br>2. 外部的 `XBLOCK` 大小必须是分裂轴`divisor`的整数倍或其约数。                               |
| CreateAddpr              | 根据分析得到的 `PtrState` 对象，重新构造一个新的 `AddPtrOp` 指针计算操作。新生成的指针表达式将消除原表达式中的整数除法 (`//`) 和取模 (`%`) 操作。                                                                       | 依赖于 `RewriteAddPtrOp` 成功生成的、合法的 `PtrState`。                                                                                                     |
| RewriteLoadOp            | 分析 `tl.load` 操作中的掩码 (`mask`) 表达式。将包含整除/取余的复杂掩码条件分解并建模为包含各维度边界信息的 `MaskState` 对象。例如，对于 `mask = x // 1024 < 8 and x % 1024 < 1024 and y < 4`，分析出各维度的独立约束条件。                                                                     | 1. 所涉及的原始迭代轴（如`x`）必须能被分裂轴（如`1024`）整除。<br>2. 外部的 `XBLOCK` 大小必须是分裂轴`divisor`的整数倍或其约数。                               |
| BuildMask                | 根据分析得到的 `MaskState` 对象，重新构造一个新的掩码 (`mask`) 表达式。新掩码将消除原表达式中的整数除法 (`//`) 和取模 (`%`) 操作。                                                                                            | 仅处理由 `RewriteLoadOp` 或 `RewriteStoreOp` 生成的 `MaskState`。无法处理任意复杂的、非规范化的掩码表达式。                                                   |
| CreateLoad               | 使用由 `CreateAddpr` 生成的新指针表达式和由 `BuildMask` 生成的新掩码表达式，重新创建（替换）原始的 `tl.load` 操作，完成指令重写。                                                                                                                                                               | 依赖于 `RewriteAddPtrOp`, `CreateAddpr`, `RewriteLoadOp`, `BuildMask` 等前置步骤均成功执行。                                                                |
| RewriteStoreOp           | 分析 `tl.store` 操作中的掩码 (`mask`) 表达式。其功能与 `RewriteLoadOp` 类似，将包含整除/取余的复杂掩码条件分解并建模为 `MaskState` 对象。                                                                                                                                                       | 与 `RewriteLoadOp` 相同。                                                                                                                                    |
| CreateStore              | 使用由 `CreateAddpr` 生成的新指针表达式和由 `BuildMask` 生成的新掩码表达式，重新创建（替换）原始的 `tl.store` 操作，完成指令重写。                                                                                                                                                              | 依赖于 `RewriteAddPtrOp`, `CreateAddpr`, `RewriteStoreOp`, `BuildMask` 等前置步骤均成功执行。                                                               |
| RewriteAtomicRWMOp       | 处理原子读写修改操作（如 `atomic.add`, `atomic.max` 等）中的指针问题。                                                                                                                | 通常继承与 `RewriteAddPtrOp` 相同的局限性。对于某些特殊的、非连续或条件性的原子操作模式可能不支持。                                                           |
| RewriteAtomicCASOp       | 处理原子比较并交换操作 (`atomic.cas`) 中的指针线性化问题。分析其指针表达式，通过升维方法消除整除和取余操作，以匹配硬件原子指令的寻址要求。                                                                                                                                                      |                         |
| RewriteWhile             | 处理 `while` 循环体内的指针叠加操作。                                                          | 不支持循环体内包含条件分支 (`if`) 的复杂指针路径变换。                                         |
| RewriteFor               | 处理 `for` 循环体内的指针叠加操作。                        |                                              |

##### 3.2.2.2 TritonToUnstructured

| 序号 | Pass / 转换器                              | 描述  |
|------|-------------------------------------------|-----------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| 1    | discrete-mask-access-conversion           | 将Triton中基于离散索引掩码（Discrete Mask）的内存访问模式（如`triton.language.load`带非连续mask）进行分析与转换，为后续将离散轴展开为循环做准备。该Pass识别出那些无法被后端硬件高效处理的、非规律性的或稀疏的访问模式。 |
| 2    | triton-to-unstructured           | 将经过`discrete-mask-access-conversion`识别出的、包含离散轴（Discrete Axes）的张量操作，转换为基于显式标量循环的标量访存。 |
| 3    | bubble-up-operation                       | 主要对`extract op/extract_slice`顺序上移优化。这可以优化数据局部性，有些场景能消除转换后产生的不必要的循环，从而提升生成代码的执行效率。 |

###### 3.2.2.2.1 discrete-mask-access-conversion

| 转换器名称                  | 描述|
|----------------------------|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| DiscreteMaskStoreConversion | 首先进行mask分析，如果mask分析结果是非连续的，将原始的store操作转化为以下序列：<br>1. load（加载目标存储地址的内容）<br>2. select（根据mask挑选目标存储内容和待存储的value内容）<br>3. store（将select的结果存储回目标地址） |
| DiscreteMaskLoadConversion  | 首先进行mask分析，如果mask分析结果是非连续的，将原始的load操作转化为以下序列：<br>1. load（加载源tensor的所有内容）<br>2. select（根据mask挑选源tensor内容，被掩盖部分设置为other值）                     |
| DiscreteMaskAtomicAddConversion | 首先进行mask分析，如果mask分析结果是非连续的，将原始的atomic_add操作转化为以下序列：<br>1. select（根据mask挑选value的值，被掩盖部分设为0）<br>2. atomic_add（使用select后的结果重新生成atomic_add操作） |

###### 3.2.2.2.2 triton-to-unstructured

| TritonToUnstructured Converters | 描述 |
|---|---|
| UnstructuredMemAccessConverter\<triton::LoadOp\> | 将LoadOp转化为多重循环标量加载 |
| UnstructuredMemAccessConverter\<triton::StoreOp\> | 将StoreOp转化为多重循环标量存储 |
| UnstructuredMemAccessConverter\<triton::AtomicRMWOp\> | 将AtomicRMWOp转化为多重循环标量Atomic操作 |
| UnstructuredMemAccessConverter\<triton::AtomicCASOp\> | 将AtomicCASOp转化为多重循环标量Atomic操作 |

###### 3.2.2.2.3 bubble-up-operation

| 转换器名称 | 描述 |
|---|---|
| BubbleUpExtract\<tensor::ExtractOp\> | extract op顺序上移优化，在某些场景可以避免产生不必要的循环 |
| BubbleUpExtract\<tensor::ExtractSliceOp\> | extract op/extract_slice顺序上移优化，在某些场景可以避免产生不必要的循环 |

##### 3.2.2.3 TritonToLinalg

###### 3.2.2.3.1 triton-to-linalg

TritonToLinalg converts ttir to linalg ir.

| Converter                                  | 描述                                                  |
| ------------------------------------------ | ------------------------------------------------------------ |
| StoreConverter                             | triton::StoreOp to memref::copy                              |
| AddPtrConverter                            | triton::AddPtrOp to memref::ReinterpretCast                  |
| GetProgramIDConverter                      | triton::GetProgramIdOp to a param in functionOp              |
| GetNumProgramsConverter                    | triton::GetNumProgramsOp to a param in functionOp            |
| LoadConverter                              | triton::LoadOp to memref::copy and bufferization::ToTensorOp |
| AtomicRMWConverter                         | triton::AtomicRMWOp to linalg::GenericOp                     |
| AtomicCASConverter                         | triton::AtomicCASOp to linalg::GenericOp                     |
| MakeRangeConverter                         | triton::MakeRangeOp to linalg::GenericOp                     |
| SplatConverter                             | triton::SplatOp to linalg::FillOp                            |
| ClampFConverter                            | triton::ClampFOp to tensor::EmptyOp, linalg::FillOp          |
| PreciseDivConverter                        | triton::PreciseDivFOp to arith::DivFOp                       |
| ArgMinConverter                            | triton::ArgMinOp to linalg::ReduceOp                         |
| ArgMaxConverter                            | triton::ArgMaxOp to linalg::ReduceOp                         |
| ReduceConverter                            | triton::ReduceOp to linalg::ReduceOp                         |
| ScanConverter                              | triton::ScanOp to func::CallOp                               |
| ReshapeConverter                           | triton::ReshapeOp to tensor::ReshapeOp                       |
| ExpandDimsConverter                        | triton::ExpandDimsOp to tensor::ExpandShapeOp                 |
| BroadcastConverter                         | triton::BroadcastOp to linalg::BroadcastOp                   |
| DenseConstantConverter                     | arith::ConstantOp to linalg::FillOp                          |
| ExternElementwiseClOpConverter             | triton::ExternElementwiseOp to linalg::MapOp                 |
| TritonMulhiuiConverter                     | triton::MulhiUIOp to arith::MulSIExtendedOp                  |
| TritonPreciseSqrtConverter                 | triton::PreciseSqrtOp to math::SqrtOp                        |
| AdvanceConverter                           | triton::AdvanceOp to memref::ReinterpretCastOp               |
| TransposeConverter                         | triton::TransOp to linalg::TransposeOp                       |
| SplitConverter                             | triton::SplitOp to tensor::ExtractSliceOp                    |
| JoinConverter                              | triton::JoinOp to tensor::InsertSliceOp                      |
| CatConverter                               | triton::CatOp to tensor::InsertSliceOp                       |
| BitcastConverter                           | triton::BitcastOp to arith::BitcastOp                        |
| LoopConverter\<scf::ForOp\>                | scf::ForOp to scf::ForOp                                     |
| LoopConverter\<scf::WhileOp\>              | scf::WhileOp to scf::WhileOp                                 |
| YieldConverter                             | scf::YieldOp to scf::YieldOp                                 |
| GatherConverter                            | triton::GatherOp to func::FuncOp                             |
| GatherLoadConverter                        | triton::GatherLoadOp to scf::ForOp                           |
| DeviceAssertConverter                      | triton::AssertOp to func::FuncOp                             |
| DevicePrintConverter                       | triton::PrintOp to func::FuncOp                              |
| MatmulConverter                            | triton::DotOp to linalg::MatmulOp                            |
| SortOpConverter                            | triton::SortOp to func::FuncOp                               |
| DotScaledConverter                         | triton::DotScaledOp to linalg::MatmulOp                      |
| PtrToIntConverter                          | triton::PtrToIntOp                                           |
| MakeTensorPtrConverter                     | triton::PtrToIntOp to arith::IndexCastOp                     |

##### 3.2.2.4 other passes 

| Pass名称 | 功能描述 | 核心转换器 | 转换器描述 |
|---|---|---|---|
| triton-to-annotation | 处理Ascend NPU特有的编译提示指令 (`tl.compile_hint`)，将其转换为后端的Annotation方言，用于指导后续的硬件特定优化或资源配置。 | TritonAnnotationConversion | 将 `triton::AnnotationOp` 转换为 `annotation::MarkOp`，实现高级编译提示信息向底层注释标记的传递。 |
| triton-to-hfusion | 将Triton中的`TTIR`转换为Ascend NPU硬件加速器`HFusion`方言中的对应操作 | TritonHistogramToHFusionConversion | 将 `triton::HistogramOp` 转换为 `hfusion::HistogramOp`，使能在NPU的专用硬件上高效执行。 |
| triton-to-hivm | 处理Triton的块同步操作 (`tl.sync_block_all`, `tl.sync_block_set`, `tl.sync_block_wait`)，将其转换为Ascend NPU的`HIVM`方言中的跨核心同步指令。这些指令用于管理多核流水线中的同步与数据依赖，是流水优化的关键。 | TritonCustomOpToHIVMSyncOpConversion | 实现Triton同步指令到HIVM同步指令的转换：<br>• `sync_block_all`：全局块同步<br>• `sync_block_set`：设置同步点<br>• `sync_block_wait`：等待同步点 |
| triton-to-llvm | 将Triton中的内联汇编操作 (`tl.inline_assembly`) 转换为LLVM方言的内联汇编，并最终映射为Ascend NPU的CCE硬件固有函数（Intrinsics） | ElementwiseInlineAsmOpConversion | 将 `triton::ElementwiseInlineAsmOp` 转换为 `LLVM::InlineAsmOp` |

#### 3.2.3 Ascend affinitive Operators

| 序号 | Operator | 功能描述 |
|---|---|---|
| 1 | tl.custom_op | Ascend NPU扩展的自定义算子集，用于支持硬件特定的内存访问与数据搬运模式，例如：<br>• `index_select`: 基于索引选择数据<br>• `index_put`: 基于索引放置数据<br>• `gather_out_to_ub`: 将外部数据收集到Unified Buffer (UB)<br>• `scatter_ub_to_out`: 将UB中的数据分散输出<br>• `indirect_load`: 间接地址加载<br>• `indirect_store`: 间接地址存储 |
| 2 | tl.compile_hint | 向编译器传递硬件特定的编译提示信息，用于指导后端优化策略、资源分配或内核配置。 |
| 3 | tl.sync_block_wait(`sender, receiver, event_id`) | 块同步等待操作。指定接收核 (`receiver`) 等待由发送核 (`sender`) 发出的事件信号 (`event_id`)，用于管理跨核流水线中的数据依赖与执行顺序。 |
| 4 | tl.sync_block_set(`sender, receiver, event_id`) | 块同步设置操作。指定发送核 (`sender`) 向接收核 (`receiver`) 发出一个事件信号 (`event_id`)，表明某个执行阶段或数据已准备就绪。 |
| 5 | tl.sync_block_all(`mode, event_id`) | 全局块同步操作。根据指定的同步模式 (`mode`)，向所有相关的接收核广播一个事件信号 (`event_id`)，用于实现全核同步或特定模式的集体同步。 |
