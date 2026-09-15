# 昇腾拓展API

Triton-Ascend 在标准 Triton 语言之上提供了面向昇腾 NPU 的扩展接口，包括扩展算子（`triton.language.extra.cann.extension`，简写 `al`）与 Buffer 语言（`bl`）。

## 扩展算子 (al)

|api|简要说明|
|--|--|
|[al.copy](./al/al.copy.md)| 在 `copy_from_ub_to_l1` 的基础上增加了 UB 到 UB 的复制。 |
|[al.ascend_address_space](./al/ascend_address_space.md) | 用于在 buffer 分配（`bl.alloc`）时指定昇腾硬件地址空间的枚举对象。 |
|[al.cast](./al/cast.md) | 将张量转换为指定的数据类型，支持数值类型转换、位级别重解释（bitcast）、浮点降精度舍入模式以及昇腾扩展的整数溢出处理模式。 |
|[al.compile_hint](./al/compile_hint.md) | 编译器提示（hint）机制，允许用户为张量附加元数据信息，指导编译器优化和代码生成。 |
|[al.copy_from_ub_to_l1](./al/copy_from_ub_to_l1.md) | 已弃用：自引入通用 `al.copy` 接口后被标记为 deprecated，调用时触发弃用警告。 |
|[al.custom](./al/custom_op.md) | CustomOp 和 CustomMacro：把已有的设备侧实现接入 Triton 编译流程。 |
|[al.debug_barrier](./al/debug_barrier.md) | 支持 VF 手动同步。 |
|[al.extract_slice](./al/extract_slice.md) | 从输入张量中按照操作指定的偏移量、大小和步幅参数提取一个张量。 |
|[al.fixpipe](./al/fixpipe.md) | 使用昇腾 A5 及后续架构的 L0C → UB 专用数据通路（fixpipe），将矩阵乘输出从 L0C 高效搬运到 UB。 |
|[al.flip](./al/flip.md) | 将 tensor 沿某一维度进行翻转。 |
|[al.get_element](./al/get_element.md) | 根据给定的索引，从输入张量中读取单个元素。 |
|[al.index_select_simd](./al/index_select_simd.md) | 在非最后一个维度上并行 gather 多个索引，并以 tile 为单位将数据零拷贝地从 GM 直接搬运到 UB。 |
|[al.insert_slice](./al/insert_slice.md) | 将一个张量（子张量）按照操作指定的偏移量、大小和步幅参数插入到另一个张量的指定位置。 |
|[al.multibuffer](./al/multibuffer.md) | 为张量设置多缓冲，允许编译器对同一张量创建多个副本。 |
|[al.parallel](./al/parallel.md) | 专门用于多核心并行执行的迭代器，继承自 `range`，提供显式的多核心并行语义。 |
|[al.scope](./al/scope.md) | 允许内核开发者显式指定计算核心类型（Cube Unit / Vector Unit）。 |
|[al.sort](./al/sort.md) | 对输入张量按维度进行升序或者降序排序。 |
|[al.sub_vec_id](./al/sub_vec_id.md) | 描述 AIC 与 AIV 核数配比下 Vector 核的编号信息。 |
|[al.sub_vec_num](./al/sub_vec_num.md) | 描述每个 AIC 对应的 AIV 数量。 |
|[al.sync_block_all](./al/sync_block_all.md) | 多核（Cube 核之间、Vector 核之间、或 Cube↔Vector）共享 GM 时的全量同步。 |
|[al.sync_block_set](./al/sync_block_set.md) | AIC+AIV 分离模式下的跨核同步：发送方完成写入后设置同步标志。 |
|[al.sync_block_wait](./al/sync_block_wait.md) | AIC+AIV 分离模式下的跨核同步：接收方等待发送方设置的同步标志。 |

```{toctree}
:maxdepth: 3
:hidden:

al/al.copy.md
al/ascend_address_space.md
al/cast.md
al/compile_hint.md
al/copy_from_ub_to_l1.md
al/custom_op.md
al/debug_barrier.md
al/extract_slice.md
al/fixpipe.md
al/flip.md
al/get_element.md
al/index_select_simd.md
al/insert_slice.md
al/multibuffer.md
al/parallel.md
al/scope.md
al/sort.md
al/sub_vec_id.md
al/sub_vec_num.md
al/sync_block_all.md
al/sync_block_set.md
al/sync_block_wait.md
```

## Buffer Language (bl)

|api|简要说明|
|--|--|
|[bl.alloc](./bl/alloc.md) | 用户手动创建指定地址空间上的内存（buffer），底层对接 `memref.alloc`。 |
|[bl.bind_buffer](./bl/bind_buffer.md) | 将 tensor 绑定到 buffer 上。 |
|[bl.subview](./bl/subview.md) | 在已有 buffer 上通过偏移、大小和步幅定义新视图，不复制底层数据。 |
|[bl.to_buffer](./bl/to_buffer.md) | 将 `tl.tensor` 张量对象转换为昇腾硬件专用的 `bl.buffer` 缓冲区对象。 |
|[bl.to_tensor](./bl/to_tensor.md) | 将 `bl.buffer` 对象转换为 `tl.tensor`，使其可参与 Triton 的张量计算。 |
|[triton_launch_kernel](./bl/triton_launch_kernel.md) | Ascend 后端 launcher stub 动态库（`.so`）中导出的 C 语言运行时接口。 |

```{toctree}
:maxdepth: 3
:hidden:

bl/alloc.md
bl/bind_buffer.md
bl/subview.md
bl/to_buffer.md
bl/to_tensor.md
bl/triton_launch_kernel.md
```
