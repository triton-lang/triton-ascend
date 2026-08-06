# 昇腾拓展API

|api|简要说明|
|--|--|
|[extract_slice](./extract_slice.md)|  从输入张量中按照操作指定的偏移量、大小和步幅参数提取一个张量。 |
|[insert_slice](./insert_slice.md)| 将一个张量（子张量）插入到另一个张量的指定位置，按照操作指定的偏移量、大小和步幅参数插入到另一个张量中。 |
|[sync_block](./sync_block.md) | 显式的核心间同步指令，用于协调 Cube-Vector 架构中不同核心间的执行顺序和数据一致性。 |
|[compile_hint](./compile_hint.md) | 一个编译器提示（hint）机制，允许用户为张量附加元数据信息，这些信息会被传递到编译器后端，用于指导优化和代码生成。|
|[multibuffer](./multibuffer.md) | 为张量设置多缓冲，允许编译器对同一张量创建多个副本。 |
|[parallel](./parallel.md) | `parallel` 是一个专门用于多核心并行执行的迭代器,提供显式的多核心并行语义。 |
|[get_element](./get_element.md)| 根据给定的索引，从输入张量中读取单个元素。 |
|[index_select 高性能接口](./index_select_simd.md) | 在非尾轴维度上并行 gather 多个索引，并以 tile 为单位将数据零拷贝地从全局内存（GM）直接搬运到统一缓冲区（UB）的正确位置。该操作等效于 `torch.index_select` 的高性能实现，适用于嵌入层查找、稀疏索引访问等场景。 |

```{toctree}
:maxdepth: 3
:hidden:
extract_slice.md
insert_slice.md
sync_block.md
compile_hint.md
multibuffer.md
parallel.md
get_element.md
index_select_simd.md
```
