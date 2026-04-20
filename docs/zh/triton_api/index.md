# triton.language

## 昇腾拓展API列表

|api|简要说明|
|--|--|
|[extract_slice](./Extension_Ops/extract_slice.md)|  从输入张量中按照操作指定的偏移量、大小和步幅参数提取一个张量。 |
|[insert_slice](./Extension_Ops/insert_slice.md)| 将一个张量（子张量）插入到另一个张量的指定位置，按照操作指定的偏移量、大小和步幅参数插入到另一个张量中。 |
|[sync_block](./Extension_Ops/sync_block.md) | 显式的核心间同步指令，用于协调 Cube-Vector 架构中不同核心间的执行顺序和数据一致性。 |
|[compile_hint](./Extension_Ops/compile_hint.md) | 一个编译器提示（hint）机制，允许用户为张量附加元数据信息，这些信息会被传递到编译器后端，用于指导优化和代码生成。|
|[multibuffer](./Extension_Ops/multibuffer.md) | 为张量设置多缓冲，允许编译器对同一张量创建多个副本。 |
|[parallel](./Extension_Ops/parallel.md) | `parallel` 是一个专门用于多核心并行执行的迭代器,提供显式的多核心并行语义。 |
|[get_element](./Extension_Ops/get_element.md)| 根据给定的索引，从输入张量中读取单个元素。 |
|[index_select 高性能接口](./Extension_Ops/index_select_simd.md) | 在非尾轴维度上并行 gather 多个索引，并以 tile 为单位将数据零拷贝地从全局内存（GM）直接搬运到统一缓冲区（UB）的正确位置。该操作等效于 `torch.index_select` 的高性能实现，适用于嵌入层查找、稀疏索引访问等场景。 |

```{toctree}
:maxdepth: 3
:hidden:

Extension_Ops/extract_slice.md
Extension_Ops/insert_slice.md
Extension_Ops/sync_block.md
Extension_Ops/compile_hint.md
Extension_Ops/multibuffer.md
Extension_Ops/parallel.md
Extension_Ops/get_element.md
Extension_Ops/index_select_simd.md
```

## 原子操作

|api|简要说明|
|--|--|
|[atomic_add](./Atomic_Ops/atomic_add.md)  |在由 pointer 指定的内存位置执行原子加法 |
|[atomic_and](./Atomic_Ops/atomic_and.md)  |在由 pointer 指定的内存位置执行原子逻辑和操作 |
|[atomic_cas](./Atomic_Ops/atomic_cas.md)  |在由 pointer 指定的内存位置执行 1 个原子比较并交换操作 |
|[atomic_max](./Atomic_Ops/atomic_max.md)  |在由 pointer 指定的内存位置执行 1 个原子最大值操作 |
|[atomic_min](./Atomic_Ops/atomic_min.md)  |在由 pointer 指定的内存位置执行 1 个原子最小值操作 |
|[atomic_or](./Atomic_Ops/atomic_or.md)  |在由 pointer 指定的内存位置执行 1 个原子逻辑或操作 |
|[atomic_xchg](./Atomic_Ops/atomic_xchg.md)  |在由 pointer 指定的内存位置执行 1 个原子交换操作 |
|[atomic_xor](./Atomic_Ops/atomic_xor.md)  |在由 pointer 指定的内存位置执行原子逻辑异或操作 |

```{toctree}
:maxdepth: 3
:hidden:

Atomic_Ops/atomic_add.md
Atomic_Ops/atomic_and.md
Atomic_Ops/atomic_cas.md
Atomic_Ops/atomic_max.md
Atomic_Ops/atomic_min.md
Atomic_Ops/atomic_or.md
Atomic_Ops/atomic_xchg.md
Atomic_Ops/atomic_xor.md
```

## 比较操作

|api|简要说明|
|--|--|
| [eq](./Comparing_Ops/eq.md) | 用于比较两个张量的元素，与`==`等价 |
| [le](./Comparing_Ops/le.md) | 用于比较两个张量的元素，与`<=`等价。 |
| [ge](./Comparing_Ops/ge.md) | 用于比较两个张量的元素，与`>=`等价。 |
| [lt](./Comparing_Ops/lt.md) | 用于比较两个张量的元素，与`<`等价。 |
| [gt](./Comparing_Ops/gt.md) | 用于比较两个张量的元素，与`>`等价。 |
| [ne](./Comparing_Ops/ne.md) | 用于比较两个张量的元素，与`!=`等价。 |

```{toctree}
:maxdepth: 3
:hidden:

Comparing_Ops/eq.md
Comparing_Ops/le.md
Comparing_Ops/ge.md
Comparing_Ops/lt.md
Comparing_Ops/gt.md
Comparing_Ops/ne.md
```

## 编译器提示操作

|api|简要说明|
|--|--|
|[debug_barrier](./Compiler_Hint_Ops/debug_barrier.md) |插入 1 个屏障以同步 1 个块中的所有线程 |
|[max_constancy](./Compiler_Hint_Ops/max_constancy.md) |告知编译器 input 中的第 1 个值是常量 |
|[max_contiguous](./Compiler_Hint_Ops/max_contiguous.md) |告知编译器 input 中的第 1 个值是连续 |
|[multiple_of](./Compiler_Hint_Ops/multiple_of.md) |告知编译器 input 中的所有值都是 value 的倍数 |
|[assume](./Compiler_Hint_Ops/assume.md)         | 用于向编译器提供条件假设信息，允许编译器基于已知为真的条件进行优化。 |
|[compile_hint](./Extension_Ops/compile_hint.md) | 一个编译器提示（hint）机制，允许用户为张量附加元数据信息，这些信息会被传递到编译器后端，用于指导优化和代码生成。|
|[multibuffer](./Extension_Ops/multibuffer.md) | 为张量设置多缓冲，允许编译器对同一张量创建多个副本。 |
|[parallel](./Extension_Ops/parallel.md) | `parallel` 是一个专门用于多核心并行执行的迭代器,提供显式的多核心并行语义。 |
|[sync_block指令](./Extension_Ops/sync_block.md) | 显式的核心间同步指令，用于协调 Cube-Vector 架构中不同核心间的执行顺序和数据一致性。 |

```{toctree}
:maxdepth: 3
:hidden:

Compiler_Hint_Ops/debug_barrier.md
Compiler_Hint_Ops/max_constancy.md
Compiler_Hint_Ops/max_contiguous.md
Compiler_Hint_Ops/multiple_of.md
Compiler_Hint_Ops/assume.md
Extension_Ops/compile_hint.md
Extension_Ops/multibuffer.md
Extension_Ops/parallel.md
Extension_Ops/sync_block.md
```

## 创建操作

|api|简要说明|
|--|--|
|[arange](./Creation_Ops/arange.md) | 返回半开区间 [start, end) 内的连续值 |
|[cat](./Creation_Ops/cat.md) | 连接给定的块 |
|[full](./Creation_Ops/full.md) | 返回一个张量，该张量填充了指定 shape 和 dtype 的标量值|
|[zeros](./Creation_Ops/zeros.md)| 返回一个张量，该张量用指定 shape 和 dtype 填充了标量值 0 |
|[zeros_like](./Creation_Ops/zeros_like.md)| 返回一个 shape 和 dtype 与给定张量相同的全零张量 |
|[cast](./Creation_Ops/cast.md)| 将张量转换为指定的 dtype|

```{toctree}
:maxdepth: 3
:hidden:

Creation_Ops/arange.md
Creation_Ops/cat.md
Creation_Ops/full.md
Creation_Ops/zeros.md
Creation_Ops/zeros_like.md
Creation_Ops/cast.md
```

## 调试操作

|api|简要说明|
|--|--|
|[static_print](./Debug_Ops/static_print.md) |在编译时打印数值 |
|[static_assert](./Debug_Ops/static_assert.md) |在编译时断言条件 |
|[device_print](./Debug_Ops/device_print.md) |在运行时从设备打印数值 |
|[device_assert](./Debug_Ops/device_assert.md) |在运行时从设备上断言条件 |

```{toctree}
:maxdepth: 3
:hidden:

Debug_Ops/static_print.md
Debug_Ops/static_assert.md
Debug_Ops/device_print.md
Debug_Ops/device_assert.md

```

## 索引与元素操作

|api|简要说明|
|--|--|
|[flip](./Indexing_Ops/flip.md) |沿着维度 dim 翻转张量 x |
|[where](./Indexing_Ops/where.md) |根据 condition 返回来自 x 或 y 的元素组成的张量 |
|[swizzle2d](./Indexing_Ops/swizzle2d.md) |将行主序排列为 size_i * size_j 的矩阵的索引，转换为每组 size_g 行的列主序矩阵的索引 |
|[get_element](./Extension_Ops/get_element.md)| 根据给定的索引，从输入张量中读取单个元素。 |
|[index_select 高性能接口](./Extension_Ops/index_select_simd.md) | 在非尾轴维度上并行 gather 多个索引，并以 tile 为单位将数据零拷贝地从全局内存（GM）直接搬运到统一缓冲区（UB）的正确位置。该操作等效于 `torch.index_select` 的高性能实现，适用于嵌入层查找、稀疏索引访问等场景。 |
|[gather](./Indexing_Ops/gather.md) | 对`src`tensor沿`axis`维度按照`index`执行gather操作 |

```{toctree}
:maxdepth: 3
:hidden:

Indexing_Ops/flip.md
Indexing_Ops/where.md
Indexing_Ops/swizzle2d.md
Extension_Ops/get_element.md
Extension_Ops/index_select_simd.md
Indexing_Ops/tl.gather.md
```

## 内联汇编

|api|简要说明|
|--|--|
|[inline_asm_elementwise](./Inline_Assembly/inline_asm_elementwise.md) |在张量上执行内联汇编 |

```{toctree}
:maxdepth: 3
:hidden:

Inline_Assembly/inline_asm_elementwise.md
```

## 迭代器

|api|简要说明|
|--|--|
|[range](./Iterators/range.md)  |永远向上计数的迭代器 |
|[static_range](./Iterators/static_range.md) | 永远向上计数的迭代器 |

```{toctree}
:maxdepth: 3
:hidden:

Iterators/range.md
Iterators/static_range.md
```

## 线性代数操作

|api|简要说明|
|--|--|
|[dot](./Linear_Algebra_Ops/dot.md)| 返回两个块的矩阵乘积|
|[dot_scaled](./Linear_Algebra_Ops/dot_scaled.md) | 计算以缩放格式表示两个矩阵块的矩阵乘积 |

```{toctree}
:maxdepth: 3
:hidden:

Linear_Algebra_Ops/dot.md
Linear_Algebra_Ops/dot_scaled.md
```

## 逻辑操作

|api|简要说明|
|--|--|
|[and](./Logical_Ops/and.md) | 逻辑与操作 |
|[or](./Logical_Ops/or.md) | 逻辑或操作 |
|[not](./Logical_Ops/not.md) | 逻辑非操作 |
|[logical_and](./Logical_Ops/logical_and.md)| 用于对两个张量进行逐元素逻辑与运算 |
|[logical_or](./Logical_Ops/logical_or.md)| 用于对两个张量进行逐元素逻辑或运算 |
|[not](./Logical_Ops/not.md) | 将tensor的值按位取反。 |
|[invert](./Logical_Ops/invert.md) | 将tensor每个值按比特位进行翻转。 |
|[lshift](./Logical_Ops/lshift.md) | 根据给定的位移数将tensor张量进行左移位。 |
|[rshift](./Logical_Ops/rshift.md) | 根据给定的位移数将tensor张量进行右移位。 |
|[xor](./Logical_Ops/xor.md) | 计算两个元素的异或值。 |

```{toctree}
:maxdepth: 3
:hidden:

Logical_Ops/and.md
Logical_Ops/or.md
Logical_Ops/not.md
Logical_Ops/logical_and.md
Logical_Ops/logical_or.md
Logical_Ops/not.md
Logical_Ops/invert.md
Logical_Ops/lshift.md
Logical_Ops/rshift.md
Logical_Ops/xor.md
```

## 数学操作

|api|简要说明|
|--|--|
|[add](./Math_Ops/add.md) | 四则运算加法 ‘+’ |
|[sub](./Math_Ops/sub.md) | 四则运算减法 ‘-’ |
|[mul](./Math_Ops/mul.md) | 四则运算乘法 ‘*’ |
|[div](./Math_Ops/div.md) | 四则运算除法 ‘/’ |
|[abs](./Math_Ops/abs.md) |计算 x 的逐元素绝对值 |
|[neg](./Math_Ops/neg.md) | 将tensor的值取负。 |
|[cdiv](./Math_Ops/cdiv.md) |计算 x 除以 div 的向上取整除法 |
|[ceil](./Math_Ops/ceil.md) |计算 x 的逐元素向上取整值 |
|[clamp](./Math_Ops/clamp.md) |将输入张量 x 的值限制在 [min, max] 范围内 |
|[cos](./Math_Ops/cos.md) |计算 x 的逐元素余弦值 |
|[div_rn](./Math_Ops/div_rn.md) |计算 x 和 y 的逐元素精确除法（根据 IEEE 标准四舍五入到最近的值） |
|[erf](./Math_Ops/erf.md) |计算 x 的逐元素误差函数 |
|[exp](./Math_Ops/exp.md) |计算 x 的逐元素指数 |
|[exp2](./Math_Ops/exp2.md) |计算 x 的逐元素指数（以 2 为底）|
|[fdiv](./Math_Ops/fdiv.md) |计算 x 和 y 的逐元素快速除法 |
|[floor](./Math_Ops/floor.md) |计算 x 的逐元素向下取整 |
|[fma](./Math_Ops/fma.md) |计算 x、y 和 z 的逐元素融合乘加运算 |
|[log](./Math_Ops/log.md) |计算 x 的逐元素自然对数 |
|[log2](./Math_Ops/log2.md) |计算 x 的逐元素对数（以 2 为底）|
|[mod](./Math_Ops/mod.md) | 取模运算 |
|[maximum](./Math_Ops/maximum.md) |计算 x 和 y 的逐元素最大值 |
|[minimum](./Math_Ops/minimum.md) |计算 x 和 y 的逐元素最小值 |
|[rsqrt](./Math_Ops/rsqrt.md) |计算 x 的逐元素的平方根倒数 |
|[sigmoid](./Math_Ops/sigmoid.md) |计算 x 的逐元素 sigmoid 函数值 |
|[sin](./Math_Ops/sin.md) |Computes the element-wise sine of x. 计算 x 的逐元素正弦值 |
|[softmax](./Math_Ops/softmax.md) |计算 x 的逐元素 softmax 值 |
|[sqrt](./Math_Ops/sqrt.md) |计算 x 的逐元素快速平方根 |
|[sqrt_rn](./Math_Ops/sqrt_rn.md) |计算 x 的逐元素精确平方根（根据 IEEE 标准四舍五入到最近的值） |
|[umulhi](./Math_Ops/umulhi.md)  |计算 x 和 y 的 2N 位乘积的逐元素最高有效 N 位 |

```{toctree}
:maxdepth: 3
:hidden:

Math_Ops/add.md
Math_Ops/sub.md
Math_Ops/mul.md
Math_Ops/div.md
Math_Ops/abs.md
Math_Ops/neg.md
Math_Ops/cdiv.md
Math_Ops/ceil.md
Math_Ops/clamp.md
Math_Ops/cos.md
Math_Ops/div_rn.md
Math_Ops/erf.md
Math_Ops/exp.md
Math_Ops/exp2.md
Math_Ops/fdiv.md
Math_Ops/floor.md
Math_Ops/fma.md
Math_Ops/log.md
Math_Ops/log2.md
Math_Ops/mod.md
Math_Ops/maximum.md
Math_Ops/minimum.md
Math_Ops/rsqrt.md
Math_Ops/sigmoid.md
Math_Ops/sin.md
Math_Ops/softmax.md
Math_Ops/sqrt.md
Math_Ops/sqrt_rn.md
Math_Ops/umulhi.md
```

## 内存/指针操作

|api|简要说明|
|--|--|
|[load](./Memory_Pointer_Ops/tl.load.md) |返回一个张量，其值从由指针定义的内存位置加载|
|[store](./Memory_Pointer_Ops/tl.store.md) |将数据张量存储到由指针定义的内存位置|
|[make_block_ptr](./Memory_Pointer_Ops/tl.make_block_ptr.md) |返回指向父张量中某个块的指针|
|[advance](./Memory_Pointer_Ops/tl.advance.md) |推进一个块指针|
|[load_tensor_descriptor](./Memory_Pointer_Ops/load_tensor_descriptor.md) | 从张量描述符加载数据块 |
|[make_tensor_descriptor](./Memory_Pointer_Ops/make_tensor_descriptor.md) | 创建张量描述符对象 |

```{toctree}
:maxdepth: 3
:hidden:

Memory_Pointer_Ops/tl.load.md
Memory_Pointer_Ops/tl.store.md
Memory_Pointer_Ops/tl.make_block_ptr.md
Memory_Pointer_Ops/tl.advance.md
Memory_Pointer_Ops/load_tensor_descriptor.md
Memory_Pointer_Ops/make_tensor_descriptor.md
```

## 编程模型

|api|简要说明|
|--|--|
| tensor | 表示一个值或指针的 N 维数组 |
| [program_id](./Programming_Model/program_id.md) | 沿指定轴返回当前程序实例的 id |
| [num_programs](./Programming_Model/num_programs.md) | 沿指定轴返回当前程序实例的数量 |

```{toctree}
:maxdepth: 3
:hidden:

Programming_Model/program_id.md
Programming_Model/num_programs.md
```

## 随机数生成

|api|简要说明|
|--|--|
|[randint4x](./Random_Number_Generation/randint4x.md) |给定 1 个seed 标量和 1 个offset 块，返回 4 个 int32 类型的随机块 |
|[randint](./Random_Number_Generation/randint.md) |给定 1 个 seed 标量和 1 个 offset 块，返回 1 个 int32 类型的随机块 |
|[rand](./Random_Number_Generation/rand.md)   |给定 1 个 seed 标量和 1 个 offset 块，返回 1 个在 U(0,1) 中的 float32 类型的随机块 |
|[randn](./Random_Number_Generation/randn.md)   |给定 1 个 seed 标量和 1 个 offset 块，返回 1 个在 N(0,1) 中的 float32 类型的随机块 |

```{toctree}
:maxdepth: 3
:hidden:

Random_Number_Generation/randint4x.md
Random_Number_Generation/randint.md
Random_Number_Generation/rand.md
Random_Number_Generation/randn.md
```

## 归约操作

|api|简要说明|
|--|--|
|[argmax](./Reduction_Ops/argmax.md) |返回沿指定 axis 轴上 input 张量中所有元素的最大索引 |
|[argmin](./Reduction_Ops/argmin.md) |返回沿指定 axis 轴上 input 张量中所有元素的最小索引 |
|[max](./Reduction_Ops/max.md) |返回沿指定 axis 轴上 input 张量中所有元素的最大值 |
|[min](./Reduction_Ops/min.md) |返回沿指定 axis 轴上 input 张量中所有元素的最小值 |
|[reduce](./Reduction_Ops/reduce.md) |将 combine_fn 应用于沿指定 axis 的 input 张量中的所有元素 |
|[sum](./Reduction_Ops/sum.md) |返回 input 张量中，沿指定 axis 的所有元素的总和 |
|[xor_sum](./Reduction_Ops/xor_sum.md) |返回 input 张量中，沿指定 axis 的所有元素的异或和 |

```{toctree}
:maxdepth: 3
:hidden:

Reduction_Ops/argmax.md
Reduction_Ops/argmin.md
Reduction_Ops/max.md
Reduction_Ops/min.md
Reduction_Ops/reduce.md
Reduction_Ops/sum.md
Reduction_Ops/xor_sum.md
```

## 扫描/排序操作

|api|简要说明|
|--|--|
|[associative_scan](./Scan_Sort_Ops/associative_scan.md) |沿指定 axis 将 combine_fn 应用于 input 张量的每个元素和携带的值，并更新携带的值 |
|[cumprod](./Scan_Sort_Ops/cumprod.md) |返回沿指定 axis 的 input 张量中所有元素的累积乘积 |
|[cumsum](./Scan_Sort_Ops/cumsum.md)  |返回沿指定 axis 的 input 张量中所有元素的累积和 |
|[histogram](./Scan_Sort_Ops/histogram.md) |基于 input 张量计算 1 个具有 num_bins 个 bin 的直方图，每个 bin 宽度为 1，起始值为0 |
|[sort](./Scan_Sort_Ops/sort.md) |沿着指定维度对张量进行排序 |

```{toctree}
:maxdepth: 3
:hidden:

Scan_Sort_Ops/associative_scan.md
Scan_Sort_Ops/cumprod.md
Scan_Sort_Ops/cumsum.md
Scan_Sort_Ops/histogram.md
Scan_Sort_Ops/sort.md
```

## Shape操作

|api|简要说明|
|--|--|
|[broadcast](./Shape_Manipulation_Ops/broadcast.md) | 尝试将两个给定的块广播到一个共同兼容的 shape |
|[broadcast_to](./Shape_Manipulation_Ops/broadcast_to.md) | 尝试将给定的张量广播到新的 shape |
|[expand_dims](./Shape_Manipulation_Ops/expand_dims.md) | 通过插入新的长度为 1 的维度来扩展张量的形状
|[interleave](./Shape_Manipulation_Ops/interleave.md) | 沿着最后一个维度交错两个张量的值 |
|[join](./Shape_Manipulation_Ops/join.md) | 在一个新的次要维度中连接给定的张量 |
|[permute](./Shape_Manipulation_Ops/permute.md) | 排列张量的维度 |
|[ravel](./Shape_Manipulation_Ops/ravel.md) | 返回 x 的连续扁平视图 |
|[reshape](./Shape_Manipulation_Ops/reshape.md) | 返回一个具有与输入相同元素数但具有提供的形状的张量|
|[split](./Shape_Manipulation_Ops/split.md) | 将张量沿其最后一个维度分成两部分，该维度大小必须为 2 |
|[trans](./Shape_Manipulation_Ops/trans.md) | 排列张量的维度。 |
|[view](./Shape_Manipulation_Ops/view.md) | 返回具有与输入相同元素但形状不同的张量 |
|[extract_slice](./Extension_Ops/extract_slice.md)|  从输入张量中按照操作指定的偏移量、大小和步幅参数提取一个张量。 |
|[insert_slice](./Extension_Ops/insert_slice.md)| 将一个张量（子张量）插入到另一个张量的指定位置，按照操作指定的偏移量、大小和步幅参数插入到另一个张量中。 |

```{toctree}
:maxdepth: 3
:hidden:

Shape_Manipulation_Ops/broadcast.md
Shape_Manipulation_Ops/broadcast_to.md
Shape_Manipulation_Ops/expand_dims.md
Shape_Manipulation_Ops/interleave.md
Shape_Manipulation_Ops/join.md
Shape_Manipulation_Ops/permute.md
Shape_Manipulation_Ops/ravel.md
Shape_Manipulation_Ops/reshape.md
Shape_Manipulation_Ops/split.md
Shape_Manipulation_Ops/trans.md
Shape_Manipulation_Ops/view.md
Extension_Ops/extract_slice.md
Extension_Ops/insert_slice.md
```
