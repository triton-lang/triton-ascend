# Ascend 社区测试迁移用例说明

本文逐项说明本目录迁入的社区测试函数用途，便于评审时核对测试目标。
源码基线为 `main-dev@396df6cb5b001314e36f22220be07a560de44664`；
共 231 个顶层测试函数、2184 个参数节点。逐函数来源及历史节点状态见
`MIGRATION_MANIFEST.tsv`，本轮直接验证结果以 PR 描述为准。
函数数量不表示每个参数节点都通过：冻结清单包含 1908 个 Pass、275 个
Skip 和 1 个 XFail；包含混合状态参数的函数须结合清单逐节点理解。

## `python/test/unit/language/test_annotations.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_int_annotation` | 验证不同符号位和位宽的整数类型注解会生成对应 TTIR 参数类型及整数到浮点转换；迁移版仅把临时输出缓冲区从 1 个元素扩大到 4 个，以覆盖内核对 `X + 3` 的写入，参数矩阵和 IR 断言均未改变。 |
| `test_unknown_annotation` | 验证 Triton 不认识的 Python 类型注解不会妨碍内核正常编译启动，错误形态的指针实参也不会导致未预期异常。 |

## `python/test/unit/language/test_block_pointer.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_block_ptr_matmul_no_scf` | 验证无 SCF 的 block pointer 矩阵乘在不同 K 块和 warp 数下可正确 advance、带边界补零加载，并得到与 torch.matmul 一致的结果。 |

## `python/test/unit/language/test_compile_errors.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_constexpr_annotated_global_var_access` | 验证仅用 tl.constexpr 类型注解的普通全局变量仍禁止在 JIT 内访问，并给出明确的全局变量编译错误。 |
| `test_constexpr_global_var_access` | 验证用 tl.constexpr(...) 创建的全局常量可在 JIT 内访问并成功编译。 |
| `test_defaults_assign_no_err` | 验证带普通默认参数和 constexpr 默认参数的 JIT 函数可按给定签名成功编译。 |
| `test_dot_scaled_shape_verification` | 验证 tl.dot_scaled 会拒绝尺寸错误的缩放张量，并报告准确的期望形状与实际形状。 |
| `test_err_constexpr_and_do_not_specialize` | 验证同一参数同时声明为 constexpr 和 do_not_specialize 时，无论直接编译还是启动都产生明确编译错误。 |
| `test_err_in_binary_op` | 验证对浮点数执行非法移位时，编译错误定位到正确源码行且不泄露 code_generator 内部栈帧。 |
| `test_err_in_binary_operator` | 验证对整数与字符串执行非法加法时，编译错误定位到表达式所在源码行且不暴露编译器内部栈帧。 |
| `test_err_in_builtin` | 验证内建函数参数错误同时保留 core.py 根因和调用点位置，并隐藏 code_generator 内部栈帧。 |
| `test_err_in_nested_call` | 验证嵌套 JIT 调用中的未定义变量错误分别准确标出被调函数根因和外层调用位置。 |
| `test_err_in_unary_op` | 验证对 tuple 执行不支持的 not 运算时，错误准确指向运算符且包含可用源码。 |
| `test_err_static_assert` | 验证失败的 tl.static_assert 产生 CompileTimeAssertionFailure，定位到断言调用且不带多余内部异常链。 |
| `test_err_undefined_variable` | 验证引用未定义局部变量时给出包含“is not defined”的源码级编译错误，而非内部实现栈。 |
| `test_global_access_in_fn_default_arg` | 验证普通全局值可在 JIT 函数定义时固化为默认参数并成功编译。 |
| `test_global_type_alias_access` | 验证全局 tl.pointer_type 类型别名可在 JIT 内访问并成功编译。 |
| `test_global_var_access` | 验证普通可变全局变量禁止在 JIT 函数体中访问，并报告全局变量错误。 |
| `test_not_const_annotate_no_err` | 验证仅带 Python int 注解的参数不会被误判为 constexpr，按 i32 签名可成功编译。 |
| `test_returns_branched_on_constexpr` | 验证 constexpr 分支可在编译期选定不同返回形状，并分别与调用方期望形状成功匹配。 |
| `test_returns_branched_on_non_constexpr` | 验证运行时分支返回不同形状时编译失败，并准确定位调用点及冲突的第二个 return。 |
| `test_two_returns_no_err` | 验证不可达的第二个 return 不会污染返回类型推断，调用方按首个返回形状成功编译。 |
| `test_unused_result` | 验证带 must_use_result 的 JIT 函数结果被使用时可编译、被丢弃时产生包含自定义附加文本的精确错误。 |
| `test_where_warning` | 验证以 uint32 张量作为 tl.where 条件时编译器会发出 UserWarning。 |

## `python/test/unit/language/test_compile_only.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_signature_ordering` | 验证 ASTSource 编译始终采用函数 arg_names 的参数顺序，而不受签名字典插入顺序影响。 |

## `python/test/unit/language/test_core.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_aliasing` | 验证同一设备缓冲区可同时传给两个形参，别名参数不会妨碍内核正确写入。 |
| `test_arange` | 验证 tl.arange 对多个非零起点生成连续整数序列，并与 torch.arange 完全一致。 |
| `test_atomic_min_max_neg_zero` | 验证浮点 atomic_min/atomic_max 处理 -0.0 后，输出在零容差数值比较下等于输入；当前断言不检查正零与负零的符号位是否相同。 |
| `test_atomic_rmw_predicate` | 验证带控制流谓词的 atomic_max 只纳入满足 val<64 的 program，最终最大值为 63。 |
| `test_call` | 验证 inline 辅助函数被多次调用时数值结果正确；对于非 interpreter 的 noinline 参数，社区现有断言仅要求启动产生任意异常，不校验异常类型、内容或具体失败原因。 |
| `test_clamp` | 验证逐元素 tl.clamp(x,min,max) 与 minimum(maximum(x,min),max) 的参考实现一致。 |
| `test_clamp_symmetric` | 验证后端可能优化的对称限幅 tl.clamp(x,-limit,limit) 仍与组合 min/max 参考结果一致。 |
| `test_constexpr_assignment` | 验证 constexpr 字面量经参数、带注解赋值及普通变量提升后保持正确的值和类型。 |
| `test_constexpr_flattens` | 验证嵌套构造 tl.constexpr 会自动扁平化，结果与单层 constexpr 相等。 |
| `test_constexpr_if_return` | 验证 constexpr 提前返回可与原子同步等非平凡控制流共存，单 program 与多 program 两条路径都正确完成。 |
| `test_constexpr_scalar_shape` | 验证 constexpr 标量参与整除后生成的张量表达式形状和值均正确。 |
| `test_constexpr_shape` | 验证由 constexpr 算术计算出的 tl.arange 终点可形成正确的 256 元素张量。 |
| `test_convert_float16_to_float32` | 穷举 half/bfloat16 位模式，验证辅助转换到 float32 后普通值、NaN 和 Inf 均保持正确语义。 |
| `test_cumsum_dtype` | 验证 int1/bool 输入的 tl.cumsum 会得到可存为 int32 的累积计数 1、2、3、4。 |
| `test_default` | 验证 JIT 辅助函数及内核的默认参数在省略和显式传值时分别取正确值。 |
| `test_dot_mulbroadcasted` | 验证用 expand_dims、逐元素乘法和 sum 手写的分块矩阵乘得到与 NumPy matmul 一致的结果。 |
| `test_dot_multidim` | 验证 2 至 6 维批量 tl.dot 在 A/B 可选转置时与 PyTorch float32 参考结果完全一致。 |
| `test_dot_without_load` | 验证由 tl.full 直接构造、未经内存 load 的矩阵可参与 tl.dot 并得到正确乘积。 |
| `test_dtype` | 验证指针元素类型可在 JIT 中作为 constexpr 读取，并支持相等、or 等静态类型判断。 |
| `test_dtype_codegen` | 验证各 Triton dtype 对象的 repr 可生成完整且可求值的 triton.language 类型名称。 |
| `test_dtype_tensor` | 验证有符号、无符号及标准浮点 tl.dtype 可作为 constexpr 参数用于构造对应类型的零张量。 |
| `test_expand_dims` | 验证 tl.expand_dims 对正/负轴、多个轴、标量及 constexpr 标量产生预期静态形状。 |
| `test_expand_dims_error_cases` | 验证 tl.expand_dims 对越界轴和规范化后重复轴均拒绝编译，并报告对应非法轴信息。 |
| `test_generic_reduction` | 验证自定义 combine_fn 的 tl.reduce 支持单值及多值 tuple，并正确计算 sum、均值和总体方差。 |
| `test_histogram` | 验证不同输入/桶规模下 tl.histogram 与 torch.histc 计数一致，且输出可参与广播。 |
| `test_histogram_silent_data_corruption` | 验证单桶 histogram 只写目标元素，不会越界破坏相邻缓冲区内容。 |
| `test_if` | 验证静态/动态 if、elif、逻辑与和条件表达式等多种前端分支形式均选择并写入正确值。 |
| `test_if_else` | 验证由设备内存条件控制的 if/else 在真假两条路径分别写入对应输入值。 |
| `test_if_return` | 验证动态条件和 constexpr 条件中的提前 return 均能正确区分提前退出与继续执行路径。 |
| `test_index1d` | 验证一维张量使用 None/冒号扩维索引时结果正确；对于目标 rank 或维度不匹配的内核，若发生异常则必须是 CompilationError，但当前代码也允许其不报错通过。 |
| `test_interleave` | 验证 tl.interleave 对两个向量按元素交错排列，在 debug 开关两种模式下均与参考结果一致。 |
| `test_interleave_scalars` | 验证 tl.interleave 可将两个标量组成长度为 2 的张量并按顺序存储。 |
| `test_invalid_pid_axis` | 验证 tl.program_id 对轴 20 产生明确错误，合法轴范围必须为 0、1、2。 |
| `test_invalid_slice` | 验证 Triton 张量不支持的 data[10:] 切片会产生“unsupported tensor index”错误。 |
| `test_jit_function_arg` | 验证 JITFunction 可作为参数传入另一 JITFunction，并正确完成逐元素平方。 |
| `test_join` | 验证 tl.join 将两个等长向量沿新末维组合，结果与 torch.stack 一致。 |
| `test_join_scalars` | 验证 tl.join 将两个标量组合为形状 [2] 的张量且元素顺序正确。 |
| `test_join_with_mma` | 验证 join 后 reshape 的张量可继续参与 tl.dot，并与 PyTorch 矩阵乘结果一致。 |
| `test_load_scalar_with_mask` | 验证标量索引和标量 mask 的 tl.load/tl.store 可正确读取并写回零值。 |
| `test_load_store_same_ptr` | 验证同一指针先 load 后原位 store 的内核重复运行时始终把输入 1 正确写成 2。 |
| `test_map_elementwise` | 验证 tl.map_elementwise 可把自定义三路比较函数逐元素应用到两个 int32 向量，并得到与 NumPy 比较参考值一致的 -1、0、1 结果。 |
| `test_masked_load_scalar` | 验证 constexpr 标量 mask 为真时加载输入、为假时采用 other 值，输出与参考张量一致。 |
| `test_masked_load_shared_memory` | 验证带 mask 的矩阵加载作为 tl.dot 输入时，计算结果与 torch.matmul 一致；源码未检查 Ascend IR 或实际存储层级，因此不能据此确认数据被放入共享内存。 |
| `test_math_divide_op` | 验证 tl.math.fdiv 和 tl.math.div_rn 的 float32 向量除法与 NumPy 除法在容差内一致。 |
| `test_max_returns_zero` | 验证全零向量的 tl.max 归约确实返回 0，而不会误保留输出初值。 |
| `test_nested_if_else_return` | 枚举三个动态条件，验证嵌套 if/else 中 return 路径不写输出，其余路径选择正确值。 |
| `test_nested_while` | 验证外层 for 与内层 while 的嵌套循环正确累计 40 次更新。 |
| `test_num_programs` | 验证三维 launch grid 中 tl.num_programs(0/1/2) 分别返回 11、21、31。 |
| `test_propagate_nan` | 验证 minimum、maximum、clamp 在 PropagateNan.NONE/ALL 下对一侧或两侧 NaN 的传播规则。 |
| `test_reshape` | 验证多个等元素数形状之间的 tl.reshape 保持元素排列并与 NumPy reshape 一致。 |
| `test_reshape_err` | 验证元素数量不匹配的 tl.reshape 在 warmup 编译阶段失败并包含 reshape 错误信息。 |
| `test_scalar_overflow` | 验证超大 constexpr 整数与 int32 张量运算时被范围检查拒绝，而不是静默溢出。 |
| `test_scan2d` | 在受支持的参数组合中，验证二维 cumsum/cumprod 及自定义 associative_scan 在不同轴、方向、形状和类型下与参考算法一致；部分 bfloat16 组合按上游条件 Skip。 |
| `test_scan_1d` | 验证一维 tl.cumsum 后 reshape 与 broadcast_to 的组合结果与 PyTorch 完全一致。 |
| `test_shapes_as_params` | 验证 expand_dims、reshape、permute/trans 等形状 API 同时接受独立维度参数与 tuple，并推导正确静态形状。 |
| `test_short_circuiting` | 验证 and 条件按 Python 短路语义处理 None、constexpr、不同指针类型，只有满足全部条件的 int32 数据被改写。 |
| `test_slice` | 验证 None 与空 slice 形式能为向量和标量正确插入维度，并产生预期静态 shape。 |
| `test_split` | 验证 tl.split 将末维为 2 的张量拆成两个向量，分别等于原张量两列。 |
| `test_split_to_scalar` | 验证长度为 2 的一维张量经 tl.split 后得到两个 shape=[] 的标量 tensor 且值正确。 |
| `test_static_range` | 验证 tl.static_range 的起止与 step 语义，编译期展开后的累加结果与 Python range 一致。 |
| `test_temp_var_in_loop` | 验证循环分支内临时变量的重新定义与复用不会生成错误 IR，最终累加值与 PyTorch 参考一致。 |
| `test_tensor_atomic_add_access_patterns` | 验证多种重复/乱序索引和 mask 下 tensor atomic_add 的冲突累加结果与逐元素参考实现一致。 |
| `test_tensor_atomic_add_non_exclusive_offset` | 验证多个输入元素映射到同一输出地址时 tensor atomic_add 能正确合并成相邻元素之和。 |
| `test_tensor_atomic_add_shift_1` | 验证二维偏移错位造成的重叠写入可由 tensor atomic_add 正确累加。 |
| `test_tensor_atomic_rmw_block` | 验证二维块 atomic_min 内核运行后，输出张量中至少产生一个值为 0 的元素；当前断言仅检查整个张量的最小值，不逐元素核对 8×8 结果。 |
| `test_tensor_member` | 验证 tl.tensor 的 abs()、sum() 成员方法与对应 tl.abs、tl.sum 函数结果相同。 |
| `test_tl_range_num_stages` | 验证 tl.range(num_stages=5) 驱动的分块矩阵乘数值正确，并在支持的 CUDA 路径检查流水级数代码生成。 |
| `test_tl_range_option_none` | 验证 tl.range 的 num_stages 与 loop_unroll_factor 显式传 None 时不会在 TTIR 中生成相应属性。 |
| `test_tma_load_block_shape_err` | 验证 tensor descriptor load 的块最后一维不足 16 字节时被拒绝并报告最小块大小错误。 |
| `test_tma_store_block_shape_err` | 验证 tensor descriptor store 的块最后一维不足 16 字节时被拒绝并报告最小块大小错误。 |
| `test_umulhi` | 验证 tl.umulhi 对 int32 输入返回 32×32 位无符号乘积的高 32 位，并逐元素等于 NumPy int64 参考计算。 |
| `test_unroll_attr` | 验证 tl.range(loop_unroll_factor) 对多种展开因子在 TTIR 中生成足够数量的循环体原子操作。 |
| `test_unsigned_name_mangling` | 验证编译器对 uint32 与 int32 生成不同函数特化/名称修饰，使 abs 语义分别正确。 |
| `test_unsplat` | 验证单元素 tensor 条件无论隐式还是通过 item() 显式标量化，都能正确控制分支写入。 |
| `test_value_specialization` | 验证跨 i32、i64、u64 边界的整数实参被推导为正确签名类型，并在编译内核名中体现常量特化。 |
| `test_value_specialization_overflow` | 验证 64 位有符号/无符号范围内的标量可启动，超出可表示范围时抛出 OverflowError。 |
| `test_where_broadcast` | 验证 tl.where 对一维条件广播到二维数据及 constexpr 标量条件时均与 NumPy where 一致。 |
| `test_while` | 验证动态 while 循环正确维护初值、循环携带变量和迭代计数，且不意外改写循环外初值。 |

## `python/test/unit/language/test_decorator.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_decorator_with_def` | 验证装饰器参数字符串中包含“def”不会干扰 JIT 对真实函数定义位置的解析。 |
| `test_triton_heuristic` | 验证叠加 autotune 与多层 heuristics 时可从 kwargs/位置参数计算 constexpr，并保留 base_fn 元数据。 |

## `python/test/unit/language/test_frontend.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_aggregate_constexpr_function` | 验证 aggregate 的 constexpr 字段和 constexpr 成员函数在前端中分别折叠为 4 与 16。 |
| `test_aggregate_initializers` | 验证 aggregate 内建初始化器及后续字段修改被正确展开为预期 make_range 和调用 IR。 |
| `test_aggregate_modification_in_for_loop` | 验证 aggregate 字段在 for 循环中作为 iter_arg 正确携带并在循环后保留更新值。 |
| `test_aggregate_modification_in_while_loop` | 验证 aggregate 字段在 while 循环中作为循环携带值正确更新并传出。 |
| `test_assign_attribute` | 验证 Triton aggregate 字段不允许在 JIT 内直接赋值，前端应拒绝编译。 |
| `test_assign_item` | 验证 Triton aggregate 不允许通过下标在 JIT 内修改元素，前端应拒绝编译。 |
| `test_assign_tuple_attrs_kernel` | 验证将 tuple 返回值解包后同时赋给 aggregate 多个属性的写法被前端拒绝。 |
| `test_augassign_attribute` | 验证 Triton aggregate 字段不允许在 JIT 内执行 += 等增量赋值。 |
| `test_augassign_item` | 验证 Triton aggregate 不允许通过下标执行 += 等增量修改。 |
| `test_call_in_loop` | 验证 for 循环内调用返回值 JIT 函数可正确生成 scf.for 和函数调用 IR。 |
| `test_call_in_while` | 验证 while 循环的不同分支中调用无返回值 JIT 函数可成功生成前端 IR。 |
| `test_constexpr_function_from_jit` | 验证 JIT 内调用 constexpr_function 会在编译期得到 8，并用于生成对应 tl.arange。 |
| `test_constexpr_function_from_python` | 验证同一 constexpr_function 在普通 Python 调用中也直接返回正确值 8。 |
| `test_constexpr_function_taking_list` | 验证 constexpr_function 可接收由 builtin 构造的列表并在编译期取出指定元素 8。 |
| `test_constexpr_getitem` | 验证 constexpr tuple 支持下标访问，其维度之和可在编译期用于构造正确 range。 |
| `test_constexpr_max_error` | 验证 constexpr max 遇到 NaN 或负零这类不稳定比较输入时产生 CompilationError。 |
| `test_constexpr_min_error` | 验证 constexpr min 遇到 NaN 或负零这类不稳定比较输入时产生 CompilationError。 |
| `test_constexpr_min_max` | 验证多参数 constexpr min/max 在编译期分别折叠为 1、-3、4、5。 |
| `test_for_loop_iv_modification` | 验证在 for 循环体内修改迭代变量只生成新的 SSA 值，不破坏原 scf.for 归纳变量。 |
| `test_jit_getitem` | 验证 aggregate 的 @triton.jit __getitem__ 被编译为独立调用，并正确返回底层 tensor。 |
| `test_jit_method` | 验证 aggregate 的 JIT 成员方法可返回多个字段，调用点正确接收并使用两个结果。 |
| `test_late_bound_class_reference` | 验证 constexpr 工厂动态生成的 aggregate 类能在随后定义的 JIT 内解析并正确传递其 tensor 字段。 |
| `test_reassign_aggregate_with_constexpr` | 验证含 constexpr 字段的 aggregate 不允许在 JIT 内用修改后对象重新赋值，前端应拒绝。 |
| `test_retrieve_item` | 验证 aggregate 的自定义 __getitem__ 可在编译期取出正确字段并生成对应 IR 值。 |
| `test_return_in_while` | 验证 while/for 循环体中出现 return 会被前端明确拒绝并给出规定错误信息。 |
| `test_specialized_recursion` | 验证递归 JIT 函数按张量形状逐层特化，16→8→4→2 的调用结构与预期 IR 一致。 |

## `python/test/unit/language/test_libdevice.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_libdevice_rename` | 验证 libdevice 函数重命名导入别名不会破坏模块加载，随后简单 Triton copy 内核仍可编译启动。 |

## `python/test/unit/language/test_line_info.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_map_elementwise_has_lineinfo` | 验证 tl.map_elementwise 及其回调生成的 TTIR 均带有效源码位置信息，不出现 loc(unknown)。 |

## `python/test/unit/language/test_pipeliner.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_pipeline_epilogue` | 验证含 1 至 5 级流水的逐行 load/add/store 尾部在 0 至 3 行边界下均输出全 1。 |
| `test_pipeline_matmul` | 在 Ascend 上验证 scale=False 的多 stage float16 矩阵乘与 torch.matmul 一致；scale=True 分支仅面向 CUDA/HIP CDNA，在 Ascend 上明确 Skip。异步拷贝和同步 IR 断言也仅在 CUDA 分支执行。 |
| `test_pipeline_vecadd` | 验证多 stage 循环中的分块向量加法数值正确；异步拷贝级数和分配数量仅在 CUDA 分支检查。 |

## `python/test/unit/language/test_random.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_rand` | 验证 tl.rand 对常量/运行时 seed 和 int32/int64 offset 生成 [0,1] 内且通过均匀分布 KS 检验的样本。 |
| `test_rand_limits` | 验证极端有符号整数经 uint_to_uniform_float 映射后严格小于 1 且位于最大可表示均匀随机值范围。 |
| `test_randn` | 验证 tl.randn 对常量/运行时 seed 和不同 offset 类型生成均值约 0、标准差约 1 的正态样本。 |
| `test_seed_is_int` | 验证 tl.rand 的 seed 必须是整数标量，tensor seed 和浮点 seed 都在编译期被拒绝。 |

## `python/test/unit/language/test_standard.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_flip_inf` | 验证 tl.flip 对含 Inf 的二维重排数据保持正确元素顺序和值，不因 Inf 发生错误。 |
| `test_ravel` | 验证二维 Triton 张量经 tl.ravel 展平后保持 0 至 255 的连续元素顺序。 |
| `test_swizzle2d` | 验证 tl.swizzle2d 按给定分组重映射二维坐标，得到与手工期望矩阵完全一致的布局。 |

## `python/test/unit/language/test_tensor_descriptor.py`

本节五个带 `num_ctas` 参数的二维 load/store 或 matmul 用例在 Ascend 上仅执行
`num_ctas=1`；`num_ctas=2` 按上游非 CUDA 条件 Skip。

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_host_tensor_descriptor_load` | 验证从 Python 侧构造并传入的 TensorDescriptor 元数据正确，指定二维块 load 结果等于原张量切片。 |
| `test_host_tensor_descriptor_matmul` | 验证 Python 侧 TensorDescriptor 作为 JIT 参数完成分块矩阵乘，结果与 torch.matmul 一致；六组 block shape 按比例缩小，以适配 Ascend UB 容量，同时保留横向、纵向、方形及不同流水阶段的覆盖。 |
| `test_make_tensor_descriptor_matmul` | 在 Ascend 的 num_ctas=1 配置下，验证设备侧 make_tensor_descriptor 的 load/store 分块矩阵乘在多种块大小和 stage 数下与 torch.matmul 一致；num_ctas=2 在非 CUDA 后端明确 Skip。 |
| `test_tensor_descriptor_functional_interface` | 验证函数式 load_tensor_descriptor/store_tensor_descriptor 可逐块完整复制二维张量。 |
| `test_tensor_descriptor_load` | 验证设备侧二维 descriptor 的 shape/stride/block_shape 元数据及偏移块加载结果均正确。 |
| `test_tensor_descriptor_load3d` | 在满足 descriptor 最小块字节数的参数组合中，验证带非满尾块和实际 stride 的三维 descriptor load 可覆盖原张量并保持所有有效元素。 |
| `test_tensor_descriptor_padding` | 验证设备侧和主机侧 descriptor 越界 load 均按 padding_option="nan" 填充 NaN。 |
| `test_tensor_descriptor_store` | 验证设备侧二维 descriptor 的元数据及网格分块 store 可完整重建输入张量。 |
| `test_tensor_descriptor_store3d` | 在满足 descriptor 最小块字节数的参数组合中，验证三维 descriptor store 可把非规则有效区域写入更大目标，并保持有效切片与输入一致。 |
| `test_tensor_descriptor_store_downcast` | 验证 descriptor.store 将 float32 值隐式下转换为 float16/bfloat16 时结果与显式 PyTorch 转换一致。 |
| `test_tma_gather_dot_pipeline` | 验证连续 descriptor gather 组成的 K 维流水矩阵乘与 PyTorch matmul 一致；`ttng.async_tma_gather` IR 仅在 CUDA CC≥10 时检查。 |

## `python/test/unit/language/test_tuple.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_add` | 验证 Python tuple 与 tl.tuple 的 + 运算都按拼接语义生成 0 至 7 的正确元素序列。 |
| `test_assign_return` | 验证 JIT 函数可返回 tuple 并由调用方解包，三个加减乘结果与内联写法一致。 |
| `test_eq` | 验证 Python tuple 与 tl.tuple 的相等比较对相同/不同元素分别返回真/假。 |
| `test_modifying_tuples` | 验证 tl.tuple 是不可变对象，对元素下标赋值会产生 CompilationError。 |
| `test_passing_nested_tuple_with_constexpr_and_jit_hook` | 验证含 constexpr 的嵌套 tuple 可序列化进特化数据并 preload，预热与预加载内核 hash 相同。 |
| `test_tuple_float` | 验证 JIT 中可对 float("-inf")、float("inf") 进行 tuple 解包并成功编译启动。 |
| `test_tuple_logic` | 验证 tuple 的 and/or 真值与短路语义，包括空 tuple 对动态表达式的编译期短路。 |

## `python/test/unit/runtime/test_autotuner.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_hooks` | 验证 autotuner 每次配置试跑的 pre_hook/post_hook 成对调用，并正确传递是否发生异常。 |
| `test_kwargs` | 验证 autotune 内核位置/关键字参数任意顺序均可正确建键，并对两个 M 值形成两个缓存项；Ascend 执行 use_cuda_graph=False，use_cuda_graph=True 因 CUDA 不可用而 XFail。 |
| `test_no_do_bench` | 验证未显式提供 do_bench 时 autotuner 使用默认基准流程并为当前 key 选出一个缓存配置。 |
| `test_prune_all_configs` | 验证 early_config_prune 删除全部候选配置时抛出内容准确的“无有效配置”TritonError。 |
| `test_prune_configs` | 验证 early_config_prune 或 perf_model 能接收正确实参与命名参数、裁剪配置，并保持内核复制结果正确。 |
| `test_restore` | 验证 autotune 试跑会恢复 restore_value 指定的原地修改输入，使最终只累加一次且支持位置/关键字调用。 |

## `python/test/unit/runtime/test_bindings.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_module_walk` | 验证对外 MLIR Python binding 可遍历 TTIR operation、result、operand、region、block、argument 及符号属性。 |
| `test_python_func_in_visit_call` | 验证 JIT 前端可在编译期调用 Python math.log2/math.e 形成常量并成功执行内核。 |

## `python/test/unit/runtime/test_build.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_compile_module` | 验证运行时可从 C 源码编译并导入 Python 扩展、正确调用函数，且相同源码复用缓存 so。 |
| `test_compile_module_bad_cache` | 验证缓存返回损坏 so 时编译模块流程能重建可用扩展，而非继续使用无效缓存文件。 |

## `python/test/unit/runtime/test_cache.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_async_compile` | 验证真实线程池异步 warmup 多个 dtype/constexpr 特化后，启动时等待并取得各自正确编译结果，且可继续同步编译新特化。 |
| `test_async_compile_mock` | 用可控假线程池验证异步编译任务延迟执行、相同特化去重，完成后可从设备缓存启动正确内核。 |
| `test_cache_builtin_as_global` | 验证作为全局值引用的 Triton builtin dtype 被纳入一致性检查，运行后更换类型会报全局变量变化。 |
| `test_cache_closure` | 验证 JIT 闭包捕获的 constexpr 被记录，编译后原地改变其值会触发明确缓存一致性错误。 |
| `test_changed_line_numbers_invalidate_cache` | 验证源码内容相同但文件行号不同的 JIT 函数具有不同 cache_key，以保留调试位置信息。 |
| `test_combine_fn_change` | 验证 tl.reduce/associative_scan 的种类及 combine_fn 运算内容都会进入 hash，四种组合键互不相同。 |
| `test_conflicting_global_in_inner_function` | 验证嵌套 JIT 首次编译后其 constexpr 全局值变化时，另一外层内核调用会报告全局值冲突。 |
| `test_constexpr_cache_invalidation_recreated` | 验证重复创建捕获相同/不同 constexpr 的局部 JIT 内核时不会错误复用缓存，各值均正确输出。 |
| `test_constexpr_fn_change` | 验证 constexpr_function 源码变化会改变使用者 cache_key，恢复源码后键也恢复基线。 |
| `test_higher_order_kernel` | 验证作为 constexpr 传入的高阶 JITFunction 源码原地修改会使调用内核重新编译，并可正确复用磁盘缓存。 |
| `test_hooks` | 验证 JIT cache/post-compile hook 收到一致的特化数据、warmup 标志、缓存 key 和完整函数名。 |
| `test_invalid_constexpr_fn` | 验证计算 constexpr_function 的 cache_key 时，依赖扫描会拒绝不受支持的 Python callable `torch.cuda.get_device_capability`，并抛出 RuntimeError。 |
| `test_jit_debug` | 验证 debug=False 与 debug=True 形成两个独立设备缓存项且生成不同 TTIR。 |
| `test_jit_warmup_cache` | 验证用 dtype 描述和真实 tensor 对同一签名反复 warmup 只产生一个设备缓存项。 |
| `test_kernel_default_arg` | 验证 JIT 默认参数在定义时捕获全局值，之后修改全局变量不会改变输出或产生新缓存项。 |
| `test_kernel_global_var_change` | 验证 JIT 运行后更换其引用的 constexpr 全局对象会触发全局变量变化错误。 |
| `test_local_does_not_shadow_global` | 验证先读取后赋值的同名局部并不会使 constexpr 全局引用脱离追踪，全局变化仍触发错误。 |
| `test_local_shadows_global` | 验证 JIT 内真正的同名局部变量遮蔽模块全局后，外部全局值变化不影响内核或缓存。 |
| `test_nested1_change` | 验证被调用链第一层依赖函数 function_2 的源码变化会改变顶层 kernel cache_key。 |
| `test_nested2_change` | 验证条件分支中另一嵌套依赖 function_0 的源码变化也会改变顶层 kernel cache_key。 |
| `test_no_cache_callable` | 验证全局 JITFunction 调用依赖由函数 hash 机制处理，不会错误写入 used_global_vals。 |
| `test_no_cache_module_as_global` | 验证通过 tl 模块调用 builtin 不会把整个模块对象错误记录进 used_global_vals。 |
| `test_nochange` | 验证依赖函数源码未发生实质变化时顶层 kernel cache_key 保持不变。 |
| `test_preload` | 验证特化数据可预加载同一 JIT 内核、复用原 hash 且免重新编译，并拒绝加载到不匹配函数。 |
| `test_reuse` | 验证同一参数重复启动十次只触发一次 JIT cache hook，即后续运行复用已编译内核。 |
| `test_toplevel_change` | 验证顶层直接依赖 function_1 的源码变化会使 kernel cache_key 改变。 |
| `test_use_builtin` | 验证使用 Python builtin float 不会被误当作可变全局，重复启动不产生全局值变化错误。 |

## `python/test/unit/runtime/test_compilation_listener.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_compile_stats` | 验证编译 listener 首次收到 cache miss、有效 target/hash 和各阶段耗时，再次运行收到 cache hit 且无 lowering/store 耗时。 |

## `python/test/unit/runtime/test_driver.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_is_lazy` | 验证 runtime driver 在模块重载后保持惰性未初始化，首次访问 active/default 时再创建合法 DriverBase。 |
| `test_kernel_in_thread` | 验证各线程按 Ascend 要求显式选择缓冲区所在 NPU 后，同一已编译 Triton Kernel 能在主线程和新线程中完成启动。 |

## `python/test/unit/runtime/test_launch.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_load_hook` | 验证 kernel load start/end hook 在加载时均被调用，并收到相同内核 hash。 |
| `test_memory_leak` | 验证已编译内核重复启动 100 次后 Python 跟踪内存增长低于 30 KB，不存在明显启动路径泄漏。 |
| `test_metadata` | 验证 launch_metadata 回调可从 grid 与参数构造延迟元数据，launch_enter_hook 读取到正确值。 |
| `test_multiple_hooks` | 验证可同时注册多个 kernel load start/end hook，单次加载会调用全部四个 hook。 |
| `test_pre_run_hooks` | 验证 JITFunction pre-run hook 在普通 launch 和显式 run 两条入口都会先清零输入，随后内核输出全 2。 |

## `python/test/unit/runtime/test_specialize.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_specialize_impl` | 以 96 组 host/native 参数组合核对 CUDABackend 与 HIPBackend 的 native specialization 结果和 Python 参考实现一致；该用例不启动 NPU Kernel，也不验证 AscendBackend。 |

## `python/test/unit/runtime/test_subproc.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_compile_in_forked_subproc_with_forced_gc` | 验证父进程编译后清缓存并 fork，子进程强制 GC 再编译时 MLIRContext/线程池资源可安全释放且进程正常退出。 |

## `python/test/unit/test_debug.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_device_assert_barrier` | 验证启用 TRITON_DEBUG 后，条件为真的 device_assert 内核完成同步且不会误触发断言。 |
| `test_static_assert` | 验证 tl.static_assert 对 true 可成功启动，对 false 抛出 CompileTimeAssertionFailure。 |

## `python/test/unit/test_debug_dump.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_fn_dump` | 验证 MLIR_ENABLE_DUMP 可开启全部或按函数名过滤 IR dump，匹配时输出 _kernel IR、不匹配时不输出。 |

## `python/test/unit/test_filecheck.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_filecheck_negative` | 验证 FileCheck 模式不匹配时抛出包含缺失 CHECK 文本的 ValueError。 |
| `test_filecheck_positive` | 验证 FileCheck 能在生成 IR 中匹配常量 42 及紧随其后的 anchor 调用。 |

## `python/test/unit/test_knobs.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_env_updated` | 验证直接设置 knob 会同步更新对应环境变量，包括布尔开关和 TRITON_HOME 路径。 |
| `test_knobs_scope` | 验证 knob scope 内可临时切换到环境值/显式值，退出后恢复原设置且保持环境读取规则。 |
| `test_knobs_utils` | 验证各类 knob 默认值、赋值、独立 copy 与 reset 均按设计工作且副本不污染原实例。 |
| `test_opt_bool` | 验证可选布尔 knob 在环境变量 0、1 和缺失三种状态下分别解析为 False、True、None。 |
| `test_read_env` | 验证多种真假字符串及路径/类/set 环境变量可经 refresh_knobs 解析到对应 knob 和派生目录。 |
| `test_set_knob_directly` | 验证显式 knob 值优先于后续环境变化，设置为 env 或删除后恢复环境值，并覆盖多种数据类型。 |
| `test_triton_home` | 验证默认及动态修改 TRITON_HOME/knob 后 cache、dump、override 三个目录都按根路径正确派生。 |

## `python/test/unit/tools/test_linear_layout.py`

| 测试函数 | 用途与核心通过条件 |
|---|---|
| `test_compose` | 验证两个一维恒等 LinearLayout 组合后仍把 reg 索引原样映射到 tensor 索引。 |
| `test_get_matrix_view_from_bases` | 验证由两组二维 bases 构造的 LinearLayout 导出预期的 4×4 单位矩阵视图。 |
| `test_get_matrix_view_identity` | 验证长度 4 的一维恒等 LinearLayout 导出预期 2×2 单位矩阵视图。 |
| `test_get_matrix_view_strided` | 验证长度 4、步长 2 的一维 LinearLayout 导出包含偏移位的预期矩阵视图。 |
| `test_identity_1d` | 验证一维恒等 LinearLayout 对 0 至 7 原样映射且映射为满射。 |
| `test_identity_2d` | 验证由 bases 构造的二维恒等布局把两个输入维度正确映射为行列输出。 |
| `test_invert` | 验证一维恒等布局求逆后可从输出恢复每个原始输入值。 |
| `test_invert_and_compose` | 验证 invert_and_compose 可把共享中间维的两个恒等布局转换为正确的 inp→out 映射。 |
| `test_operator_mul_disjoint_dims` | 验证两个输入/输出维度互不相交的 LinearLayout 相乘后各维仍独立正确映射。 |
| `test_operator_mul_identity` | 验证同名维度的长度 4 与长度 8 恒等布局相乘后扩展为长度 8 恒等映射。 |
| `test_zeros_1d` | 验证 zeros_1d 将所有输入映为 0，并按输出维大小正确判断是否满射。 |
