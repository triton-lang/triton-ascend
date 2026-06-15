# _ascend_constraints.py
# Ascend platform constraints for Triton API
# Fields: constraints (text list), example (optional file name)

CONSTRAINTS = {
    "triton.Config": {
        "constraints": [
            "``num_stages``: 适用于 SM80+ GPU 的软件流水线阶段数，Ascend 平台不适用",
            "``num_ctas``: 块集群特性仅适用于 SM90+ GPU，Ascend 平台不适用",
            "``maxnreg``: 对应 PTX .maxnreg 指令，非所有平台支持，Ascend 平台无效",
        ],
        "example":
        "triton.Config",
    },
    "triton.autotune": {
        "constraints": [
            "``use_cuda_graph``: CUDA Graph 是 NVIDIA 特性，Ascend 平台不适用",
        ],
        "example": "triton.autotune",
    },
    "triton.extension.buffer.language.alloc": {
        "constraints": [
            "DataType: Ascend supports int8, int16, int32, uint8, uint64, int64, fp32, bf16, bool. Does not support uint16, uint32, fp16.",
            "Shape: Each element must be a positive integer.",
            "Address space: must fit within the specified address space size limits.",
        ],
        "example":
        "triton.extension.buffer.language.alloc",
    },
    "triton.extension.buffer.language.fixpipe": {
        "constraints": [
            "DataType: Operates on L0C to UB data movement (Ascend A5/Ascend hardware specific).",
            "Source must be the result of a dot (matrix multiply) operation.",
            "Destination must be a buffer with UB memory scope.",
        ],
        "example":
        "triton.extension.buffer.language.fixpipe",
    },
    "triton.extension.buffer.language.to_buffer": {
        "constraints": [
            "Address space must be one of UB, L1, L0A, L0B, L0C.",
            "When using bind_buffer, tensor and bind_buffer must have identical shapes and element types.",
            "A tensor cannot be bound to multiple buffers.",
        ],
        "example":
        "triton.extension.buffer.language.to_buffer",
    },
    "triton.extension.buffer.language.to_tensor": {
        "constraints": [
            "Same type support constraints as alloc.",
        ],
        "example": "triton.extension.buffer.language.to_tensor",
    },
    "triton.heuristics": {
        "example": "triton.heuristics",
    },
    "triton.jit": {
        "example": "triton.jit",
    },
    "triton.language.abs": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.abs",
    },
    "triton.language.add": {
        "constraints": [
            "Ascend A3 对比 GPU 不支持 fp64",
        ],
        "example": "triton.language.add",
    },
    "triton.language.advance": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
            "仅允许通过调整order参数的顺序来表达转置语义，不能通过调整stride参数的顺序实现转置语义",
            "与较复杂的循环和分支语句搭配使用可能出现编译问题",
        ],
        "example":
        "triton.language.advance",
    },
    "triton.language.arange": {
        "constraints": [
            "DataType: Ascend does not support int32 (hardware limitation).",
            "start和end必须大于等于0",
            "end - start < 1048576 (TRITON_MAX_TENSOR_NUMEL)",
            "start和end必须是编译时常量（tl.constexpr）",
            "``output_type`` (Ascend extension): Ascend输出类型为uint16，GPU输出类型为int32",
            "``relaxed_requirement`` (Ascend extension): CUDA要求range=(end-start)必须为2的幂次方，Ascend无此要求",
        ],
        "example":
        "triton.language.arange",
    },
    "triton.language.argmax": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.argmax",
    },
    "triton.language.argmin": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.argmin",
    },
    "triton.language.associative_scan": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
            "reverse=True要求tl.load加载数据时对齐，不能使用mask过滤多余数据索引",
        ],
        "example":
        "triton.language.associative_scan",
    },
    "triton.language.atomic_add": {
        "constraints": [
            "DataType: Ascend does not support fp64, int64 (hardware limitation).",
            "``sem``: 只支持acq_rel",
            "``scope``: 只支持gpu",
        ],
        "example":
        "triton.language.atomic_add",
    },
    "triton.language.atomic_and": {
        "constraints": [
            "``sem``: 只支持acq_rel",
            "``scope``: 只支持gpu",
        ],
        "example": "triton.language.atomic_and",
    },
    "triton.language.atomic_cas": {
        "constraints": [
            "DataType: Ascend does not support bf16, fp64 (hardware limitation).",
            "``sem``: 只支持acq_rel",
            "``scope``: 只支持gpu",
        ],
        "example":
        "triton.language.atomic_cas",
    },
    "triton.language.atomic_max": {
        "constraints": [
            "DataType: Ascend does not support int64 (hardware limitation).",
            "``sem``: 只支持acq_rel",
            "``scope``: 只支持gpu",
        ],
        "example":
        "triton.language.atomic_max",
    },
    "triton.language.atomic_min": {
        "constraints": [
            "DataType: Ascend does not support int64 (hardware limitation).",
            "``sem``: 只支持acq_rel",
            "``scope``: 只支持gpu",
        ],
        "example":
        "triton.language.atomic_min",
    },
    "triton.language.atomic_or": {
        "constraints": [
            "``sem``: 只支持acq_rel",
            "``scope``: 只支持gpu",
        ],
        "example": "triton.language.atomic_or",
    },
    "triton.language.atomic_xchg": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
            "``sem``: 只支持acq_rel",
            "``scope``: 只支持gpu",
        ],
        "example":
        "triton.language.atomic_xchg",
    },
    "triton.language.atomic_xor": {
        "constraints": [
            "``sem``: 只支持acq_rel",
            "``scope``: 只支持gpu",
        ],
        "example": "triton.language.atomic_xor",
    },
    "triton.language.broadcast": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.broadcast",
    },
    "triton.language.broadcast_to": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
            "输入tensor的shape和目标shape的rank必须一致",
        ],
        "example":
        "triton.language.broadcast_to",
    },
    "triton.language.cast": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
            "``overflow_mode`` (Ascend extension): Ascend扩展参数，整数溢出处理模式（trunc/saturate），仅对整型有意义",
        ],
        "example":
        "triton.language.cast",
    },
    "triton.language.cat": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
            "``can_reorder``: Ascend和GPU都只支持can_reorder=True",
        ],
        "example":
        "triton.language.cat",
    },
    "triton.language.cdiv": {
        "constraints": [
            "DataType: Ascend does not support bool, uint16, uint32, uint64, uint8 (hardware limitation).",
            "输入范围：0~16777216",
        ],
        "example":
        "triton.language.cdiv",
    },
    "triton.language.ceil": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.ceil",
    },
    "triton.language.clamp": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.clamp",
    },
    "triton.language.compile_hint": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
            "hint_name 参数必须为字符串类型",
            "list 类型 hint_val 仅支持整数数组，不支持浮点数或混合类型",
        ],
        "example":
        "triton.language.compile_hint",
    },
    "triton.language.core.__rshift__": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64, uint8 (hardware limitation).",
            "``other``: 仅支持标量，不支持 tensor（即 x >> 2 合法，x >> y（y 为 tensor）暂不支持）",
        ],
    },
    "triton.language.cos": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.cos",
    },
    "triton.language.cumprod": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.cumprod",
    },
    "triton.language.cumsum": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.cumsum",
    },
    "triton.language.device_print": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
            "device_print 只能打印参与运算的结果值，无法打印纯粹用于访存的 offset 变量（编译器优化掉）",
            "特定情况下 device_print 会展开辅助 DMA 代码导致底层报错，功能优化中",
        ],
        "example":
        "triton.language.device_print",
    },
    "triton.language.div": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64, uint8 (hardware limitation).",
            "Ascend A3 对比 GPU 不支持 fp64",
        ],
        "example":
        "triton.language.div",
    },
    "triton.language.dot_scaled": {
        "constraints": [
            "DataType: Ascend does not support fp4, fp8 (hardware limitation).",
            "``lhs_k_pack``: 不支持 fp4/fp8 格式，矩阵解压缩能力缺失（硬件限制）",
            "``rhs_k_pack``: 不支持 fp4/fp8 格式，矩阵解压缩能力缺失（硬件限制）",
            "Ascend 不支持 fp4、fp8 格式（硬件限制）",
            "输入矩阵 lhs、rhs 推荐输入范围为 [-5, 5]",
        ],
        "example":
        "triton.language.dot_scaled",
    },
    "triton.language.equal": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
    },
    "triton.language.erf": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.erf",
    },
    "triton.language.exp": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.exp",
    },
    "triton.language.exp2": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.exp2",
    },
    "triton.language.expand_dims": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.expand_dims",
    },
    "triton.language.extra.ascend.libdevice.index_select_simd": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
            "``index``: index 数据类型必须为 int32 或 int64",
            "``dim``: dim 不能为尾轴（最后一个维度），即 dim < len(src_shape) - 1",
            "GPU 平台不支持此操作（Ascend 专用 intrinsic）",
            "dim 不支持在尾轴（最后一个维度）上执行 index_select 操作",
            "不检查 index 中的索引是否越界，用户需自行保证索引合法性",
            "index 必须是 1D 张量",
        ],
    },
    "triton.language.extra.cann.extension.ascend_address_space": {
        "constraints": [
            "Provides UB, L1, L0A, L0B, L0C address space constants for use with bl.alloc.",
        ],
        "example": "triton.language.extra.cann.extension.ascend_address_space",
    },
    "triton.language.extra.cann.extension.copy": {
        "constraints": [
            "Source must be in UB address space. Destination must be L1 or UB address space.",
            "Source and destination must have the same data type and shape.",
        ],
        "example":
        "triton.language.extra.cann.extension.copy",
    },
    "triton.language.extra.cann.extension.copy_from_ub_to_l1": {
        "constraints": [
            "Deprecated: use al.copy instead.",
            "Source must be in UB address space. Destination must be in L1 address space.",
            "Source and destination must have the same data type and shape.",
        ],
        "example":
        "triton.language.extra.cann.extension.copy_from_ub_to_l1",
    },
    "triton.language.extra.cann.extension.debug_barrier": {
        "constraints": [
            "sync_mode: must be a SYNC_IN_VF enum value.",
            "Intended for use within an al.scope context.",
        ],
        "example": "triton.language.extra.cann.extension.debug_barrier",
    },
    "triton.language.extra.cann.extension.scope": {
        "constraints": [
            "core_mode: must be 'vector', 'cube', 'SIMT', or 'SIMD'.",
            "Each kernel supports one cube scope and one vector scope; they execute in parallel.",
            "Explicit synchronization (sync_block_set/sync_block_wait) required for cross-scope data dependencies.",
        ],
        "example":
        "triton.language.extra.cann.extension.scope",
    },
    "triton.language.extra.cann.extension.sub_vec_id": {
        "constraints": [
            "Only valid in mixed AIC+AIV scenarios (Cube + Vector cores).",
            "Using it in pure-Cube or pure-Vector kernels will cause a compilation error.",
        ],
        "example":
        "triton.language.extra.cann.extension.sub_vec_id",
    },
    "triton.language.extra.cann.extension.subview": {
        "constraints": [
            "Offsets, sizes, and strides must be non-negative.",
            "Each dimension size cannot exceed the original buffer's corresponding dimension.",
            "All stride elements must be 1.",
            "Offset must be 32-byte aligned.",
        ],
        "example":
        "triton.language.extra.cann.extension.subview",
    },
    "triton.language.extra.cann.extension.sync_block_all": {
        "constraints": [
            "mode: must be one of 'all_cube', 'all_vector', 'all', 'all_sub_vector'.",
            "event_id: must be in range [0, 15].",
        ],
        "example":
        "triton.language.extra.cann.extension.sync_block_all",
    },
    "triton.language.extra.cann.extension.sync_block_set": {
        "constraints": [
            "sender/receiver: must be 'cube' or 'vector', and must differ from each other.",
            "event_id: must be in range [0, 15].",
            "Must be used within an al.scope context matching the sender's core type.",
        ],
        "example":
        "triton.language.extra.cann.extension.sync_block_set",
    },
    "triton.language.extra.cann.extension.sync_block_wait": {
        "constraints": [
            "sender/receiver: must be 'cube' or 'vector', and must differ from each other.",
            "event_id: must match the ID used by the corresponding sync_block_set.",
            "Must be used within an al.scope context matching the receiver's core type.",
        ],
        "example":
        "triton.language.extra.cann.extension.sync_block_wait",
    },
    "triton.language.extract_slice": {
        "constraints": [
            "DataType: Ascend does not support bool (hardware limitation).",
        ],
        "example": "triton.language.extract_slice",
    },
    "triton.language.flip": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.flip",
    },
    "triton.language.floor": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.floor",
    },
    "triton.language.floordiv": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.floordiv",
    },
    "triton.language.full": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.full",
    },
    "triton.language.gather": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.gather",
    },
    "triton.language.get_element": {
        "constraints": [
            "DataType: Ascend does not support bool (hardware limitation).",
        ],
        "example": "triton.language.get_element",
    },
    "triton.language.greater_than": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64 (hardware limitation).",
        ],
    },
    "triton.language.histogram": {
        "constraints": [
            "当前triton3.2版本暂未支持mask参数",
            "input输入范围限制在 [0, num_bins-1]",
        ],
        "example": "triton.language.histogram",
    },
    "triton.language.inline_asm_elementwise": {
        "constraints": [
            "DataType: Ascend does not support bf16, bool, fp16, fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
            "内联汇编寄存器仅支持 int64(s64) 和 float32(f32)",
            "约束限制仅支持 'l'（LLVM 约束）",
            "目前仅支持输入一维张量，计算高维张量需展开",
        ],
        "example":
        "triton.language.inline_asm_elementwise",
    },
    "triton.language.insert_slice": {
        "constraints": [
            "DataType: Ascend does not support bool (hardware limitation).",
        ],
        "example": "triton.language.insert_slice",
    },
    "triton.language.join": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.join",
    },
    "triton.language.less_equal": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
    },
    "triton.language.less_than": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
    },
    "triton.language.load": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
            "``cache_modifier``: has no effect on Ascend.",
            "``eviction_policy``: has no effect on Ascend.",
            "``volatile``: has no effect on Ascend.",
            "``padding_option``: is not supported on Ascend.",
            "与分支、循环语句搭配使用时的泛化性问题，pointer和mask的计算过程涉及较复杂的循环和分支语句可能出现编译问题",
        ],
        "example":
        "triton.language.load",
    },
    "triton.language.load_tensor_descriptor": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
            "make_tensor_descriptor / load_tensor_descriptor / store_tensor_descriptor 需配套使用，不能与 tl.load() / tl.store() 混用",
            "Triton 3.2.0 存在部分函数（如cast）的兼容性问题，建议升级至 3.4.0",
        ],
        "example":
        "triton.language.load_tensor_descriptor",
    },
    "triton.language.log": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.log",
    },
    "triton.language.log2": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.log2",
    },
    "triton.language.make_block_ptr": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
            "仅允许通过调整order参数的顺序来表达转置语义，不能通过调整stride参数的顺序实现转置语义",
            "与分支、循环语句搭配使用时的泛化性问题",
        ],
        "example":
        "triton.language.make_block_ptr",
    },
    "triton.language.make_tensor_descriptor": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
            "``strides``: 前面的维度必须是16字节的整数倍，最后一维必须是连续存储的",
            "make_tensor_descriptor / load_tensor_descriptor / store_tensor_descriptor 需配套使用，不能与 tl.load() / tl.store() 混用",
            "不支持padding_option入参",
            "Triton 3.2.0 存在部分函数（如cast）的兼容性问题，建议升级至 3.4.0",
        ],
        "example":
        "triton.language.make_tensor_descriptor",
    },
    "triton.language.max": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64 (hardware limitation).",
            "``return_indices``: 当axis=None时不支持return_indices=True",
        ],
        "example":
        "triton.language.max",
    },
    "triton.language.max_constancy": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.max_constancy",
    },
    "triton.language.max_contiguous": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.max_contiguous",
    },
    "triton.language.maximum": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.maximum",
    },
    "triton.language.min": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64 (hardware limitation).",
            "``return_indices``: 当axis=None时不支持return_indices=True",
        ],
        "example":
        "triton.language.min",
    },
    "triton.language.minimum": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.minimum",
    },
    "triton.language.mod": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64, uint8 (hardware limitation).",
            "Ascend A3 对比 GPU 不支持 fp64",
        ],
        "example":
        "triton.language.mod",
    },
    "triton.language.neg": {
        "constraints": [
            "DataType: Ascend does not support bool, fp64, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.neg",
    },
    "triton.language.parallel": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
            "``bind_sub_block``: bind_sub_block 为 True 时在 IR 中是否能体现与 range 的差异，功能是否实现待验证",
        ],
        "example":
        "triton.language.parallel",
    },
    "triton.language.rand": {
        "constraints": [
            "seed 参数不支持 fp16/fp32/fp64/bf16 类型，仅支持整数和 bool 类型",
            "相对社区能力暂不支持（具体差异待补充）",
        ],
        "example": "triton.language.rand",
    },
    "triton.language.randint": {
        "constraints": [
            "seed 参数不支持 fp16/fp32/fp64/bf16 类型，仅支持整数和 bool 类型",
            "相对社区能力暂不支持（具体差异待补充）",
        ],
        "example": "triton.language.randint",
    },
    "triton.language.randint4x": {
        "constraints": [
            "seed 参数不支持 fp16/fp32/fp64/bf16 类型，仅支持整数和 bool 类型",
            "相对社区能力暂不支持（具体差异待补充）",
        ],
        "example": "triton.language.randint4x",
    },
    "triton.language.randn": {
        "constraints": [
            "seed 参数不支持 fp16/fp32/fp64/bf16 类型，仅支持整数和 bool 类型",
            "相对社区能力暂不支持（具体差异待补充）",
        ],
        "example": "triton.language.randn",
    },
    "triton.language.range": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64, uint8 (hardware limitation).",
            "``disallow_acc_multi_buffer``: Ascend 上相关功能不全",
            "``flatten``: Ascend 上相关功能不全",
            "``warp_specialize``: 仅 Blackwell GPU 支持，Ascend 无效",
            "``disable_licm``: Ascend 上相关功能不全",
        ],
        "example":
        "triton.language.range",
    },
    "triton.language.ravel": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.ravel",
    },
    "triton.language.reduce": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
            "keep_dims=True需更多测试规格覆盖，目前已测3D dim=2情况下支持",
        ],
        "example":
        "triton.language.reduce",
    },
    "triton.language.reshape": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
            "``can_reorder``: 仅支持False",
        ],
        "example":
        "triton.language.reshape",
    },
    "triton.language.rsqrt": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.rsqrt",
    },
    "triton.language.sigmoid": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.sigmoid",
    },
    "triton.language.sin": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.sin",
    },
    "triton.language.softmax": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
            "``ieee_rounding``: Ascend 不使用 PTX 精确舍入，该参数无效",
        ],
        "example":
        "triton.language.softmax",
    },
    "triton.language.sort": {
        "constraints": [
            "DataType: Ascend does not support bool, fp64, int32, int64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.sort",
    },
    "triton.language.split": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.split",
    },
    "triton.language.sqrt": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.sqrt",
    },
    "triton.language.static_print": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.static_print",
    },
    "triton.language.static_range": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.static_range",
    },
    "triton.language.store": {},
    "triton.language.store_tensor_descriptor": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.store_tensor_descriptor",
    },
    "triton.language.sub": {
        "constraints": [
            "Ascend A3 对比 GPU 不支持 fp64",
        ],
        "example": "triton.language.sub",
    },
    "triton.language.sync_block_all": {
        "constraints": [
            "``mode``: 必须为 'all_cube'、'all_vector' 或 'all' 之一",
            "``event_id``: 取值范围 0-15",
            "``status`` (Ascend extension): ascend_added",
            "``note`` (Ascend extension): Ascend Cube-Vector 架构同步原语，上游 Triton 不存在",
        ],
    },
    "triton.language.sync_block_set": {
        "constraints": [
            "``sender``: 发送方核心类型，必须为 'cube' 或 'vector",
            "``receiver``: 接收方核心类型，必须为 'cube' 或 'vector'，且不能与 sender 相同",
            "``event_id``: 取值范围 0-15（共 16 个独立事件）",
            "``status`` (Ascend extension): ascend_added",
            "``note`` (Ascend extension): Ascend Cube-Vector 架构同步原语，上游 Triton 不存在",
        ],
    },
    "triton.language.sync_block_wait": {
        "constraints": [
            "``sender``: 发送方核心类型，必须为 'cube' 或 'vector",
            "``receiver``: 接收方核心类型，必须为 'cube' 或 'vector'，且不能与 sender 相同",
            "``event_id``: 必须与对应 sync_block_set 使用的 ID 一致",
            "``status`` (Ascend extension): ascend_added",
            "``note`` (Ascend extension): Ascend Cube-Vector 架构同步原语，上游 Triton 不存在",
        ],
    },
    "triton.language.tensor.__invert__": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
        ],
    },
    "triton.language.tensor.__neg__": {
        "constraints": [
            "DataType: Ascend does not support bool, fp64, uint16, uint32, uint64 (hardware limitation).",
        ],
    },
    "triton.language.tensor.logical_and": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
    },
    "triton.language.tensor.logical_or": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
    },
    "triton.language.trans": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
            "不支持维度高于8的转置",
        ],
        "example":
        "triton.language.trans",
    },
    "triton.language.umulhi": {
        "constraints": [
            "DataType: Ascend does not support int64 (hardware limitation).",
        ],
        "example": "triton.language.umulhi",
    },
    "triton.language.view": {
        "constraints": [
            "DataType: Ascend does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.view",
    },
    "triton.language.where": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.where",
    },
    "triton.language.xor_sum": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
            "keep_dims=True需更多测试规格覆盖，目前已测3D dim=2情况下支持",
        ],
        "example":
        "triton.language.xor_sum",
    },
    "triton.language.zeros": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.zeros",
    },
    "triton.language.zeros_like": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.zeros_like",
    },
}
