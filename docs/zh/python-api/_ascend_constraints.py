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
            "DataType: Operates on L0C to UB data movement (Ascend 950/Ascend hardware specific).",
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
    "triton.language.permute": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.permute",
    },
    "triton.heuristics": {
        "example": "triton.heuristics",
    },
    "triton.jit": {
        "example": "triton.jit",
    },
    "triton.language.abs": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64.",
        ],
        "example":
        "triton.language.abs",
    },
    "triton.language.add": {
        "constraints": [
            "Ascend A3 对比 GPU 不支持 fp64",
        ],
        "example": "triton.language.add",
    },
    "triton.language.advance": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "Compilation issues may arise when used with complex loops and branch statements."
        ],
        "example":
        "triton.language.advance",
    },
    "triton.language.arange": {
        "constraints": [
            "Arguments must be tl.constexpr of uint or int type (int64 not supported).",
            "start and end must be greater than or equal to 0.",
            "end - start < 1048576 (TRITON_MAX_TENSOR_NUMEL), applicable to both NV and Ascend.",
            "``relaxed_requirement`` (Ascend extension): CUDA requires range=(end-start) to be a power of 2, Ascend has no such requirement.",
        ],
        "example":
        "triton.language.arange",
    },
    "triton.language.argmax": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64; Ascend 950 does not fp64, fp8e4(E4M3), fp8e5(E5M2) (hardware limitation).",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.argmax",
    },
    "triton.language.argmin": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64; Ascend 950 does not fp64, fp8e4(E4M3), fp8e5(E5M2) (hardware limitation).",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.argmin",
    },
    "triton.language.associative_scan": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64 (hardware limitation).",
            "reverse=True requires alignment when loading data with tl.load, and mask cannot be used to filter redundant data indices",
        ],
        "example":
        "triton.language.associative_scan",
    },
    "triton.language.assume": {
        "example": "triton.language.assume",
    },
    "triton.language.atomic_add": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
            "``sem``: Ascend does not support \"acquire\",\"release\",\"relaxed\"",
            "``scope``: Ascend does not support \"cta\",\"sys\"",
        ],
        "example":
        "triton.language.atomic_add",
    },
    "triton.language.atomic_and": {
        "constraints": [
            "``sem``: Ascend does not support \"acquire\",\"release\",\"relaxed\"",
            "``scope``: Ascend does not support \"cta\",\"sys\"",
        ],
        "example":
        "triton.language.atomic_and",
    },
    "triton.language.atomic_cas": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
            "``sem``: Ascend does not support \"acquire\",\"release\",\"relaxed\"",
            "``scope``: Ascend does not support \"cta\",\"sys\"",
        ],
        "example":
        "triton.language.atomic_cas",
    },
    "triton.language.atomic_max": {
        "constraints": [
            "``sem``: Ascend does not support \"acquire\",\"release\",\"relaxed\"",
            "``scope``: Ascend does not support \"cta\",\"sys\"",
        ],
        "example":
        "triton.language.atomic_max",
    },
    "triton.language.atomic_min": {
        "constraints": [
            "``sem``: Ascend does not support \"acquire\",\"release\",\"relaxed\"",
            "``scope``: Ascend does not support \"cta\",\"sys\"",
        ],
        "example":
        "triton.language.atomic_min",
    },
    "triton.language.atomic_or": {
        "constraints": [
            "``sem``: Ascend does not support \"acquire\",\"release\",\"relaxed\"",
            "``scope``: Ascend does not support \"cta\",\"sys\"",
        ],
        "example":
        "triton.language.atomic_or",
    },
    "triton.language.atomic_xchg": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
            "``sem``: Ascend does not support \"acquire\",\"release\",\"relaxed\"",
            "``scope``: Ascend does not support \"cta\",\"sys\"",
        ],
        "example":
        "triton.language.atomic_xchg",
    },
    "triton.language.atomic_xor": {
        "constraints": [
            "``sem``: Ascend does not support \"acquire\",\"release\",\"relaxed\"",
            "``scope``: Ascend does not support \"cta\",\"sys\"",
        ],
        "example":
        "triton.language.atomic_xor",
    },
    "triton.language.broadcast": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.broadcast",
    },
    "triton.language.broadcast_to": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
            "The rank of the input tensor's shape must match the rank of the target shape.",
        ],
        "example":
        "triton.language.broadcast_to",
    },
    "triton.language.cast": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.cast",
    },
    "triton.language.cat": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64 (hardware limitation).",
            "``can_reorder``: Both Ascend and GPU only support can_reorder=True",
        ],
        "example":
        "triton.language.cat",
    },
    "triton.language.cdiv": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/bool, \
                Ascend 950 does not support bool.",
            "Input Range: 0~16777216",
        ],
        "example":
        "triton.language.cdiv",
    },
    "triton.language.ceil": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.ceil",
    },
    "triton.language.clamp": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
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
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.cos",
    },
    "triton.language.cumprod": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64 (hardware limitation).",
            "reverse=True requires alignment when loading data with tl.load, and mask cannot be used to filter redundant data indices",
        ],
        "example":
        "triton.language.cumprod",
    },
    "triton.language.cumsum": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64 (hardware limitation).",
            "reverse=True requires alignment when loading data with tl.load, and mask cannot be used to filter redundant data indices",
        ],
        "example":
        "triton.language.cumsum",
    },
    "triton.language.debug_barrier": {
        "example": "triton.language.debug_barrier",
    },
    "triton.language.device_assert": {
        "example": "triton.language.device_assert",
    },
    "triton.language.device_print": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "``prefix``: the first argument must be a string prefix; omitting it causes a compilation error.",
            "Set environment variable ``TRITON_DEVICE_PRINT=1`` to enable.",
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
    "triton.language.div_rn": {
        "constraints": [],
        "example": "triton.language.div_rn",
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
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64.",
        ],
    },
    "triton.language.erf": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.erf",
    },
    "triton.language.exp": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.exp",
    },
    "triton.language.exp2": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.exp2",
    },
    "triton.language.expand_dims": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.expand_dims",
    },
    "triton.language.fdiv": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.fdiv",
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
    "triton.language.extra.cann.extension.extract_slice": {
        "constraints": [
            "DataType: Ascend does not support bool (hardware limitation).",
        ],
        "example": "triton.language.extra.cann.extension.extract_slice",
    },
    "triton.language.extra.cann.extension.get_element": {
        "constraints": [
            "DataType: Ascend does not support bool (hardware limitation).",
        ],
        "example": "triton.language.extra.cann.extension.get_element",
    },
    "triton.language.extra.cann.extension.insert_slice": {
        "constraints": [
            "DataType: Ascend does not support bool (hardware limitation).",
        ],
        "example": "triton.language.extra.cann.extension.insert_slice",
    },
    "triton.language.extra.cann.extension.multibuffer": {
        "constraints": [
            "only support bufferize equals 2.",
        ],
        "example": "triton.language.extra.cann.extension.multibuffer",
    },
    "triton.language.extra.cann.extension.parallel": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "When using ``parallel`` as an iterator, the operations within the loop body must have no dependencies and no conflicts between them:",
            "Memory access (load/store) operations can be executed in parallel.",
            "Compute operations are allowed, but there must be no more than one compute operation. Multiple compute operations would produce intermediate UB buffers, which cannot be accessed in parallel.",
        ],
        "example":
        "triton.language.extra.cann.extension.parallel",
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
    "triton.language.flip": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.flip",
    },
    "triton.language.floor": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.floor",
    },
    "triton.language.fma": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.fma",
    },
    "triton.language.floordiv": {
        "constraints": [
            "DataType: Ascend does not support uint16, uint32, uint64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.floordiv",
    },
    "triton.language.full": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.full",
    },
    "triton.language.gather": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.gather",
    },
    "triton.language.greater_than": {
        "constraints": [
            "DataType: Ascend does not support fp64, uint16, uint32, uint64 (hardware limitation).",
        ],
    },
    "triton.language.greater_equal": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64.",
        ],
    },
    "triton.language.histogram": {
        "constraints": [
            "triton3.2 does not support mask",
            "input range is limited to [0, num_bins-1]",
        ],
        "example": "triton.language.histogram",
    },
    "triton.language.inline_asm_elementwise": {
        "constraints": [
            "Inline assembly registers only support int64 (s64) and float32 (f32).",
            "Only the 'l' LLVM constraint is supported.",
            "Only 1-D input tensors are supported; higher-dimensional tensors must be flattened.",
        ],
        "example":
        "triton.language.inline_asm_elementwise",
    },
    "triton.language.interleave": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.interleave",
    },
    "triton.language.join": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.join",
    },
    "triton.language.less_equal": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64.",
        ],
    },
    "triton.language.less_than": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64.",
        ],
    },
    "triton.language.load": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "``cache_modifier``: Ascend A5 SIMT-only accepts the Triton upstream strings (``''``, ``'.ca'``, ``'.cg'``, ``'.cv'``); the backend folds them to a two-level cache / uncache mapping (``'.ca'`` -> cache, ``'.cg'`` / ``'.cv'`` -> uncache). Effective on global memory; ignored on shared memory. A2/A3 and non-SIMT-only paths ignore the modifier.",
            "``eviction_policy``: has no effect on Ascend.",
            "``volatile``: ``bool``; on Ascend A5 SIMT-only, guarantees memory ordering and prevents the compiler/hardware from hoisting or eliding the load. Effective on both global and shared memory. A2/A3 and non-SIMT-only paths ignore the modifier.",
            "Compatibility issues with branch and loop statements: \
                Complex pointer and mask calculations involving branches or loops may cause compilation failures.",
        ],
        "example":
        "triton.language.load",
    },
    "triton.language.load_tensor_descriptor": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "`make_tensor_descriptor`, `load_tensor_descriptor` and `store_tensor_descriptor` \
                must be used as a suite in Triton 3.2.x. Do not mix them with `tl.load()` or `tl.store()`.",
            "Compatibility issues exist for certain functions (e.g. cast) in Triton 3.2.x."
        ],
        "example":
        "triton.language.load_tensor_descriptor",
    },
    "triton.language.log": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.log",
    },
    "triton.language.log2": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.log2",
    },
    "triton.language.make_block_ptr": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "Transpose logic can only be implemented by adjusting the `order` parameter. \
                Do not reorder strides to achieve transpose.",
            "Compatibility issues may occur when used together with branch and loop statements."
        ],
        "example":
        "triton.language.make_block_ptr",
    },
    "triton.language.make_tensor_descriptor": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "`make_tensor_descriptor`, `load_tensor_descriptor` and `store_tensor_descriptor` \
                must be used as a suite in Triton 3.2.x. Do not mix them with `tl.load()` or `tl.store()`.",
            "Compatibility issues exist for certain functions (e.g. cast) in Triton 3.2.x."
        ],
        "example":
        "triton.language.make_tensor_descriptor",
    },
    "triton.language.max": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64; Ascend 950 does not fp64, fp8e4(E4M3), fp8e5(E5M2) (hardware limitation).",
            "``return_indices``: return_indices=True is not supported when axis=None.",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.max",
    },
    "triton.language.max_constancy": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "``values``: rank must match the rank of ``input`` (e.g. ``[1, 1]`` for a 2-D input).",
        ],
        "example":
        "triton.language.max_constancy",
    },
    "triton.language.max_contiguous": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "``values``: rank must match the rank of ``input`` (e.g. ``[1, 1]`` for a 2-D input).",
        ],
        "example":
        "triton.language.max_contiguous",
    },
    "triton.language.maximum": {
        "constraints": [
            "DataType: Ascend does not support fp64 (hardware limitation).",
        ],
        "example": "triton.language.maximum",
    },
    "triton.language.min": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64; Ascend 950 does not fp64, fp8e4(E4M3), fp8e5(E5M2) (hardware limitation).",
            "``return_indices``: return_indices=True is not supported when axis=None.",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
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
    "triton.language.multiple_of": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "``values``: describes the divisibility of the first value along each dimension, so its rank must match ``input`` (e.g. ``[1, 1]`` for a 2-D input).",
        ],
        "example":
        "triton.language.multiple_of",
    },
    "triton.language.neg": {
        "constraints": [
            "DataType: Ascend does not support bool, fp64, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.neg",
    },
    "triton.language.rand": {
        "example": "triton.language.rand",
    },
    "triton.language.randint": {
        "example": "triton.language.randint",
    },
    "triton.language.randint4x": {
        "example": "triton.language.randint4x",
    },
    "triton.language.randn": {
        "example": "triton.language.randn",
    },
    "triton.language.range": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "``disallow_acc_multi_buffer``, ``flatten``, ``disable_licm``: related functionality is incomplete on Ascend.",
            "``warp_specialize``: only supported on Blackwell GPU; has no effect on Ascend.",
        ],
        "example":
        "triton.language.range",
    },
    "triton.language.ravel": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
        ],
        "example":
        "triton.language.ravel",
    },
    "triton.language.reduce": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64; Ascend 950 does not fp8e4(E4M3), fp8e5(E5M2) (hardware limitation).",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.reduce",
    },
    "triton.language.reshape": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
            "``can_reorder``: Only supports False",
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
            "``ieee_rounding``: Ascend does not use PTX precise rounding, this parameter is invalid.",
        ],
        "example":
        "triton.language.softmax",
    },
    "triton.language.sort": {
        "constraints": [
            "Note: tl.sort may have precision problems in big shape or multidimensional shape, more recommendeded to use triton.language.extra.cann.extension.sort."
        ],
        "example":
        "triton.language.sort",
    },
    "triton.language.topk": {
        "constraints": [
            "DataType: Ascend does not support bool, fp64, int32, int64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.topk",
    },
    "triton.language.extra.cann.extension.sort": {
        "constraints": [
            "DataType: Ascend does not support bool, fp64, int32, int64, uint8 (hardware limitation).",
        ],
        "example": "triton.language.extra.cann.extension.sort",
    },
    "triton.language.split": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
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
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
        ],
        "example":
        "triton.language.static_print",
    },
    "triton.language.static_assert": {
        "constraints": [
            "``cond``: must be a compile-time constant (tl.constexpr); a non-constant condition causes a compilation error.",
        ],
        "example":
        "triton.language.static_assert",
    },
    "triton.language.static_range": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "``start``, ``end``, ``step`` must be compile-time constants (tl.constexpr).",
        ],
        "example":
        "triton.language.static_range",
    },
    "triton.language.store": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "``cache_modifier``: Ascend A5 SIMT-only accepts the Triton upstream strings (``''``, ``'.wb'``, ``'.cg'``, ``'.cs'``, ``'.wt'``); the backend folds them to a two-level cache / uncache mapping (``'.wb'`` -> cache, ``'.cg'`` / ``'.cs'`` / ``'.wt'`` -> uncache). Effective on global memory; ignored on shared memory. A2/A3 and non-SIMT-only paths ignore the modifier.",
            "``eviction_policy``: has no effect on Ascend.",
            "``volatile``: not accepted (the Triton upstream ``tl.store`` signature does not expose a ``volatile=`` keyword).",
            "Compatibility issues with branch and loop statements: \
                Complex pointer and mask calculations involving branches or loops may cause compilation failures.",
        ],
        "example":
        "triton.language.store",
    },
    "triton.language.store_tensor_descriptor": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp64, \
                Ascend 950 does not support fp64 (hardware limitation).",
            "`make_tensor_descriptor`, `load_tensor_descriptor` and `store_tensor_descriptor` \
                must be used as a suite in Triton 3.2.x. Do not mix them with `tl.load()` or `tl.store()`.",
            "Compatibility issues exist for certain functions (e.g. cast) in Triton 3.2.x."
        ],
        "example":
        "triton.language.store_tensor_descriptor",
    },
    "triton.language.sub": {
        "constraints": [
            "Ascend A3 对比 GPU 不支持 fp64",
        ],
        "example": "triton.language.sub",
    },
    "triton.language.sum": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64.",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.sum",
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
    "triton.language.tensor.logical_and": {
        "constraints": [],
    },
    "triton.language.tensor.logical_or": {
        "constraints": [],
    },
    "triton.language.trans": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
            "Transpose on dimensions greater than 8 is not supported.",
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
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
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
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64; Ascend 950 does not fp8e4(E4M3), fp8e5(E5M2).",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.xor_sum",
    },
    "triton.language.zeros": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.zeros",
    },
    "triton.language.zeros_like": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64 (hardware limitation).",
        ],
        "example": "triton.language.zeros_like",
    },
    "triton.language.extra.cann.extension.index_put": {
        "constraints": [
            "Only supported on Ascend 950.",
            "``ptr.dtype``: only supports float16, bfloat16, float32.",
            "``ptr`` and ``value`` must have the same rank.",
            "``index``: must be an integer tensor. If ``index.rank`` != 1, it will be reshaped to 1D.",
            "``index.numel``: must equal ``value.shape[dim]``.",
            "``value``: supports 2~5D tensors.",
            "``dim``: must satisfy 0 <= dim < rank(value) - 1.",
        ],
        "example":
        "triton.language.extra.cann.extension.index_put",
    },
    "triton.language.extra.cann.extension.gather_out_to_ub": {
        "constraints": [
            "Only supported on Ascend 950.",
            "``src.dtype``: only supports float16, bfloat16, float32.",
            "``src`` and ``index`` must have the same rank.",
            "``index``: must be an integer tensor, with rank in [1, 5].",
            "``dim``: must satisfy 0 <= dim < rank(index).",
            "``other``: must be a scalar value.",
            "For every dimension ``i`` not equal to ``dim``, ``index.size[i]`` <= ``src.size[i]``.",
            "The output shape is the same as ``index.shape``.",
        ],
        "example":
        "triton.language.extra.cann.extension.gather_out_to_ub",
    },
    "triton.language.extra.cann.extension.scatter_ub_to_out": {
        "constraints": [
            "Only supported on Ascend 950.",
            "``ptr.dtype``: only supports float16, bfloat16, float32.",
            "``ptr``, ``index`` and ``value`` must have the same rank.",
            "``index``: must be an integer tensor, with rank in [1, 5].",
            "``dim``: must satisfy 0 <= dim < rank(index).",
            "For every dimension ``i`` not equal to ``dim``, ``index.size[i]`` <= ``ptr.size[i]``.",
            "The output shape is the same as ``index.shape``.",
        ],
        "example":
        "triton.language.extra.cann.extension.scatter_ub_to_out",
    },
}
