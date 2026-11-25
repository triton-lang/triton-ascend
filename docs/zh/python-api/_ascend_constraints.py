# _ascend_constraints.py
# Ascend platform constraints for Triton / TLA API
# Fields: constraints (text list), example (optional file name)

# TLA entries use _TLA_PREFIX; their examples live under docs/zh/examples/tla.
_TLA_PREFIX = "triton.language.extra.tla"

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
            "DataType: Ascend A2/A3/950 does not support fp8/fp64.",
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
            "DataType: Ascend A2/A3/950 does not support uint8/uint16/uint32/uint64/fp8/fp64.",
        ],
        "example": "triton.language.div",
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
    "triton.language.extra.cann.extension.index_select_simd": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64 (hardware limitation).",
            "``index``: The data type of the index must be int32 or int64.",
            "``dim``: The dimension cannot be the trailing axis (the last dimension), i.e., dim < len(src_shape) - 1.",
            "The index_select operation is not supported along the trailing axis (the last dimension).",
            "Out-of-bounds indices are not checked; users must ensure index validity on their own.",
            "The index must be a 1D tensor.",
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
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64.",
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
            "DataType: Ascend A2/A3 does not support uint16/uint32/uint64/fp8/fp64, \
                Ascend 950 does not support fp8/fp64.",
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
            "Multi-input reduce fallback notation: ``K`` is the number of input tensors; ``T[i]`` and ``Res[i]`` are input tensor ``i`` and its scalar result, respectively; ``L = shape(T[0])[0]`` is the reduction length; ``x[i]`` is the current element passed to combine-region argument ``i``; and ``s[i]`` is the carried state passed to argument ``K + i`` (``0 <= i < K``).",
            "``R`` is the number of effective combine-region operations: only operations backward-reachable from ``tt.reduce.return`` are counted, while ``ext``, ``trunc``, and ``bitcast`` are ignored. The explicit ``scf.for`` fallback is attempted only when ``K > 2`` and ``R != 1``; ``R = 1`` keeps the existing single-reduction lowering.",
            "Input shape and axis: every ``T[i]`` must be a ``RankedTensor`` with the same static rank-1 shape, ``axis = 0``, and a positive static ``L``. Different ``T[i]`` values may have different element types.",
            "Results: there must be exactly ``K`` results, and every ``Res[i]`` must be a scalar with the same element type as its corresponding ``T[i]``.",
            "Combine region: it must have exactly ``2K`` block arguments and ``K`` ``tt.reduce.return`` operands. The fallback initializes ``s[i]`` from ``T[i][0]`` and combines elements in order for ``i = 1 ... L - 1``.",
            "Semantics and performance: this path clones the combine region into a sequential ``scf.for``. It can have a different floating-point rounding order from a tree reduction, and it does not guarantee that every multi-input ``reduce`` will pass later lowering or have equivalent performance.",
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
    "triton.language.extra.cann.extension.dot": {
        "constraints": [
            "DataType: Ascend supports f16, bf16, f32, int8. A and B must have the same dtype.",
            "format_a/b/c: must be ``\"fractal\"`` (zN 4D), ``\"nd\"`` (2D), or ``\"\"`` (default ND).",
            "Fractal operand must be 4D (zN); ND operand must be 2D.",
            "Fractal block_col must be 32/elem_bytes (f16/bf16→16, f32→8, int8→32).",
            "Fractal block_row must be 16; for int8 the right operand requires 32 (zN[N/32, K/32, 32, 32]).",
            "Output dtype: f32 for float inputs, i32 for int8 (cube L0C accumulator dtype).",
        ],
        "example":
        "triton.language.extra.cann.extension.dot",
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
            "DataType: Ascend A2/A3/950 does not support fp8/fp64.",
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
    "triton.language.extra.cann.libdevice.acos": {
        "example": "triton.language.extra.cann.libdevice.acos",
    },
    "triton.language.extra.cann.libdevice.acosh": {
        "example": "triton.language.extra.cann.libdevice.acosh",
    },
    "triton.language.extra.cann.libdevice.asin": {
        "example": "triton.language.extra.cann.libdevice.asin",
    },
    "triton.language.extra.cann.libdevice.asinh": {
        "example": "triton.language.extra.cann.libdevice.asinh",
    },
    "triton.language.extra.cann.libdevice.atan": {
        "example": "triton.language.extra.cann.libdevice.atan",
    },
    "triton.language.extra.cann.libdevice.atan2": {
        "example": "triton.language.extra.cann.libdevice.atan2",
    },
    "triton.language.extra.cann.libdevice.atanh": {
        "example": "triton.language.extra.cann.libdevice.atanh",
    },
    "triton.language.extra.cann.libdevice.copysign": {
        "example": "triton.language.extra.cann.libdevice.copysign",
    },
    "triton.language.extra.cann.libdevice.cosh": {
        "example": "triton.language.extra.cann.libdevice.cosh",
    },
    "triton.language.extra.cann.libdevice.cyl_bessel_i0": {
        "example": "triton.language.extra.cann.libdevice.cyl_bessel_i0",
    },
    "triton.language.extra.cann.libdevice.div_rz": {
        "example": "triton.language.extra.cann.libdevice.div_rz",
    },
    "triton.language.extra.cann.libdevice.erfinv": {
        "example": "triton.language.extra.cann.libdevice.erfinv",
    },
    "triton.language.extra.cann.libdevice.expm1": {
        "example": "triton.language.extra.cann.libdevice.expm1",
    },
    "triton.language.extra.cann.libdevice.fast_dividef": {
        "example": "triton.language.extra.cann.libdevice.fast_dividef",
    },
    "triton.language.extra.cann.libdevice.fast_expf": {
        "example": "triton.language.extra.cann.libdevice.fast_expf",
    },
    "triton.language.extra.cann.libdevice.float_as_int": {
        "example": "triton.language.extra.cann.libdevice.float_as_int",
    },
    "triton.language.extra.cann.libdevice.fmod": {
        "example": "triton.language.extra.cann.libdevice.fmod",
    },
    "triton.language.extra.cann.libdevice.gamma": {
        "example": "triton.language.extra.cann.libdevice.gamma",
    },
    "triton.language.extra.cann.libdevice.hypot": {
        "example": "triton.language.extra.cann.libdevice.hypot",
    },
    "triton.language.extra.cann.libdevice.ilogb": {
        "example": "triton.language.extra.cann.libdevice.ilogb",
    },
    "triton.language.extra.cann.libdevice.isinf": {
        "example": "triton.language.extra.cann.libdevice.isinf",
    },
    "triton.language.extra.cann.libdevice.isnan": {
        "example": "triton.language.extra.cann.libdevice.isnan",
    },
    "triton.language.extra.cann.libdevice.ldexp": {
        "example": "triton.language.extra.cann.libdevice.ldexp",
    },
    "triton.language.extra.cann.libdevice.lgamma": {
        "example": "triton.language.extra.cann.libdevice.lgamma",
    },
    "triton.language.extra.cann.libdevice.log10": {
        "example": "triton.language.extra.cann.libdevice.log10",
    },
    "triton.language.extra.cann.libdevice.log1p": {
        "example": "triton.language.extra.cann.libdevice.log1p",
    },
    "triton.language.extra.cann.libdevice.nearbyint": {
        "example": "triton.language.extra.cann.libdevice.nearbyint",
    },
    "triton.language.extra.cann.libdevice.nextafter": {
        "example": "triton.language.extra.cann.libdevice.nextafter",
    },
    "triton.language.extra.cann.libdevice.pow": {
        "example": "triton.language.extra.cann.libdevice.pow",
    },
    "triton.language.extra.cann.libdevice.reciprocal": {
        "example": "triton.language.extra.cann.libdevice.reciprocal",
    },
    "triton.language.extra.cann.libdevice.relu": {
        "example": "triton.language.extra.cann.libdevice.relu",
    },
    "triton.language.extra.cann.libdevice.rint": {
        "example": "triton.language.extra.cann.libdevice.rint",
    },
    "triton.language.extra.cann.libdevice.round": {
        "example": "triton.language.extra.cann.libdevice.round",
    },
    "triton.language.extra.cann.libdevice.signbit": {
        "example": "triton.language.extra.cann.libdevice.signbit",
    },
    "triton.language.extra.cann.libdevice.sinh": {
        "example": "triton.language.extra.cann.libdevice.sinh",
    },
    "triton.language.extra.cann.libdevice.tan": {
        "example": "triton.language.extra.cann.libdevice.tan",
    },
    "triton.language.extra.cann.libdevice.tanh": {
        "example": "triton.language.extra.cann.libdevice.tanh",
    },
    "triton.language.extra.cann.libdevice.trunc": {
        "example": "triton.language.extra.cann.libdevice.trunc",
    },
    # ------------------------------------------------------------------
    # TLA interfaces
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # Core types
    # ------------------------------------------------------------------
    f"{_TLA_PREFIX}.buffer": {
        "constraints": [
            "``buf[i0, i1, ...]`` 返回 ``buffer_access``，用作 ``simd.vlds`` / ``simd.vsts`` 的地址参数。",
            "GM buffer 必须通过 ``tla.view`` 从内核指针获得，不能用 ``alloc`` 分配。",
        ],
        "example":
        "buffer",
    },
    f"{_TLA_PREFIX}.buffer_type": {
        "constraints": [
            "``shape`` / ``strides`` 必须是正的编译期整数，且 ``strides`` 的秩与 ``shape`` 一致。",
            "``space`` 只能是 ``gm`` / ``ub`` / ``l1`` / ``l0a`` / ``l0b`` / ``l0c``。",
        ],
        "example":
        "buffer",
    },
    f"{_TLA_PREFIX}.vector": {
        "constraints": [
            "不支持隐式转换为 Triton tensor，也不能作为标量条件使用（``__bool__`` 会报错）。",
        ],
        "example": "buffer",
    },
    f"{_TLA_PREFIX}.vector_type": {
        "constraints": [
            "``lanes`` 必须是正的编译期整数。",
            "元素类型必须是标量整型或浮点型 dtype。",
        ],
        "example": "buffer",
    },

    # ------------------------------------------------------------------
    # Allocation / view / copy
    # ------------------------------------------------------------------
    f"{_TLA_PREFIX}.alloc": {
        "constraints": [
            "``space`` 默认 ``\"ub\"``，不能为 ``\"gm\"``；GM buffer 必须由内核指针经 ``tla.view`` 获得。",
            "``versions`` 必须是正整数；``1`` 表示关闭多缓冲。",
        ],
        "example":
        "buffer",
    },
    f"{_TLA_PREFIX}.alloc_buffer": {
        "constraints": [
            "``space`` 默认 ``\"ub\"``，不能为 ``\"gm\"``；GM buffer 必须由内核指针经 ``tla.view`` 获得。",
            "``versions`` 必须是正整数；``1`` 表示关闭多缓冲。",
        ],
        "example":
        "buffer",
    },
    f"{_TLA_PREFIX}.annotate_buffer_versions": {
        "constraints": [
            "``versions`` 必须是正整数，``1`` 表示关闭多缓冲。",
            "版本数通过 ``hivm.multi_buffer`` 注解交给 NPU-IR 多缓冲 pass。",
        ],
        "example": "buffer",
    },
    f"{_TLA_PREFIX}.copy": {
        "constraints": [
            "``src`` 和 ``dst`` 都必须是 ``tla.buffer``，不接受指针或 tensor。",
            "源和目标元素类型必须一致。",
            "支持的地址空间组合：``gm->ub``、``ub->gm``、``gm->l1``、``l0c->*``。",
            "除 ``gm->l1`` 的 ND->NZ 和 ``l0c`` Fixpipe 的 fractal 关系外，源和目标形状必须一致。",
            "其他地址空间组合会报错。",
        ],
        "example":
        "copy",
    },
    f"{_TLA_PREFIX}.view": {
        "constraints": [
            "指针入口：``src`` 必须是指针 tensor，``strides`` 为最终 memref 步长，默认行主序。",
            "缓冲区切片：``src`` 必须是 ``tla.buffer``，``strides`` 相对于源视图，``space`` 必须保持默认值。",
            "``strides`` / ``offsets`` 的秩必须与结果形状一致。",
            "静态 offset 必须位于源 buffer 范围内。",
        ],
        "example":
        "view",
    },
    f"{_TLA_PREFIX}.reshape": {
        "constraints": [
            "只支持连续、行主序、无偏移的 buffer；带步长或偏移的切片请先使用 ``tla.view``。",
            "元素总数必须保持不变。",
        ],
        "example": "reshape",
    },

    # ------------------------------------------------------------------
    # Core-mode scopes
    # ------------------------------------------------------------------
    f"{_TLA_PREFIX}.scope": {
        "constraints": [
            "作用域在编译期确定核心类型，不需要运行时参数。",
            "使用 ``with tla.scope.vector():`` / ``with tla.scope.simd():`` / ``with tla.scope.cube():``。",
        ],
        "example":
        "scope",
    },
    f"{_TLA_PREFIX}.scope.vector": {
        "constraints": [
            "承载 ``tla.copy``、控制流等非 SIMD 操作。",
            "可以嵌套 ``tla.scope.simd()``。",
        ],
        "example": "scope",
    },
    f"{_TLA_PREFIX}.scope.simd": {
        "constraints": [
            "SIMD 向量函数区域，编译器会 outline 为 ``hivm.vector_function``。",
            "``tla.simd.*`` 只能操作 ``tla.vector``。",
        ],
        "example": "scope",
    },
    f"{_TLA_PREFIX}.scope.cube": {
        "constraints": [
            "Cube 核区域当前保留，Cube 算子尚未通过该作用域暴露。",
        ],
        "example": "scope",
    },

    # ------------------------------------------------------------------
    # SIMD load / store
    # ------------------------------------------------------------------
    f"{_TLA_PREFIX}.simd.vlds": {
        "constraints": [
            "``ptr`` 必须是 UB buffer 或 ``buf[i0, i1, ...]`` 形式的元素范围。",
            "``dist`` 必须是 AVE load 分布字符串，默认 ``NORM``。",
            "结果是一个满向量寄存器：2048 位；``i64`` / ``u64`` 使用 4096 位。",
            "支持的元素类型：``i8`` / ``i16`` / ``i32`` / ``i64`` / ``fp8e4nv`` / ``f16`` / ``bf16`` / ``f32``。",
        ],
        "example":
        "simd_load_store",
    },
    f"{_TLA_PREFIX}.simd.vsts": {
        "constraints": [
            "``ptr`` 必须是 UB buffer 或 ``buf[i0, i1, ...]`` 形式的元素范围。",
            "``dist`` 默认 ``NORM_B32``，必须是 AVE store 分布字符串。",
            "``mask`` 默认全真 ``pge <ALL>``；显式传入时 lane 数必须与值向量一致。",
        ],
        "example":
        "simd_load_store",
    },

    # ------------------------------------------------------------------
    # SIMD conversion / layout
    # ------------------------------------------------------------------
    f"{_TLA_PREFIX}.simd.vextf": {
        "constraints": [
            "当前只支持拓宽到 ``f32``，结果为 64 lane 满寄存器。",
            "``part`` 只支持 ``part_even`` / ``part_odd``，不改变结果位宽。",
        ],
        "example": "simd_convert",
    },
    f"{_TLA_PREFIX}.simd.vtruncf": {
        "constraints": [
            "目标类型只支持 ``bf16`` / ``f16`` / ``fp8e4nv``。",
            "总位宽保持不变，元素变窄后 lane 数相应增加。",
            "``part`` 只支持 ``part_even`` / ``part_odd``。",
        ],
        "example":
        "simd_convert",
    },
    f"{_TLA_PREFIX}.simd.vbitcast": {
        "constraints": [
            "源总位宽必须能被目标元素位宽整除。",
            "结果 lane 数为 ``src_lanes * src_bits / dst_bits``。",
        ],
        "example": "simd_convert",
    },
    f"{_TLA_PREFIX}.simd.vintlv": {
        "constraints": [
            "两个源寄存器类型必须完全一致。",
            "返回两个结果寄存器。",
        ],
        "example": "simd_convert",
    },
    f"{_TLA_PREFIX}.simd.vdintlv": {
        "constraints": [
            "两个源寄存器类型必须完全一致。",
            "返回两个结果寄存器，是 ``vintlv`` 的逆操作。",
        ],
        "example": "simd_convert",
    },

    # ------------------------------------------------------------------
    # SIMD arithmetic / unary
    # ------------------------------------------------------------------
    **{
        f"{_TLA_PREFIX}.simd.{name}": {
            "constraints": [
                "两个操作数类型必须完全一致，不支持隐式类型提升。",
                "使用全真 ``pge <ALL>`` 掩码。",
            ],
            "example": "simd_arith",
        }
        for name in ("vadd", "vmul", "vdiv", "vxor")
    },
    f"{_TLA_PREFIX}.simd.vmod": {
        "constraints": [
            "仅用于整数向量，两个操作数类型必须完全一致。",
            "使用全真 ``pge <ALL>`` 掩码。",
        ],
        "example": "simd_arith",
    },
    f"{_TLA_PREFIX}.simd.vfma": {
        "constraints": [
            "``acc`` / ``lhs`` / ``rhs`` 类型必须完全一致，计算 ``acc + lhs * rhs``。",
            "使用全真 ``pge <ALL>`` 掩码。",
        ],
        "example": "simd_arith",
    },
    **{
        f"{_TLA_PREFIX}.simd.{name}": {
            "constraints": [
                "保持元素类型不变。",
                "使用全真 ``pge <ALL>`` 掩码。",
            ],
            "example": "simd_arith",
        }
        for name in ("vneg", "vsqrt", "vrsqrt", "vexp")
    },

    # ------------------------------------------------------------------
    # SIMD compare / select
    # ------------------------------------------------------------------
    f"{_TLA_PREFIX}.simd.vcmp": {
        "constraints": [
            "``cmp_type`` 支持 ``eq`` / ``ne`` / ``gt`` / ``ge`` / ``lt`` / ``le``。",
            "两个操作数类型必须完全一致。",
            "返回 ``int1`` 谓词向量，lane 数与输入一致。",
        ],
        "example":
        "simd_compare",
    },
    f"{_TLA_PREFIX}.simd.vsel": {
        "constraints": [
            "``mask`` 的 lane 数必须与值向量一致。",
            "``true_val`` 和 ``false_val`` 类型必须完全一致。",
        ],
        "example": "simd_compare",
    },

    # ------------------------------------------------------------------
    # SIMD reduction / broadcast / constants / predicate
    # ------------------------------------------------------------------
    f"{_TLA_PREFIX}.simd.vreduce": {
        "constraints": [
            "``kind`` 为 ``add`` / ``min`` / ``max``。",
            "结果放在 lane 0（AVE 规约语义）。",
        ],
        "example": "simd_reduce",
    },
    f"{_TLA_PREFIX}.simd.vreduce_sum": {
        "constraints": [
            "等价于 ``vreduce(src, \"add\")``，结果放在 lane 0。",
        ],
        "example": "simd_reduce",
    },
    f"{_TLA_PREFIX}.simd.vreduce_max": {
        "constraints": [
            "接受单个寄存器或寄存器列表，所有寄存器类型必须一致。",
            "先做逐元素 ``max``，再水平规约并广播到所有 lane。",
        ],
        "example": "simd_reduce",
    },
    f"{_TLA_PREFIX}.simd.vreduce_min": {
        "constraints": [
            "接受单个寄存器或寄存器列表，所有寄存器类型必须一致。",
            "先做逐元素 ``min``，再水平规约并广播到所有 lane。",
        ],
        "example": "simd_reduce",
    },
    f"{_TLA_PREFIX}.simd.vbroadcast": {
        "constraints": [
            "``low=True`` 广播低半部分，``low=False`` 广播高半部分。",
            "结果 lane 数与源寄存器一致。",
        ],
        "example": "simd_reduce",
    },
    f"{_TLA_PREFIX}.simd.vfull": {
        "constraints": [
            "``lanes`` 必须是正的编译期整数。",
            "``lanes`` 与元素位宽的乘积不能超过寄存器位宽。",
        ],
        "example": "simd_const",
    },
    f"{_TLA_PREFIX}.simd.vsplat": {
        "constraints": [
            "lane 数由 ``dtype`` 推导：``2048 // elem_bits``，``i64`` 使用 4096 位。",
        ],
        "example": "simd_const",
    },
    f"{_TLA_PREFIX}.simd.vci": {
        "constraints": [
            "固定为 ``int32`` / 64 lane。",
            "``base`` 可以是编译期 int32，或标量 int32 tensor。",
        ],
        "example": "simd_const",
    },
    f"{_TLA_PREFIX}.simd.pge": {
        "constraints": [
            "``n`` 只支持 1 / 2 / 3 / 4 / 8 / 16 / 32 / 64 / 128。",
            "``lanes`` 是 ``int1`` 谓词向量的 lane 数。",
        ],
        "example": "simd_const",
    },
    f"{_TLA_PREFIX}.simd.membar": {
        "constraints": [
            "``scope`` 支持 AVE membar 类型，例如 ``VV_ALL``、``VST_VLD``、``VLD_VST``。",
            "完整取值：VV_ALL、VST_VLD、VLD_VST、VST_VST、VS_ALL、VST_LD、VLD_ST、VST_ST、"
            "SV_ALL、ST_VLD、LD_VST、ST_VST、SS_ALL、ST_LD、LD_ST、ST_ST。",
        ],
        "example":
        "simd_membar",
    },

    # ------------------------------------------------------------------
    # Cube
    # ------------------------------------------------------------------
    f"{_TLA_PREFIX}.cube.mmad": {
        "constraints": [
            "``a`` / ``b`` 必须是 L1 buffer，``acc`` 必须是 L0C buffer。",
            "``a`` 和 ``b`` 元素类型必须一致。",
            "形状必须满足 ``a=(M, K)``、``b=(K, N)``、``acc=(M, N)``。",
            "操作数可以是 2 维 ND 形状，也可以是等价的 4 维 zN 形状。",
            "只暴露 ``transpose_b``；``transpose_a`` 可通过 GM->L1 拷贝调整布局实现。",
        ],
        "example":
        "cube_mmad",
    },

    # ------------------------------------------------------------------
    # Synchronization
    # ------------------------------------------------------------------
    f"{_TLA_PREFIX}.sync.barrier": {
        "constraints": [
            "``pipe`` 是编译期字符串，例如 ``\"V\"``、``\"MTE2\"`` 或 ``\"PIPE_V\"``。",
            "``barrier(\"ALL\")`` 等价于 ``PIPE_ALL``。",
        ],
        "example":
        "sync",
    },
    f"{_TLA_PREFIX}.sync.set_flag": {
        "constraints": [
            "``event_id`` 范围 ``[0, 7]``。",
            "``pipe1`` / ``pipe2`` 是编译期流水字符串。",
        ],
        "example": "sync",
    },
    f"{_TLA_PREFIX}.sync.wait_flag": {
        "constraints": [
            "``event_id`` 范围 ``[0, 7]``，必须与对应的 ``set_flag`` 一致。",
            "``pipe1`` / ``pipe2`` 是编译期流水字符串。",
        ],
        "example": "sync",
    },
    f"{_TLA_PREFIX}.sync.sync_all": {
        "constraints": [
            "``mode`` 必须是编译期 ``bool``。",
            "``mode=False`` 同步 AIV 核，``mode=True`` 同步 AIC 和 AIV 全部核。",
        ],
        "example": "sync",
    },
    f"{_TLA_PREFIX}.sync.block_set_flag": {
        "constraints": [
            "``mode`` 取值：``0`` inter-block、``1`` inter-subblock、``2`` intra-block。",
            "``flag`` 范围 ``[0, 15]``。",
        ],
        "example":
        "sync",
    },
    f"{_TLA_PREFIX}.sync.block_wait_flag": {
        "constraints": [
            "``flag`` 范围 ``[0, 15]``，必须与对应的 ``block_set_flag`` 一致。",
        ],
        "example": "sync",
    },

    # ------------------------------------------------------------------
    # Math / debug
    # ------------------------------------------------------------------
    f"{_TLA_PREFIX}.compensated_rstd": {
        "constraints": [
            "``square_sum`` 必须是 ``f32`` 向量寄存器，通常为广播到所有 lane 的值。",
            "``hidden`` 必须是正的 Python / constexpr 标量。",
            "``eps`` 必须不小于最小 FP32 正规数 ``2^-126``。",
            "语义与 TileKernels ``rstd.py`` 的 ``make_rstd_impl`` 一致，不是普通 RMSNorm 公式。",
        ],
        "example":
        "compensated_rstd",
    },
    f"{_TLA_PREFIX}.device_print": {
        "constraints": [
            "只允许在 ``tla.scope`` 外使用。",
            "只支持 UB buffer、UB 元素范围和标量。",
            "``prefix`` 必须是字符串。",
        ],
        "example": "device",
    },
    f"{_TLA_PREFIX}.device_assert": {
        "constraints": [
            "``cond`` 必须是标量 ``int1``，向量条件会被拒绝。",
            "仅在开启 ``debug`` 选项时生效。",
            "``msg`` 必须是字符串。",
        ],
        "example": "device",
    },
}
