# _ascend_constraints.py
# Ascend platform constraints for Triton API
# Fields: constraints (text list), example (optional file name)

CONSTRAINTS = {
    "triton.Config": {
        "constraints": [
            "``num_stages``: Not applicable on Ascend",
            "``num_ctas``: Not applicable on Ascend",
            "``maxnreg``: No effect on Ascend",
        ],
        "example":
        "triton.Config",
    },
    "triton.autotune": {
        "constraints": [
            "``use_cuda_graph``: Not applicable on Ascend",
        ],
        "example": "triton.autotune",
    },
    "triton.extension.buffer.language.alloc": {
        "constraints": [
            "DataType: Ascend A2/A3/950 supports int8, int16, int32, uint8, uint64, int64, fp32, bf16, bool. Does not support uint16, uint32, fp16.",
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
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
        ],
        "example": "triton.language.permute",
    },
    "triton.heuristics": {
        "example": "triton.heuristics",
    },
    "triton.jit": {
        "example":
        "triton.jit",
        "replace_docstring": [
            "Decorator for JIT-compiling a function using the Triton compiler.",
            "",
            ":note: When a jit'd function is called, arguments are",
            "    implicitly converted to pointers if they have a :code:`.data_ptr()` method",
            "    and a `.dtype` attribute.",
            "",
            ":note: This function will be compiled and run on the device. It will only have access to:",
            "",
            "       * python primitives,",
            "       * builtins within the triton package,",
            "       * arguments to this function,",
            "       * other jit'd functions",
            "",
            ":param fn: the function to be jit-compiled",
            ":type fn: Callable",
        ],
    },
    "triton.language.abs": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
        ],
        "example":
        "triton.language.abs",
    },
    "triton.language.add": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp8, fp64.",
        ],
        "example": "triton.language.add",
    },
    "triton.language.advance": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "Compilation issues may arise when used with complex loops and branch statements."
        ],
        "example":
        "triton.language.advance",
    },
    "triton.language.arange": {
        "constraints": [
            "Arguments must be tl.constexpr of uint or int type (int64 not supported).",
            "start and end must be greater than or equal to 0.",
            "end - start < 1048576 (TRITON_MAX_TENSOR_NUMEL).",
            "``relaxed_requirement`` (Ascend extension): Ascend does not require range=(end-start) to be a power of 2.",
        ],
        "example":
        "triton.language.arange",
    },
    "triton.language.argmax": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64; Ascend 950 does not fp64, fp8e4(E4M3), fp8e5(E5M2).",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.argmax",
    },
    "triton.language.argmin": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64; Ascend 950 does not fp64, fp8e4(E4M3), fp8e5(E5M2).",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.argmin",
    },
    "triton.language.bitonic_merge": {
        "constraints": [],
        "example": "triton.language.bitonic_merge",
    },
    "triton.language.associative_scan": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64.",
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
            "DataType: Ascend A2/A3/950 does not support fp64.",
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
            "DataType: Ascend A2/A3/950 does not support fp64.",
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
            "DataType: Ascend A2/A3/950 does not support fp64.",
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
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
        ],
        "example": "triton.language.broadcast",
    },
    "triton.language.broadcast_to": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
            "The rank of the input tensor's shape must match the rank of the target shape.",
        ],
        "example":
        "triton.language.broadcast_to",
    },
    "triton.language.cast": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
        ],
        "example": "triton.language.cast",
    },
    "triton.language.cat": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64.",
        ],
        "example": "triton.language.cat",
    },
    "triton.language.cdiv": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, bool, \
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
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64, uint8.",
            "hint_name parameter must be of the string type.",
            "the hint_val of the list type supports only integer arrays and does not support floating-point numbers or mixed types",
        ],
        "example":
        "triton.language.compile_hint",
    },
    "triton.language.core.__rshift__": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, uint8.",
            "``other``: only scalars are supported;tensor ars not supported(x >> 2 is valid,x >> y[y is tensor] is not currently supported.)",
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
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64.",
            "reverse=True requires alignment when loading data with tl.load, and mask cannot be used to filter redundant data indices",
        ],
        "example":
        "triton.language.cumprod",
    },
    "triton.language.cumsum": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64.",
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
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "``prefix``: the first argument must be a string prefix; omitting it causes a compilation error.",
            "Set environment variable ``TRITON_DEVICE_PRINT=1`` to enable.",
        ],
        "example":
        "triton.language.device_print",
        "replace_docstring": [
            "Print the values at runtime from the device.  String formatting does not work for runtime values, so you should",
            "provide the values you want to print as arguments.  The first value must be a string, all following values must",
            "be scalars or tensors.",
            "",
            "Calling the Python builtin :code:`print` is the same as calling this function, and the requirements for the arguments will match",
            "this function (not the normal requirements for :code:`print`).",
            "",
            ".. highlight:: python",
            ".. code-block:: python",
            "",
            "    tl.device_print(\"pid\", pid)",
            "    print(\"pid\", pid)",
            "",
            "On Ascend, device_print is available when the environment variable ``TRITON_DEVICE_PRINT=1``",
            "is set. The output is streamed through a device buffer of limited size.",
            "",
            ":param prefix: a prefix to print before the values. This is required to be a string literal.",
            ":param args: the values to print. They can be any tensor or scalar.",
            ":param hex: print all values as hex instead of decimal",
        ],
    },
    "triton.language.div": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support uint8, uint16, uint32, uint64, fp8, fp64.",
        ],
        "example": "triton.language.div",
    },
    "triton.language.div_rn": {
        "constraints": [],
        "example": "triton.language.div_rn",
    },
    "triton.language.dot": {
        "replace_docstring": [
            "Returns the matrix product of two blocks.",
            "",
            "The two blocks must both be two-dimensional or three-dimensional and have compatible inner dimensions.",
            "For three-dimensional blocks, `tl.dot` performs the batched matrix product,",
            "where the first dimension of each block represents the batch dimension.",
            "",
            ":param input: The first tensor to be multiplied.",
            ":type input: 2D or 3D tensor of scalar-type in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}",
            ":param other: The second tensor to be multiplied.",
            ":type other: 2D or 3D tensor of scalar-type in {:code:`int8`, :code:`float8_e5m2`, :code:`float16`, :code:`bfloat16`, :code:`float32`}",
            ":param acc: The accumulator tensor. If not None, the result is added to this tensor.",
            ":type acc: 2D or 3D tensor of scalar-type in {:code:`float16`, :code:`float32`, :code:`int32`}",
            ":param input_precision: How to exercise the Tensor Cores for f32 x f32. If",
            "  the device does not have Tensor Cores or the inputs are not of dtype f32,",
            "  this option is ignored. For devices that do have tensor cores, the",
            "  default precision is tf32.",
            ":type input_precision: string. Available options: :code:`\"tf32\"`, :code:`\"tf32x3\"`, :code:`\"ieee\"`. Default: :code:`\"tf32\"`.",
            ":param allow_tf32: *Deprecated.* If true, input_precision is set to \"tf32\".",
            "  Only one of :code:`input_precision` and :code:`allow_tf32` can be",
            "  specified (i.e. at least one must be :code:`None`).",
        ],
        "example":
        "triton.language.dot",
    },
    "triton.language.dot_scaled": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp4, fp8.",
            "``lhs_k_pack``: does not support fp4,fp8.",
            "``rhs_k_pack``: does not support fp4,fp8.",
            "Ascend does not support fp4, fp8.",
            "the recommended input range for the input matrices lhs and rhs is [-5, 5]",
        ],
        "example":
        "triton.language.dot_scaled",
    },
    "triton.language.equal": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
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
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
        ],
        "example": "triton.language.expand_dims",
    },
    "triton.language.fdiv": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.fdiv",
    },
    "triton.language.extra.cann.extension.index_select_simd": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
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
            "DataType: Ascend A2/A3/950 does not support bool.",
        ],
        "example": "triton.language.extra.cann.extension.extract_slice",
    },
    "triton.language.extra.cann.extension.get_element": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support bool.",
        ],
        "example": "triton.language.extra.cann.extension.get_element",
    },
    "triton.language.extra.cann.extension.insert_slice": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support bool.",
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
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
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
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64, uint8, Ascend 950 does not support fp64.",
        ],
        "example":
        "triton.language.flip",
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
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64.",
        ],
        "example": "triton.language.floordiv",
    },
    "triton.language.full": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64.",
        ],
        "example": "triton.language.full",
    },
    "triton.language.gather": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.gather",
    },
    "triton.language.greater_than": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64, Ascend 950 does not support fp64.",
        ],
    },
    "triton.language.greater_equal": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
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
        "replace_docstring": [
            "Execute inline assembly over a tensor.  Essentially, this is :code:`map`",
            "where the function is inline assembly.",
            "",
            "The input tensors :code:`args` are implicitly broadcasted to the same shape.",
            "",
            ":code:`dtype` can be a tuple of types, in which case the output is a",
            "tuple of tensors.",
            "",
            "Each invocation of the inline asm processes :code:`pack` elements at a",
            "time.  Exactly which set of inputs a block receives is unspecified.",
            "Input elements of size less than 4 bytes are packed into 4-byte",
            "registers.",
            "",
            "This op does not support empty :code:`dtype` -- the inline asm must",
            "return at least one tensor, even if you don't need it.  You can work",
            "around this by returning a dummy tensor of arbitrary type; it shouldn't",
            "cost you anything if you don't use it.",
            "",
            "On Ascend, inline assembly supports int64 (s64) and float32 (f32) register types",
            "with the 'l' LLVM constraint.",
            "",
            ":param asm: the inline assembly code",
            ":param constraints: the LLVM constraint string (only 'l' is supported on Ascend)",
            ":param args: the input tensors",
            ":param dtype: the output data type(s)",
            ":param is_pure: whether the assembly is pure (no side effects)",
            ":param pack: the number of elements to process at a time",
        ],
    },
    "triton.language.interleave": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
        ],
        "example": "triton.language.interleave",
    },
    "triton.language.join": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
        ],
        "example": "triton.language.join",
    },
    "triton.language.less_equal": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
        ],
    },
    "triton.language.less_than": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
        ],
    },
    "triton.language.load": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "``cache_modifier``: Ascend A5 SIMT-only accepts the Triton upstream strings (``''``, ``'.ca'``, ``'.cg'``, ``'.cv'``); the backend folds them to a two-level cache / uncache mapping (``'.ca'`` -> cache, ``'.cg'`` / ``'.cv'`` -> uncache). Effective on global memory; ignored on shared memory. A2/A3 and non-SIMT-only paths ignore the modifier.",
            "``eviction_policy``: has no effect on Ascend.",
            "``volatile``: ``bool``; on Ascend A5 SIMT-only, guarantees memory ordering and prevents the compiler/hardware from hoisting or eliding the load. Effective on both global and shared memory. A2/A3 and non-SIMT-only paths ignore the modifier.",
            "Compatibility issues with branch and loop statements: \
                Complex pointer and mask calculations involving branches or loops may cause compilation failures.",
        ],
        "example":
        "triton.language.load",
        "replace_docstring": [
            "Return a tensor of data whose values are loaded from memory at location defined by `pointer`:",
            "",
            "    (1) If `pointer` is a single element pointer, a scalar is be loaded.  In",
            "        this case:",
            "",
            "        - `mask` and `other` must also be scalars,",
            "        - `other` is implicitly typecast to `pointer.dtype.element_ty`, and",
            "        - `boundary_check` and `padding_option` must be empty.",
            "",
            "    (2) If `pointer` is an N-dimensional tensor of pointers, an",
            "        N-dimensional tensor is loaded.  In this case:",
            "",
            "        - `mask` and `other` are implicitly broadcast to `pointer.shape`,",
            "        - `other` is implicitly typecast to `pointer.dtype.element_ty`, and",
            "        - `boundary_check` and `padding_option` must be empty.",
            "",
            "    (3) If `pointer` is a block pointer defined by `make_block_ptr`, a",
            "        tensor is loaded.  In this case:",
            "",
            "        - `mask` and `other` must be `None`, and",
            "        - `boundary_check` and `padding_option` can be specified to control the behavior of out-of-bound access.",
            "",
            ":param pointer: Pointer to the data to be loaded",
            ":type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`",
            ":param mask: if `mask[idx]` is false, do not load the data at address `pointer[idx]`",
            "    (must be `None` with block pointers)",
            ":type mask: Block of `triton.int1`, optional",
            ":param other: if `mask[idx]` is false, return `other[idx]`",
            ":type other: Block, optional",
            ":param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check",
            ":type boundary_check: tuple of ints, optional",
            ":param padding_option: should be one of {\"\", \"zero\", \"nan\"}, the padding value to use while out of bounds. \"\" means an undefined value.",
            ":param cache_modifier: changes cache option for the load operation.",
            "  On Ascend A5 SIMT-only, accepts the upstream strings (``''``, ``'.ca'``, ``'.cg'``, ``'.cv'``);",
            "  the backend folds them to a two-level cache / uncache mapping.",
            "  Effective on global memory; ignored on shared memory.",
            "  A2/A3 and non-SIMT-only paths ignore the modifier.",
            ":type cache_modifier: str, optional, should be one of {\"\", \".ca\", \".cg\", \".cv\"}",
            ":param eviction_policy: has no effect on Ascend.",
            ":type eviction_policy: str, optional",
            ":param volatile: on Ascend A5 SIMT-only, guarantees memory ordering and prevents",
            "  the compiler/hardware from hoisting or eliding the load.",
            "  Effective on both global and shared memory.",
            "  A2/A3 and non-SIMT-only paths ignore the modifier.",
            ":type volatile: bool, optional",
        ],
    },
    "triton.language.load_tensor_descriptor": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
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
    "triton.language.map_elementwise": {
        "constraints": [
            "`while` loops are not supported inside the scalar function.",
            "`pack` has no semantic effect on NPU backends as the implementation is always vectorized.",
        ],
        "example":
        "triton.language.map_elementwise",
        "replace_docstring": [
            "Map a scalar function over a tensor.",
            "",
            "The input tensors :code:`args` are implicitly broadcasted to the same shape.",
            "",
            "This may be useful in allowing control flow over single elements in a tensor,",
            "for example a multi-branch function with :code:`if`/:code:`elif`/:code:`else`",
            "chains or :code:`for` loops, providing more flexible element-wise computation",
            "than :code:`tl.where` which only supports binary selection.",
            "",
            ".. highlight:: python",
            ".. code-block:: python",
            "",
            "    @triton.jit",
            "    def compare_scalar(x, y):",
            "        if x < y:",
            "            return -1",
            "        elif x == y:",
            "            return 0",
            "        else:",
            "            return 1",
            "",
            "    @triton.jit",
            "    def compare(x, y):",
            "        return tl.map_elementwise(compare_scalar, x, y)",
            "",
            ":param scalar_fn: the function to map over.",
            ":param pack: the number of elements to be processed by one function call.",
            "  On Ascend, this parameter has no semantic effect as the implementation",
            "  is always vectorized.",
            ":return: one tensor or a tuple of tensors, depending on the mapped function.",
        ],
    },
    "triton.language.make_block_ptr": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "Transpose logic can only be implemented by adjusting the `order` parameter. \
                Do not reorder strides to achieve transpose.",
            "Compatibility issues may occur when used together with branch and loop statements."
        ],
        "example":
        "triton.language.make_block_ptr",
    },
    "triton.language.make_tensor_descriptor": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "`make_tensor_descriptor`, `load_tensor_descriptor` and `store_tensor_descriptor` \
                must be used as a suite in Triton 3.6.x. Do not mix them with `tl.load()` or `tl.store()`.",
            "Compatibility issues exist for certain functions (e.g. cast) in Triton 3.6.x."
        ],
        "example":
        "triton.language.make_tensor_descriptor",
        "replace_docstring": [
            "Make a tensor descriptor object for loading/storing blocks of data from global memory.",
            "",
            ":param base: the base pointer of the tensor, must be 16-byte aligned",
            ":param shape: A list of non-negative integers representing the tensor shape",
            ":param strides: A list of tensor strides. Leading dimensions must be multiples",
            "    of 16-byte strides and the last dimension must be contiguous.",
            ":param block_shape: The shape of block to be loaded/stored from global memory",
            "",
            ".. rubric:: Notes",
            "",
            "On Ascend, `make_tensor_descriptor`, `load_tensor_descriptor` and `store_tensor_descriptor`",
            "must be used together as a suite. Do not mix them with `tl.load()` or `tl.store()`.",
            "",
            "Currently only 2-5 dimensional tensors are supported.",
        ],
    },
    "triton.language.max": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64; Ascend 950 does not fp64.",
            "``return_indices``: return_indices=True is not supported when axis=None.",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.max",
    },
    "triton.language.max_constancy": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "``values``: rank must match the rank of ``input`` (e.g. ``[1, 1]`` for a 2-D input).",
        ],
        "example":
        "triton.language.max_constancy",
    },
    "triton.language.max_contiguous": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "``values``: rank must match the rank of ``input`` (e.g. ``[1, 1]`` for a 2-D input).",
        ],
        "example":
        "triton.language.max_contiguous",
    },
    "triton.language.maximum": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.maximum",
    },
    "triton.language.min": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64; Ascend 950 does not fp64, fp8e4(E4M3), fp8e5(E5M2).",
            "``return_indices``: return_indices=True is not supported when axis=None.",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.min",
    },
    "triton.language.minimum": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.minimum",
    },
    "triton.language.mod": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp8, fp64, \
                Ascend 950 does not support fp8/fp64.",
        ],
        "example":
        "triton.language.mod",
    },
    "triton.language.multiple_of": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "``values``: describes the divisibility of the first value along each dimension, so its rank must match ``input`` (e.g. ``[1, 1]`` for a 2-D input).",
        ],
        "example":
        "triton.language.multiple_of",
    },
    "triton.language.neg": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support bool, fp64, uint16, uint32, uint64.",
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
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "``disallow_acc_multi_buffer``, ``flatten``, ``disable_licm``: related functionality is incomplete on Ascend.",
        ],
        "example":
        "triton.language.range",
        "replace_docstring": [
            "Iterator that counts upward forever.",
            "",
            ".. rubric:: Notes",
            "",
            "This is a special iterator used to implement similar semantics to",
            "Python's :code:`range` in the context of :code:`triton.jit` functions.",
            "In addition, it allows user to pass extra attributes to the compiler.",
            "",
            ":param arg1: the start value.",
            ":param arg2: the end value.",
            ":param step: the step value.",
            ":param num_stages: pipeline the loop into this many stages (so there are",
            "    :code:`num_stages` iterations of the loop in flight at once).",
            "",
            "    Note this is subtly different than passing :code:`num_stages` as a",
            "    kernel argument.  The kernel argument only pipelines loads that feed",
            "    into :code:`dot` operations, while this attribute tries to pipeline most",
            "    (though not all) loads in this loop.",
            ":param loop_unroll_factor: Tells the Triton IR level loop unroller how many",
            "    times to unroll a for loop that this range is used with. Less than 2 for",
            "    this value implies no unrolling.",
            ":param disallow_acc_multi_buffer: If true, prevent the accumulator of the dot",
            "    operation in the loop to be multi-buffered, if applicable.",
            ":param flatten: automatically flatten the loop nest starting at this loop to",
            "    create a single flattened loop. The compiler will try to pipeline the",
            "    flattened loop which can avoid stage stalling.",
            ":param warp_specialize: has no effect on Ascend.",
            ":param disable_licm: Tells the compiler it shouldn't hoist loop invariant",
            "    code outside the loop. This is often useful to avoid creating long liveranges",
            "    within a loop.",
        ],
    },
    "triton.language.ravel": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64, Ascend 950 does not support fp64.",
        ],
        "example":
        "triton.language.ravel",
    },
    "triton.language.reduce": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64; Ascend 950 does not fp8e4(E4M3), fp8e5(E5M2).",
            "``keep_dims=True`` requires more test coverage; currently verified for 3D tensor with dim=2.",
        ],
        "example":
        "triton.language.reduce",
    },
    "triton.language.reduce_or": {
        "constraints": [],
        "example": "triton.language.reduce_or",
    },
    "triton.language.reshape": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64, Ascend 950 does not support fp64.",
            "``can_reorder``: Only supports False",
        ],
        "example":
        "triton.language.reshape",
    },
    "triton.language.rsqrt": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.rsqrt",
    },
    "triton.language.sigmoid": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.sigmoid",
    },
    "triton.language.sin": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.sin",
    },
    "triton.language.softmax": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
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
            "DataType: Ascend does not support bool, fp64, int32, int64, uint8.",
        ],
        "example": "triton.language.topk",
    },
    "triton.language.extra.cann.extension.sort": {
        "constraints": [
            "DataType: Ascend does not support bool, fp64, int32, int64, uint8.",
        ],
        "example": "triton.language.extra.cann.extension.sort",
    },
    "triton.language.split": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
        ],
        "example": "triton.language.split",
    },
    "triton.language.sqrt": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp64.",
        ],
        "example": "triton.language.sqrt",
    },
    "triton.language.sqrt_rn": {
        "constraints": [],
        "example": "triton.language.sqrt_rn",
    },
    "triton.language.static_print": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
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
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "``start``, ``end``, ``step`` must be compile-time constants (tl.constexpr).",
        ],
        "example":
        "triton.language.static_range",
        "replace_docstring": [
            "Iterator that counts upward forever.",
            "",
            ".. rubric:: Notes",
            "",
            "This is a special iterator used to implement similar semantics to",
            "Python's :code:`range` in the context of :code:`triton.jit` functions.",
            "In addition, it also guides the compiler to unroll the loop aggressively.",
            "",
            ":param arg1: the start value.",
            ":param arg2: the end value.",
            ":param step: the step value.",
        ],
    },
    "triton.language.store": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "``cache_modifier``: Ascend A5 SIMT-only accepts the Triton upstream strings (``''``, ``'.wb'``, ``'.cg'``, ``'.cs'``, ``'.wt'``); the backend folds them to a two-level cache / uncache mapping (``'.wb'`` -> cache, ``'.cg'`` / ``'.cs'`` / ``'.wt'`` -> uncache). Effective on global memory; ignored on shared memory. A2/A3 and non-SIMT-only paths ignore the modifier.",
            "``eviction_policy``: has no effect on Ascend.",
            "``volatile``: not accepted (the Triton upstream ``tl.store`` signature does not expose a ``volatile=`` keyword).",
            "Compatibility issues with branch and loop statements: \
                Complex pointer and mask calculations involving branches or loops may cause compilation failures.",
        ],
        "example":
        "triton.language.store",
        "replace_docstring": [
            "Store a tensor of data into memory locations defined by `pointer`.",
            "",
            "    (1) If `pointer` is a single element pointer, a scalar is stored.  In",
            "        this case:",
            "",
            "        - `mask` must also be scalar, and",
            "        - `boundary_check` and `padding_option` must be empty.",
            "",
            "    (2) If `pointer` is an N-dimensional tensor of pointers, an",
            "        N-dimensional block is stored.  In this case:",
            "",
            "        - `mask` is implicitly broadcast to `pointer.shape`, and",
            "        - `boundary_check` must be empty.",
            "",
            "    (3) If `pointer` is a block pointer defined by `make_block_ptr`, a block",
            "        of data is stored.  In this case:",
            "",
            "        - `mask` must be None, and",
            "        - `boundary_check` can be specified to control the behavior of out-of-bound access.",
            "",
            "`value` is implicitly broadcast to `pointer.shape` and typecast to `pointer.dtype.element_ty`.",
            "",
            ":param pointer: The memory location where the elements of `value` are stored",
            ":type pointer: `triton.PointerType`, or block of `dtype=triton.PointerType`",
            ":param value: The tensor of elements to be stored",
            ":type value: Block",
            ":param mask: If `mask[idx]` is false, do not store `value[idx]` at `pointer[idx]`",
            ":type mask: Block of triton.int1, optional",
            ":param boundary_check: tuple of integers, indicating the dimensions which should do the boundary check",
            ":type boundary_check: tuple of ints, optional",
            ":param cache_modifier: changes cache option for the store operation.",
            "  On Ascend A5 SIMT-only, accepts the upstream strings (``''``, ``'.wb'``, ``'.cg'``, ``'.cs'``, ``'.wt'``);",
            "  the backend folds them to a two-level cache / uncache mapping.",
            "  Effective on global memory; ignored on shared memory.",
            "  A2/A3 and non-SIMT-only paths ignore the modifier.",
            ":type cache_modifier: str, optional, should be one of {\"\", \".wb\", \".cg\", \".cs\", \".wt\"}",
            ":param eviction_policy: has no effect on Ascend.",
            ":type eviction_policy: str, optional, should be one of {\"\", \"evict_first\", \"evict_last\"}",
        ],
    },
    "triton.language.store_tensor_descriptor": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64, fp64, \
                Ascend 950 does not support fp64.",
            "`make_tensor_descriptor`, `load_tensor_descriptor` and `store_tensor_descriptor` \
                must be used as a suite in Triton 3.2.x. Do not mix them with `tl.load()` or `tl.store()`.",
            "Compatibility issues exist for certain functions (e.g. cast) in Triton 3.2.x."
        ],
        "example":
        "triton.language.store_tensor_descriptor",
    },
    "triton.language.sub": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support fp8, fp64.",
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
            "``mode``: must be one of 'all_cube','all_vector' or 'all'.",
            "``event_id``: value range is 0-15",
            "``status`` (Ascend extension): ascend_added",
            "``note`` (Ascend extension): A synchronization primitive of the Ascend Cube-Vector architecture "
            "unavailable in upstream triton.",
        ],
    },
    "triton.language.sync_block_set": {
        "constraints": [
            "``sender``: core type of the sender.it must be 'cube' or 'vector",
            "``receiver``: core type of the receiver.it must be 'cube' or 'vector' and cannot be the same as the sender.",
            "``event_id``: The value ranges from 0 to 15(16 independent events in total)",
            "``status`` (Ascend extension): ascend_added",
            "``note`` (Ascend extension): A synchronization primitive of the Ascend Cube-Vector architecture "
            "unavailable in upstream triton.",
        ],
    },
    "triton.language.sync_block_wait": {
        "constraints": [
            "``sender``: core type of the sender.it must be 'cube' or 'vector",
            "``receiver``: core type of the receiver.it must be 'cube' or 'vector' and cannot be the same as the sender.",
            "``event_id``: The value ranges from 0 to 15(16 independent events in total)",
            "``status`` (Ascend extension): ascend_added",
            "``note`` (Ascend extension): A synchronization primitive of the Ascend Cube-Vector architecture "
            "unavailable in upstream triton.",
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
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
            "Transpose on dimensions greater than 8 is not supported.",
        ],
        "example":
        "triton.language.trans",
    },
    "triton.language.umulhi": {
        "constraints": [
            "DataType: Ascend A2/A3/950 does not support int64.",
        ],
        "example": "triton.language.umulhi",
    },
    "triton.language.view": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, fp8e4, fp8e5, uint16, uint32, uint64.",
        ],
        "example": "triton.language.view",
    },
    "triton.language.where": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support fp64, uint16, uint32, uint64, uint8, Ascend 950 does not support fp64.",
        ],
        "example":
        "triton.language.where",
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
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64.",
        ],
        "example": "triton.language.zeros",
    },
    "triton.language.zeros_like": {
        "constraints": [
            "DataType: Ascend A2/A3 does not support uint16, uint32, uint64.",
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
}
