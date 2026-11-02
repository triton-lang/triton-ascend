# TLA MMAD transposition and actual extents

```python
tla.cube.mmad(
    a, b, acc, transpose_b=False, clear_accum=False, unit_flag_ctrl=None,
    *, unit_flag_group_id=None, transpose_a=False,
    real_m=None, real_k=None, real_n=None,
)
```

Multiply two L1 buffers into an L0C accumulator. `clear_accum=True` initializes
that accumulator with the product; otherwise the product is added to its
existing contents. The operation lowers directly to `hivm.hir.mmadL1`.

## Physical shapes and transposition

Inputs may have 2-D logical shapes or 4-D NZ shapes. The 4-D shape
`(cols / b1, rows / b0, b0, b1)` represents the logical matrix `(rows, cols)`.
Each transpose flag applies to that represented matrix before checking matrix
dimensions and computing the product:

| Flags | A before transpose | B before transpose | Product |
|---|---|---|---|
| Neither | `(M, K)` | `(K, N)` | `(M, N)` |
| `transpose_a=True` | `(K, M)` | `(K, N)` | `(M, N)` |
| `transpose_b=True` | `(M, K)` | `(N, K)` | `(M, N)` |
| Both | `(K, M)` | `(N, K)` | `(M, N)` |

The flags become the native `a_transpose` and `b_transpose` attributes. They do
not request a separate vector transpose or change the L1 allocation. The two
input element types and the reduction dimensions after transposition must
match. Accumulator element-type support remains subject to the native MMAD
operation; the device tests use FP16/BF16 inputs and FP32 accumulation.

Existing positional arguments retain their order. `transpose_a` and the actual
extents are keyword-only. Non-square `transpose_b` inputs are checked using
the transposed dimensions, rather than treating `(N, K)` as `(K, N)`.

## Leading submatrices

`real_m`, `real_k` and `real_n` restrict the operation to the leading submatrices
of the transposed logical inputs. Each explicit extent must be a positive
compile-time integer, a multiple of 16, and no greater than its corresponding
full dimension. Omitted extents use the full dimension. These values are not
runtime loop bounds and do not supply padding or masking.

For logical input dimensions `(M, K)` and `(K, N)`, the result is:

```text
product = op(A)[:real_m, :real_k] @ op(B)[:real_k, :real_n]
acc = product                       # clear_accum=True
acc = acc + product                 # clear_accum=False
```

The full input reduction dimensions must agree even when `real_k` is smaller.
The accumulator must have exactly the logical shape `(real_m, real_n)`: changing
the actual output dimensions also changes its packed L0C layout. Allocating a
larger accumulator and expecting the remainder to be preserved is unsupported.

For an attention PV tail, P can remain physically transposed in L1 while the
operation uses only its valid reduction extent:

```python
# P represents (256, 128), V represents (256, 64), both in NZ form.
p = tla.alloc((8, 16, 16, 16), tl.bfloat16, space="l1")
v = tla.alloc((4, 16, 16, 16), tl.bfloat16, space="l1")
acc = tla.alloc((4, 8, 16, 16), tl.float32, space="l0c")
# After populating P/V and establishing the required data dependencies:
tla.cube.mmad(p, v, acc, transpose_a=True, real_k=128, clear_accum=True)
```

This computes a `(128, 64)` result from `P[:128, :].T @ V[:128, :]`. The caller
is responsible for populating the inputs and establishing synchronization;
actual extents do not replace ready/free buffer dependencies.

`unit_flag_ctrl` and `unit_flag_group_id` retain their existing behavior. They
are forwarded independently of transpose flags and actual extents. This API
change does not enable HF32 or change arithmetic precision.

## Tests

[MMAD three-stage tests](../../../../third_party/ascend/unittest/tla_ut/cube/mmad.py)
cover non-square NN/NT/TN/TT with FP16 and BF16 inputs, individual and combined
actual extents, and clearing/accumulation. The reduced-M NT case exercises the
attention QK-tail pattern; the reduced-K TN case exercises the PV-tail pattern.
These are focused API tests, not a full attention benchmark.

The run stage uses two seeds, a CPU FP32 reference computed from the quantized
inputs, finite checks, and NaN-initialized output buffers. Values outside actual
input extents remain nonzero, making accidental full-tile computation visible.
Host validation tests in
[`mmad_frontend_contract.py`](../../../../third_party/ascend/unittest/tla_ut/cube/mmad_frontend_contract.py)
cover mixed ND/NZ operands, invalid bounds/types/spaces, accumulator shape, and
legacy positional arguments. See the
[TLA test instructions](../../../../third_party/ascend/unittest/tla_ut/README.md)
for stage selection and environment requirements.
