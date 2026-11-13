# LLVM text compatibility checks

These tests follow the directory, filenames and `triton-opt %s | FileCheck %s`
organization of the [3.7 reference](https://github.com/triton-lang/triton-ascend/tree/e5991f1d4ab42456c9fcd5f2efa43ca7a5146a33/third_party/ascend/unittest/Conversion/LLVM20Compat).
The reference patch is `llvm_patch_ac5dc54.patch` (Git blob
`ac5b4256d93c50b8c6e6f89f0453bdd8c43baa3d`).

TA 3.6 pins LLVM `f6ded0be897e2878612dd903f7e8bb85448269e5` (LLVM 22),
whereas the reference uses LLVM 23. The port retains the reference's per-op
custom parsers/printers and restored `bufferization::ToMemrefOp` implementation.
Ordinary module printing produces the text passed directly from `ttadapter` to
BishengIR. There is no separate export option or bytecode stage.

## Comparison with the 3.7 reference

All 17 applicable LLVM files use the reference's change bodies. The patch has the
same LLVM file ordering, unified-diff format and 12-character blob IDs; context,
line numbers and blob IDs differ with the source revision. TA integration changes
the `ToBufferOp` construction sites to `ToMemrefOp` without changing the lowering
algorithms.

The following behaviors also follow the reference:

- `scf.for` printing omits `unsigned`.
- `linalg.matmul` uses the reference's named-op parser/printer and
  `linalg.batch_matmul` omits explicit indexing maps.
- `ToMemrefOfCast` constructs the intermediate memref from the source tensor's
  shape and element type, then creates `memref.cast`.
- `ToTensorOp::print` omits the explicit result-type suffix.
- The generic operation printer filters the `op_bundle_sizes` property by name.

The only operation adaptations omitted from the reference are required by the
pinned LLVM version:

| Operations / tests | LLVM 22 adaptation |
| --- | --- |
| `arith.divsi`, `arith.divui`, `arith.shrsi`, `arith.shrui` | The newer `isExact` property is absent. Omit those patch hunks and use native LLVM 22 inputs in the four matching tests. |
| `gpu.barrier` / `gpu_barrier.mlir` | The newer `memfence` / `address_spaces` syntax is absent. Omit the GPU hunks and test plain `gpu.barrier`. |
| `linalg.map` / `linalg_map_long.mlir` | The extra init region argument is absent. Omit that hunk and use the two-argument long form. The short-form test is unchanged. |

All 32 reference test filenames are retained. Twenty-six files are identical to
the reference; only the six version-dependent inputs listed above differ.

## Verification flow

1. Parse the input with TA's patched LLVM 22 `triton-opt`.
2. Print it using the compatible dialect printer.
3. Check the reference's output spelling and omitted fields with FileCheck.

Like the reference, parser input and printer output are not necessarily the same
syntax: `to_tensor` input requires `to tensor<...>`, while output omits it.
Generic input represents newer hints in the op tests. A `.ttadapter` override
goes directly to BishengIR as text.

`LLVM20Compat` keeps the reference's directory name. The deployed BishengIR may
identify itself as LLVM 19 with vendor backports. FileCheck verifies producer
output; consumer parsing and NPU execution require separate tests. These checks
cover the reference contract, not equivalence for arbitrary extended LLVM IR.

## How to run tests

Build TA with `-DTRITON_BUILD_UT=ON`, then run:

```bash
lit -v build/cmake.*/third_party/ascend/unittest --filter=LLVM20Compat
```

Or use matching tools to run a single file:

```bash
triton-opt third_party/ascend/unittest/Conversion/LLVM20Compat/OpCompat/arith_trunci.mlir \
  | FileCheck third_party/ascend/unittest/Conversion/LLVM20Compat/OpCompat/arith_trunci.mlir
```

The Python tests in `pytest_ut/test_bishengir_export.py` cover serialization and
debug-location controls. Backend stage connections, retired override diagnostics
and UB tuner input selection are covered by `test_compiler.py`.
