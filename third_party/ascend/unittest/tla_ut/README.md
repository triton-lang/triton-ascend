# TLA Tests (3-in-1)

Each case is one script with three stages selected by `TLA_TEST_STAGE`:

| Stage     | Verifies                                            | NPU needed |
|-----------|-----------------------------------------------------|------------|
| `ir`      | offline HIVM IR handed to npuir (structure asserts) | no         |
| `compile` | `.o` / `.npubin` artifacts in `TRITON_CACHE_DIR`    | yes        |
| `run`     | device launch + accuracy                            | yes        |

The default `all` auto-degrades to `ir` when no NPU is available.

## Layout

- `scalar/` — select/abs/maximum/minimum/XOR/cast/max_value/cdiv/infinity/clamp cases.
  Per-interface files own `test_ir/test_compile/test_run` for actual writeback
  kernels: buffer/VF structure, target objects, and device accuracy.
  `scalar_frontend_contract.py` owns host-side parameter validation, type
  promotion, literals, diagnostics, constant folding, scalar-only adapter IR,
  and JIT helper composition (requires a TLA-enabled build, no NPU).
- `simd/` — SIMD instruction cases: per-interface files (`vadd`, `vsub`,
  `vsubc`, `vor`, `vxor`, `vmins`, `vsstb`, `vdiv`, `vexp`, `vmax`, `vmin`,
  `vmaxs`, `vdup`, `vdups`, `vdupv`), grouped cases (`add_cmp`, `elementwise`,
  `hardware_spec`), the runtime-N long-vector case (`simd_tail`), and the
  tilelang-derived `vecadd`.
- `vector/` — TLA vector kernel cases (`cast_back`, `expand`, `rope`, ...),
  including `subview.py` for compact/pitched aliases and offset-preserving reshape,
  and `view_frontend_contract.py` for host-only bounds/layout contracts.
- `sync/` — `sync_pipe` (intra-core pipes), `sync_block` (cross-core block flags,
  INTER/SUBBLOCK/BLOCK), `sync_intra_block` (`INTRA` mode), and `buffer_sync`
  (`mutex_lock` / `mutex_unlock`).
- `dma/` — explicit load/store/copy/fixpipe/nd2nz cases.
- `cube/` — matmul / mixed-core cases, including `mmad.py` for transposed
  operands and actual extents, and `mmad_frontend_contract.py` for host contracts.
- `dtype/` — host-only contract tests for the scalar dtypes exported by `tla`.
- `kernels/` — composite kernel accuracy and performance tests, with shared utilities.
  Includes [DSA](kernels/dsa/README.md); IR/compile stages skip runtime tests.
- `utils/` — shared helpers.
- `simd_frontend_contract.py` — host-only SIMD frontend contract tests.

## How to verify

```bash
cd third_party/ascend/unittest/tla_ut/simd
TLA_TEST_STAGE=ir      python vadd.py
TLA_TEST_STAGE=compile python vadd.py
TLA_TEST_STAGE=run     python vadd.py
```

The same three stages apply to cases under `scalar/`, `vector/`, `sync/`, and `cube/`.
On a real NPU, provide a TLA-capable Triton build, torch/torch_npu, and set
`ASCEND_RT_VISIBLE_DEVICES` and `TRITON_CACHE_DIR`. Passing cases print
`IR PASS` / `COMPILE PASS` / `<OP> OK` / `ALL PASS`.

Run scalar API contracts independently with:

```bash
TLA_TEST_STAGE=ir pytest third_party/ascend/unittest/tla_ut/scalar/scalar_frontend_contract.py
```

Running the whole `scalar/` directory collects both the API contracts and the
per-interface kernel tests. Frontend/IR success does not imply target compilation
or NPU accuracy has passed.

## One-click pytest runner

`conftest.py` collects every case file and maps `TLA_TEST_STAGE` to skips and
the `output_dir` fixture. Any directory level can be run directly:

```bash
TLA_TEST_STAGE=ir pytest third_party/ascend/unittest/tla_ut/simd
TLA_TEST_STAGE=ir pytest third_party/ascend/unittest/tla_ut/scalar
TLA_TEST_STAGE=ir pytest third_party/ascend/unittest/tla_ut/vector
TLA_TEST_STAGE=ir pytest third_party/ascend/unittest/tla_ut
pytest .../tla_ut/simd -k vadd
```

Host-only SIMD contract tests do not need a build or NPU:

```bash
pytest third_party/ascend/unittest/tla_ut -k simd_frontend
```
