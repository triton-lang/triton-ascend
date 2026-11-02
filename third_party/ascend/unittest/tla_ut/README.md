# TLA Tests (3-in-1)

Each case is one script with three stages selected by `TLA_TEST_STAGE`:

| Stage     | Verifies                                            | NPU needed |
|-----------|-----------------------------------------------------|------------|
| `ir`      | offline HIVM IR handed to npuir (structure asserts) | no         |
| `compile` | `.o` / `.npubin` artifacts in `TRITON_CACHE_DIR`    | yes        |
| `run`     | device launch + accuracy                            | yes        |

The default `all` auto-degrades to `ir` when no NPU is available.

## Layout

- `simd/` — SIMD instruction cases: per-interface files (`vadd`, `vsub`,
  `vor`, `vxor`, `vmins`, `vsstb`, `vdiv`, `vexp`, `vmax`, `vmin`, `vmaxs`,
  `vdup`, `vdups`, `vdupv`), grouped cases (`add_cmp`, `elementwise`,
  `hardware_spec`), the runtime-N long-vector case (`simd_tail`), and the
  tilelang-derived `vecadd`.
- `vector/` — TLA vector kernel cases (`cast_back`, `expand`, `rope`, ...).
- `sync/` — `sync_pipe` (intra-core pipes) and `sync_block` (cross-core block flags, INTER/SUBBLOCK/BLOCK).
- `dma/` — explicit load/store/copy/fixpipe/nd2nz cases.
- `cube/` — matmul / mixed-core cases.
- `utils/` — shared helpers.
- `simd_frontend_contract.py` — host-only SIMD frontend contract tests.

## How to verify

```bash
cd third_party/ascend/unittest/tla_ut/simd
TLA_TEST_STAGE=ir      python vadd.py
TLA_TEST_STAGE=compile python vadd.py
TLA_TEST_STAGE=run     python vadd.py
```

The same three stages apply to cases under `vector/`, `sync/`, and `cube/`.
On a real NPU, provide a TLA-capable Triton build, torch/torch_npu, and set
`ASCEND_RT_VISIBLE_DEVICES` and `TRITON_CACHE_DIR`. Passing cases print
`IR PASS` / `COMPILE PASS` / `<OP> OK` / `ALL PASS`.

## One-click pytest runner

`conftest.py` collects every case file and maps `TLA_TEST_STAGE` to skips and
the `output_dir` fixture. Any directory level can be run directly:

```bash
TLA_TEST_STAGE=ir pytest third_party/ascend/unittest/tla_ut/simd
TLA_TEST_STAGE=ir pytest third_party/ascend/unittest/tla_ut/vector
TLA_TEST_STAGE=ir pytest third_party/ascend/unittest/tla_ut
pytest .../tla_ut/simd -k vadd
```

Host-only SIMD contract tests do not need a build or NPU:

```bash
pytest third_party/ascend/unittest/tla_ut -k simd_frontend
```
