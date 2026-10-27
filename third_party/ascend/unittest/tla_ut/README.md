# TLA Tests (3-in-1, by op category)

Unified test scripts for TLA ops. Each script runs three stages in one file,
driven by env `TLA_TEST_STAGE`:

| Stage     | Verifies                                         | NPU needed |
|-----------|--------------------------------------------------|-----------|
| `ir`      | offline HIVM IR handed to npuir (structure asserts) | no      |
| `compile` | `.o`/`.npubin` artifacts in `TRITON_CACHE_DIR`      | yes     |
| `run`     | device launch + accuracy (`CAST_BACK OK`, ...)      | yes     |

Default `all` auto-degrades: ir only when no NPU, full chain when NPU is up.
Triton references live in the same case file (`TRITON_CAST_BACK OK`, ...).

## Layout

- `vector/` — SIMD/AIV ops (current 7 cases).
- `utils/` — shared helpers (`tla_test_utils.py`).
- `cube/` `mix/` — future matmul / mixed ops, same template.

## How to verify

```bash
cd third_party/ascend/language/tla/tests/vector
TLA_TEST_STAGE=ir      python cast_back.py   # 1. IR only, no NPU
TLA_TEST_STAGE=compile python cast_back.py   # 2. artifacts (needs Ascend driver)
TLA_TEST_STAGE=run     python cast_back.py   # 3. real-device accuracy (needs NPU)
python cast_back.py                          # 4. all (auto-degrade)
```

On a real NPU machine: use the TLA venv python, set `ASCEND_RT_VISIBLE_DEVICES`,
`TRITON_CACHE_DIR`, `PYTHONUNBUFFERED=1`, sync `vector/` from the repo, and run
each case. Pass criteria: `IR PASS` / `COMPILE PASS` / `<OP> OK` /
`ALL PASS: <op> ir compile run`.

## One-click pytest runner

The same cases are pytest-collectable (`conftest.py` collects every `*.py` via
`pytest_collect_file` since cases are named by op, not `test_*`; it also maps
`TLA_TEST_STAGE` to skips and provides the `output_dir` fixture), so any
directory level can be run directly with pytest:

```bash
TLA_TEST_STAGE=ir pytest third_party/ascend/unittest/tla_ut/vector   # IR only, no NPU
TLA_TEST_STAGE=compile pytest third_party/ascend/unittest/tla_ut/vector
TLA_TEST_STAGE=run pytest third_party/ascend/unittest/tla_ut/vector
pytest third_party/ascend/unittest/tla_ut/vector                     # all (auto-degrade)
pytest third_party/ascend/unittest/tla_ut                            # whole tla_ut
pytest .../tla_ut/vector -k rope                                      # subset
```

Collection and stage logic live in `tla_ut/conftest.py` and apply to the whole
`tla_ut` subtree from any working directory (including `pytest .` at the
`unittest` root). The root `unittest/conftest.py` NPU-per-worker assignment
applies as usual; `tla_ut/conftest.py` shadows it for no-NPU ir-only mode. The
per-case `python <case>.py` entry keeps working unchanged.
