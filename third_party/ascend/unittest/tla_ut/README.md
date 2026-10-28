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
