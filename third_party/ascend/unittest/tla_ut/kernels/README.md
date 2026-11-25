# Composite kernel tests

```text
kernels/
  utils.py              # Lazy operator loading, device snapshots, profiling and result files
  dsa/
    dsa_kernel.py       # Operator implementation
    test_dsa.py         # Global settings, input/reference, test_run and test_perf
    README.md
```

DSA currently covers four prefill shapes (B=1, SQ=KV=8192, TopK=2048, H=16/32/64/128).
Run accuracy tests from the repository root:

```bash
TLA_TEST_STAGE=run pytest -q third_party/ascend/unittest/tla_ut/kernels/dsa
```

Each kernel keeps its shapes, input generation and numerical reference in its own test module.
Import reusable helpers with `from kernels.utils import ...`; the parent `tla_ut/conftest.py`
provides the import root and assigns the NPU. Helpers import device libraries lazily so
IR/compile-stage collection does not import the operator implementation.

[Paged MLA](mla/README.md) follows the same layout under `kernels/mla/`.
Its test module covers IR, device accuracy and opt-in serial profiling.
