# Paged MLA with explicit L0 scheduling

BF16 packed Q and KV, causal paged attention, FP32 online softmax, and BF16
output with FP32 LSE. The tuned kernel stages QK and PV through L0 and fuses
the split merge when its workspace fits. The tests cover the full context and
ragged page tables.

Run from the repository root:

```sh
suite=third_party/ascend/unittest/tla_ut/kernels/mla
TLA_TEST_STAGE=ir python -m pytest -q "$suite"
TLA_TEST_STAGE=run python -m pytest -q "$suite" -k test_run
TLA_TEST_STAGE=run TLA_RUN_PERF=1 python -m pytest -s "$suite" -k test_perf
```

Historical device median on Ascend 950PR with CANN 9.1.0, B=8, H=24,
Sq=4, Skv=143360, QK=576, V=512 and page size 128:

| Implementation | Device median |
| --- | ---: |
| TLA explicit L0 | 818.277 µs |
| Source-built Ascend C | 824.594 µs |
| CANNBot DSL | 861.277 µs |

Timing sums every device task in an invocation, including the merge when
separate. The benchmark checks the FP32 reference before profiling.

On Triton Ascend `3cc904692` with AscendNPU-IR `0167d41d` and CANN 9.1.0,
the full and ragged cases passed IR and accuracy checks. The full case
measured 830.7 µs device time and 878.7 µs synchronized wall time
(three rounds of 40 device samples).
