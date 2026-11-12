# Explicit L0 Paged MLA Decode

The Ascend 950 kernel uses BF16 Q and KV, dynamic ND2NZ row windows, explicit
L0 matrix multiplication and FP32 split reduction. It returns BF16 attention
output and optionally FP32 log-sum-exp. The tests cover full pages, ragged
requests and reuse of a prepared call with changed inputs.

Run the IR tests with the installed Triton Ascend wheel from the repository root.

```bash
tests=third_party/ascend/unittest/tla_ut
TLA_TEST_STAGE=ir python -m pytest --confcutdir="$tests" "$tests/kernels/mla/test_mla.py::test_ir"
```

On an Ascend 950 with matching CANN, PyTorch and torch_npu, run the accuracy tests.

```bash
TLA_TEST_STAGE=run python -m pytest --confcutdir="$tests" "$tests/kernels/mla"
```

For manual timing, run `python "$tests/kernels/mla/benchmark.py" --help`. The
optional Ascend C and CANNBot DSL comparisons require their respective vendor
packages or source samples.
