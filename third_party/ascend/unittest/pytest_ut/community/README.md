# Ascend community test migration

This directory mirrors selected top-level tests from `python/test/unit` so they
are collected by the Ascend integration suite without changing the community
source files.

- Source baseline: `main-dev@396df6cb5b001314e36f22220be07a560de44664`
- Migrated source files: 28
- Migrated top-level test functions: 227
- Historical exhaustive raw nodes: 2132 = 1856P + 275S + 1X
- Historical Fail/Error nodes in this selected set: 0/0

The selected test functions and their parametrization decorators are copied
verbatim, except for three Ascend adaptations. In `test_int_annotation`, the
temporary output buffer is enlarged from one element to four because the
unchanged kernel stores at offset `v=3`. In
`test_host_tensor_descriptor_matmul`, six block shapes are scaled down while
retaining their aspect ratios, `BLOCK_K`, and pipeline stages so the generated
kernel fits the target's UB capacity. In `test_kernel_in_thread`, each caller
thread explicitly selects the NPU that owns the test buffer because NPU device
selection is thread-local. Unselected top-level `test_*` functions are omitted,
while shared module helpers are retained. `conftest.py` supplies the `npu`
device and the same cache/knob/allocator fixtures used by the source test tree.

`MIGRATION_MANIFEST.tsv` is the auditable function-by-function mapping. The
`destination_file` column is relative to `third_party/ascend/unittest/pytest_ut`.
The result columns mean Passed, Failed, Skipped, XFailed, and Error.

The historical result columns are screening evidence only. That frozen run
used temporary compatibility plugins, so it is not validation of this copied
tree. Current direct validation is recorded in the pull request.

## Candidates excluded after direct validation

Fifteen directly validated functions are not part of this migration. Eleven
were excluded during the initial screening:

- `test_trans_2d`, `test_trans_4d`, `test_tma_gather`, and `test_tma_scatter`
  need an Ascend-specific replacement for unsupported NPU `torch.arange`
  setup in `int8` cases.
- `test_aggregate_with_constexpr`, `test_aggregate_with_tuple`,
  `test_function_name_mangling`, and `test_list_of_functions` need their
  FileCheck symbol prefixes adapted after relocation into this package.
- `test_compile_in_subproc` and `test_compile_in_forked_subproc` are
  incompatible with cold `fork` after the parent process initializes the NPU.
- `test_indirect_matmul` currently crashes `bishengir-compile` while lowering
  the generated Linalg IR on the source baseline.

A subsequent validation on a real `Ascend950PR_9579` at PR commit
`86796b858e44d6ecd3f3bf860a6c59779838ae37` excluded three more functions:

- `test_tensor_atomic_add_non_exclusive_offset` has one failing parameter out
  of 18: NPUBIN `PlanMemory` requires 2232320 bits of UB but only 1769472 bits
  are available.
- `test_propagate_nan` has eight failing parameters out of 12: six produce
  incorrect NaN propagation and two cannot select `fmaximum` during NPUBIN
  generation.
- `test_dot_multidim` has 16 failing parameters out of 20: 13 fail NPUBIN
  generation with an unexpected rewrite operation and three produce results
  that differ from the PyTorch reference.

The A5 CI run for PR commit `ec0872d829081e428fe5689286eb74ebfff61460`
([job log](https://github.com/triton-lang/triton-ascend/actions/runs/33737977388/job/100592990619))
identified one further exclusion:

- `test_dot_without_load` exposes an A5 code-generation regression in the CI
  x86 compiler `ascendnpu-ir_1.2.0_linux-x86-pr_2863.run`
  (BiSheng/AscendNPU-IR commit `f1168a57139a`). Its `float32` parameter
  produces all zeros instead of 32; `float16` passes. Fresh-cache A/B runs on
  `Ascend950PR_9579` reproduce the failure three times on both CI merge
  `6bfed4e7a210fe35462c6fd125cfb299af1e1c91` and the later merge
  `7ed5fae3df4c15326a0ac95d3d4f286290e3a6e7`. For each merge, the same
  compiler input passes three times with the CANN 9.1.0 bundled compiler
  `8796a8ac1508`. The entire function, including its passing `float16`
  parameter, is excluded pending a compiler fix; this removes two historical
  Pass nodes and the corresponding A5 community coverage.

These four subsequent exclusions account for 52 historical Pass nodes. They
are removed as whole functions rather than partially retaining only their
passing parameters, preserving the function-level direct-migration boundary.

They remain adaptation or backend-investigation candidates; this direct
migration does not skip them, weaken their assertions, or silently rewrite
their setup.
