# Ascend community test migration

This directory mirrors selected top-level tests from `python/test/unit` so they
are collected by the Ascend integration suite without changing the community
source files.

- Source baseline: `main-dev@396df6cb5b001314e36f22220be07a560de44664`
- Migrated source files: 28
- Migrated top-level test functions: 228
- Historical exhaustive raw nodes: 2134 = 1858P + 275S + 1X
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

Fourteen directly validated functions are not part of this migration. Eleven
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

These three functions account for 50 historical Pass nodes. They are removed
as whole functions rather than partially retaining only their passing
parameters, preserving the function-level direct-migration boundary.

They remain adaptation or backend-investigation candidates; this direct
migration does not skip them, weaken their assertions, or silently rewrite
their setup.
