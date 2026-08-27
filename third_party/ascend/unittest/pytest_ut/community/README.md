# Ascend community test migration

This directory mirrors selected top-level tests from `python/test/unit` so they
are collected by the Ascend integration suite without changing the community
source files.

- Source baseline: `main-dev@396df6cb5b001314e36f22220be07a560de44664`
- Migrated source files: 28
- Migrated top-level test functions: 231
- Historical exhaustive raw nodes: 2184 = 1908P + 275S + 1X
- Historical Fail/Error nodes in this selected set: 0/0

The selected test functions and their parametrization decorators are copied
verbatim, except for one device-safety adaptation in `test_int_annotation`:
its temporary output buffer is enlarged from one element to four because the
unchanged kernel stores at offset `v=3`. Unselected top-level `test_*`
functions are omitted, while shared module helpers are retained. `conftest.py`
supplies the `npu` device and the same cache/knob/allocator fixtures used by the
source test tree.

`MIGRATION_MANIFEST.tsv` is the auditable function-by-function mapping. The
`destination_file` column is relative to `third_party/ascend/unittest/pytest_ut`.
The result columns mean Passed, Failed, Skipped, XFailed, and Error.

The historical result columns are screening evidence only. That frozen run
used temporary compatibility plugins, so it is not validation of this copied
tree. Current direct validation is recorded in the pull request.

## Candidates excluded after direct validation

Eleven initially screened functions are not part of this migration:

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

They remain adaptation or backend-investigation candidates; this direct
migration does not skip them, weaken their assertions, or silently rewrite
their setup.
