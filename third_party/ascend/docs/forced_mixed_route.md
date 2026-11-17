# Force a CostModel Mixed route for diagnostics

This control runs the normal CostModel analysis and materializes a legal Mixed
candidate at one requested scope SuperBlock factor for an explicitly forced
diagnostic experiment.

Enable it for an existing launch with this environment variable.

```bash
export TRITON_ASCEND_AUTO_SIMT_FORCE_ROUTE=mixed_simd_simt
python your_test.py
```

When the launch does not specify them, this supplies
`compile_mode="simd_simt"` and `auto_simt_scope_mode="auto"`.
The existing process-wide compile-mode override retains precedence. `None`
compile modes and empty scope defaults resolve to the forced-mode defaults.
An effective conflicting mode, including `off` or `report`, raises an error.
Pass F through the existing `superblock_factor` launch option. Its default is 1.
An explicit `auto_simt_force_route` overrides the route environment variable.

```python
kernel[grid](
    ...,
    compile_mode="simd_simt",
    auto_simt_scope_mode="auto",
    auto_simt_force_route="mixed_simd_simt",
    superblock_factor=2,
)
```

Set `auto_simt_force_route=""` on a launch to disable forcing even when the
environment variable is set, or unset the route variable to restore the
ordinary defaults. The route value must be empty or exactly `mixed_simd_simt`.

Controls are resolved before the JIT in-process cache lookup and also by
`AscendBackend.parse_options` for direct compilation. They participate in the
serialized compiler options and disk-cache hash, including at F1. Changing
`superblock_factor` in one Python process selects the corresponding executable.

The scope backend currently supports F1/F2/F4. Admission also requires the
original warp, target, logical-program, resource and scope-materialization
checks. Unsupported requests raise errors. F2/F4 also require admission through
the existing AutoBlockify V1 policy. For an eligible kernel, V1 can follow the
existing `TRITON_ALL_BLOCKS_PARALLEL=1` setting. Blacklist and resource checks
continue to determine admission.

The existing native capability contract restricts the Mixed factor set, selects
the forced legal candidate and materializes its scopes. The Python boundary
checks the effective route, selected factor, materialized-anchor count and
scope-factor IR before accepting the compile. Metadata records the forced route/factor and materialized TTIR
SHA256. NPU precision and performance still require a real runtime test.

The `test_costmodel_forced_mixed_factors` regression uses a one-dimensional
indirect load with multiple logical-program loop iterations. It verifies
Auto/F1/F2 route changes, precision, distinct compilation keys and materialized
factor IR, and cache reuse in an isolated child process.
