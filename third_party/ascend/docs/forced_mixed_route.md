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
An explicitly conflicting mode, including `off` or `report`, raises an error.
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
checks. Forcing does not make an unsupported factor or an illegal Mixed plan
legal. Unsupported requests raise errors. The ordinary auto route remains free to select any legal candidate.

The existing native capability contract restricts the Mixed factor set, selects
the forced legal candidate and materializes its scopes. The Python boundary
checks the effective route, selected factor, materialized-anchor count and scope-factor IR before accepting
the compile. Metadata records the forced route/factor and materialized TTIR
SHA256. NPU precision and performance still require a real runtime test.

For Ascend 950 Mixed lowering, an extracted index-select loop keeps its gathered
temporary in UB. CV lowering then transfers the gathered result to Cube. Leaving
that temporary unclassified can pull vector-produced indices into L1 and emit an
unsupported Cube scalar read, which failed on hardware with error 507015/271.
The opt-in `test_costmodel_forced_mixed_factors` regression checks F1/F2 precision,
distinct binaries, route environment changes and explicit factor changes through
the live JIT cache.
