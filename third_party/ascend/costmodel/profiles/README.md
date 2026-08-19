# Ascend Cost Model profiles

This directory is the canonical source for target-specific Cost Model data.

- `microbench/` contains model-neutral hardware measurements shared by the
  absolute/autotune model and the SIMD/SIMT Route Model.
- `simd_simt/` contains Route Model policy, calibration, schema, and DES
  feedback data.

Python packaging copies these files to
`triton/backends/ascend/costmodel_profiles/`. That installed directory is a
generated runtime asset location; it is not the source owner of the profiles.
