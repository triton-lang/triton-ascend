# Ascend Cost Model architecture

This directory contains two cost models and one shared evidence layer.

```text
profiles/microbench
        |
        v
AscendModelProfile
   |             |
   v             v
AscendModelAnalysis      AscendModelRouteModel
(absolute/HIVM)          (SIMD/SIMT routing)
          \               /
           v             v
           AscendModelTransforms / backend integration
```

The two models share measurements, not objectives or scoring formulas:

- `configs/`, `include/AscendModel/Analysis`, `lib/AscendModel/Analysis`,
  `IR`, and the original transforms implement the absolute/autotune and HIVM
  model.
- `profiles/microbench`, `include/AscendModel/Profile`, and
  `lib/AscendModel/Profile` own model-neutral measurements plus their loader,
  units, clock domains, target checks, and provenance.
- `profiles/simd_simt`, `include/AscendModel/RouteModel`, and
  `lib/AscendModel/RouteModel` own SIMD/SIMT feature extraction, Coverage,
  conditional candidate scoring, post-score checks, selection reporting, and
  scope materialization.

`third_party/ascend/backend/compiler.py` is an integration layer: it schedules
the native passes and locates installed assets. `python/setup.py` copies
`profiles/` to `triton/backends/ascend/costmodel_profiles/` while building a
Python package. That installed directory is generated runtime data, not a
source-of-truth Cost Model directory.

## Coverage boundary

Coverage is checked before candidate scoring.  An unknown `scf.for` trip count
is rejected by default; the only bounded exception is a recognized
`triangular_solve_loop` anchor group.  That exception still requires the
small-tensor limit, one to four materializable anchors, and the mask/reduction
limits from the SIMD/SIMT profile.  Removing the triangular mechanism evidence
therefore returns `unknown_loop_trip_count` rather than silently admitting a
generic dynamic loop.  The regression tests for both paths are in
`unittest/costmodel_ut/SimdSimtCostModelTest.cpp`.
