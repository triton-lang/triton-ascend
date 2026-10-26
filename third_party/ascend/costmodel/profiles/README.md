# Ascend Cost Model profiles

This directory is the canonical source for target-specific Cost Model data.

- `microbench/` contains model-neutral hardware measurements shared by the
  absolute/autotune model and the SIMD/SIMT Route Model.
- `simd_simt/` contains Route Model policy, calibration, and schema
  feedback data.
- `indirect_memory/` contains frozen fitted parameters, referenced by the
  selection profile's `indirect_memory_profile`. They are not measurements.
  Schema 1 uses SYS_CNT/system-cycle units and flat named numeric arrays;
  `*Nodes` stores `[feature, value, left, right]` tuples, one tree per row,
  and `*Roots` stores node indices. Mask coefficients have 17 entries per
  model, including zero padding. Feature definitions remain in C++.
  The loader validates dimensions, versions, finite values and tree bounds.
  Both profile-cache invalidation and the compiler asset hash include this
  dependency; changing coefficients does not require rebuilding the compiler.
  Profiles enabling fitted indirect models must supply this reference;
  copies of earlier selection profiles need the same relative/absolute path.

Python packaging copies these files to
`triton/backends/ascend/costmodel_profiles/`. That installed directory is a
generated runtime asset location; it is not the source owner of the profiles.

Run the active-profile wiring and evidence audit after changing either target:

```bash
python3 microbench/data_provider/audit_active_simd_simt_profiles.py
```

The audit rejects obsolete route multipliers and dangling measurement IDs.
It also keeps the incomplete 950DT profile non-runnable and prints every 950PR
selection parameter that remains an explicit, uncalibrated fallback. This is
not limited to transition parameters: generic scalar/issue, indirect-memory,
and control-flow fallbacks are included.

The individual evidence validators recompute parameters from raw samples
instead of trusting copied JSON values. In particular:

```bash
python3 microbench/data_provider/validate_operation_rate_evidence.py
python3 microbench/data_provider/validate_setup_evidence.py
python3 microbench/data_provider/validate_control_evidence.py
python3 microbench/data_provider/validate_scan_evidence.py
python3 microbench/data_provider/validate_dot_dependency_evidence.py
python3 microbench/data_provider/validate_indirect_gather_evidence.py
python3 microbench/data_provider/validate_recurrence_group_evidence.py
python3 microbench/data_provider/validate_continuous_memory_evidence.py
```

The control and scan validators may finish successfully while printing
`promotion=REJECTED`. That means the raw SYS_CNT table is internally valid,
but subtraction does not identify the independent non-negative parameter
assumed by the current Stage formula. Such data is evidence for a formula gap,
not permission to fit a convenient constant.
