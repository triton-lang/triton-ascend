# Ascend Cost Model profiles

This directory is the canonical source for target-specific Cost Model data.

- `microbench/` contains model-neutral hardware measurements shared by the
  absolute/autotune model and the SIMD/SIMT Route Model.
- `simd_simt/` contains Route Model policy, calibration, and schema
  feedback data.

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

### Reduction expression strings

First- and middle-axis segment `condition` and `expression` values are strings.
The evaluator parses them once into `ReductionExpression` trees when loading a
profile. Object-form expressions are no longer accepted. Formula coefficients,
conditions and evaluation order are unchanged by this encoding.

Supported infix operators are `+`, `-`, `*`, `/`, `**`, `<`, `<=`, `>`, `>=`,
`==`, and `!=`, with parentheses for grouping. Feature names are restricted to
the evaluator's known workload features. Function calls support `add`, `sub`,
`mul`, `div`, `pow`, `min`, `max`, `neg`, `indicator`, `ceil`, `sqrt`, `log2`,
comparison names (`lt`, `le`, `gt`, `ge`, `eq`, `ne`), and logical operations
`all_of`, `any_of`, `logical_not`. Logical aliases `and`, `or`, `not` are also
accepted as function names. Negative numeric literals and scientific notation
are supported; use `neg(expression)` to negate a nonliteral expression.

Examples: `((12.193861 * R) * G)` and
`all_of(W == 4, route_id == 2)`. Powers require a literal exponent in [-4, 4].
Parsing limits are 16,384 input characters, 1,024 nodes, tree depth 32, and
16 arguments per function. Invalid expressions reject the profile rather than
silently producing a cost. The format does not execute arbitrary code.
