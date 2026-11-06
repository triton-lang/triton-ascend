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

### Operation-rate calibration rules

These rules apply to every rate that the Stage formulas multiply a workload
count by, in particular `stage_resources.atomic_memory`.

1. **Measure the shape the model predicts.** A rate describes the operation as
   the model sees it: the TTIR operation, its element type, its address class
   and its mask. Measure it through the same compiler path the model selects
   for that operation. A rate measured on a different path, on a different
   instruction sequence, or on a private library call is a rate for something
   else, even when the arithmetic looks identical.
2. **Name the operation the way the model sees it, and record which template
   operation the slot prices.** The Stage model keys atomic rates by the TTIR
   operation name, so `fadd` and `add` are different slots. The target
   template names its operations once and separates the element type by the
   template parameter, so `add`/`fadd`, `max`/`umax`, `min`/`umin` and
   `exch`/`xchg` may lower to the same template operation. The slot name
   therefore stays the TTIR spelling, and the `source` of the slot says which
   template operation was measured.
3. **Separate the mechanisms, not the spellings.** When one operation lowers to
   more than one hardware mechanism, each mechanism needs its own rate, and
   the profile must say which mechanism it prices. The float add/min/max
   atomics are the current example: they have no native instruction on this
   target and are expanded into a load, a floating-point operation and a
   compare-and-swap retry loop, while the integer ones are a single
   instruction. A rate that does not distinguish them cannot be reconciled
   with a measured instruction count.
4. **Hold out part of the fit.** Fit on part of the sweep and report the error
   on the rest. A polynomial that needs more terms only to keep its training
   error low has overfitted the sweep rather than described the mechanism.
5. **State the domain.** A rate is valid for the address pattern, the element
   width, the footprint and the contention it was measured with. When the
   runtime footprint or the collision distribution can change the cost, the
   profile must say so, because a single rate cannot describe both regimes.
6. **Record the artifact.** `source` must name the probe, the raw result and
   the calibration step, so that the value can be recomputed. A rate whose
   probe artifact was not retained must say that explicitly and carry
   `confidence` `none`, which is what the uncalibrated atomic rates now do.
7. **Do not fill an uncalibrated slot with a plausible constant.** Leave the
   slot absent, or carry it with `confidence` `none` and a description of the
   gap. A constant that looks plausible is harder to remove later than a
   missing value.

### Reduction expression strings

First- and middle-axis segment `condition` and `expression` values are strings.
The loader validates metadata and segment structure, then parses expressions
into `ReductionExpression` trees on the first lookup of each
axis/mode/op/dtype group. Parsed groups are shared and initialized once,
including when multiple threads query the same group. Object-form expressions
are no longer accepted. Formula coefficients,
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
16 arguments per function. Invalid expression syntax is reported when the group
is first used; it is
propagated as a configuration error rather than treated as an unsupported
workload. Metadata and field-type errors still reject the initial load.
Call `ReductionProfile::validateAllFunctions()` before publishing a profile
or in validation tools/tests to check every expression, including unused
groups. This also warms every group, so normal compilation does not call it.
The format does not execute arbitrary code.
