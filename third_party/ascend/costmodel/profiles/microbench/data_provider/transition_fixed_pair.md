# Local SIMD/SIMT fixed scope-pair calibration

## Published measurement

```text
simd_simt.scope_handoff.fixed_pair = 184.663 SYS_CNT cycles/pair
```

This is an operational fixed cost for one complete local
`SIMD -> SIMT VF -> SIMD` scope pair. It includes VF entry/prologue and
exit/epilogue. It excludes tensor live-in/live-out transfer and the marginal
SIMT body cost.

## Method

One SIMD host function invokes exactly one SIMT scope. The scope contains `K`
independent, correctness-checked 64-element tensor bodies at fixed, disjoint
addresses. With the scope count and VF invocation count held constant, the
measured timer-baseline delta is modeled as:

```text
D_K = C_fixed_pair + K * C_body + E_K
```

The body cost is absorbed by the slope and the intercept estimates the fixed
scope-pair cost.

Configuration:

```text
device=1
factor=1
grid=56
AutoBlockify=off
fixed program stride=512
five independent processes and cache directories
20 warmups and 20 measured samples per K
forward and reverse K order
```

Five-process mean-point fit:

```text
D_K = 184.663081 + 11.443895 * K
R^2 = 0.9992309
maximum absolute residual = 1.005 cycles
process-vector bootstrap 95% interval = [184.431, 184.917] cycles
```

The profile stores `184.663` cycles. This measurement is a pair and must not
be presented as two independently measured directional transition latencies.
