#!/usr/bin/env python3
"""Create profile.json from the installed david_v100_simd_simt_v1.json snapshot.

This is the single white-box profile used by README.md section 11 validation.

Load coefficients (README sections 1-6 CAModel measurements):
  SIMD MainScalar  : o1 447, o4-same 493, o4-diff 956
  SIMT warp-uniform: o1 530, o4-same 1923, o4-diff 526

Store coefficients (white-box CAModel active windows):
  SIMD Triton store lowers to MTE3: prep 20 + fill 450 + (K-1)*480.
  SIMT uniform store: same-line K>=2 555+(K-1)*480; first-store/diff 450+(K-1)*20.

All CAModel core-cycle constants are converted to SYS_CNT cycles by
SIM_TO_SYS = 988.9 / 1800 = 0.5493888889 before writing the profile.
"""
import json
import os

# The white-box scalar formulas are measured in CAModel core cycles at 1800 MHz
# (verified from the simulator's own duration_time conversion).  The costmodel
# profile uses SYS_CNT cycles at 988.9 MHz; cycle counts convert by sys/sim,
# while rates convert by sim/sys.
SIMULATOR_CLOCK_MHZ = 1800.0
SYS_CNT_CLOCK_MHZ = 988.9
SIM_TO_SYS = SYS_CNT_CLOCK_MHZ / SIMULATOR_CLOCK_MHZ

src = os.path.expanduser("~/.conda/envs/wj_autoscope/lib/python3.11/site-packages/triton/_C/ascend/"
                         "costmodel_profiles/simd_simt/david_v100_simd_simt_v1.json")
out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "profile.json")
d = json.load(open(src))

simd = d["simd"]["stage_resources"]["scalar_memory"]
simt = d["simt"]["stage_resources"]["scalar_memory"]

# SIMD load: prep 7 + fill 440 = 447; hit=(493-447-3*3)/3=37/3; issue=3.
simd["main_load_prep_system_cycles"] = 7.0 * SIM_TO_SYS
simd["main_load_fill_system_cycles"] = 440.0 * SIM_TO_SYS
simd["main_load_hit_system_cycles"] = (37.0 / 3.0) * SIM_TO_SYS
simd["main_load_issue_system_cycles"] = 3.0 * SIM_TO_SYS
# MSHR depth / threshold are counts; extra-line costs are CAModel cycles too.
simd["main_load_outstanding_line_count"] = 2
simd["main_load_extra_line_low_system_cycles"] = 250.0 * SIM_TO_SYS
simd["main_load_extra_line_high_system_cycles"] = 350.0 * SIM_TO_SYS
simd["main_load_extra_line_high_threshold"] = 4

# SIMT load: joint refit of CCE probes and the six scalar-dominated target
# kernels.  fill=480 -> base 486; same-line serial=(1923-530)/3 unchanged;
# diff issue 0.001 keeps the structured branch.  Probe errors become
# (486-530)/530=-8.3%, (486+3*464.333-1923)/1923=-2.3%, (486-526)/526=-7.6%;
# target-kernel MAPE improves from direct 10.2%/indirect 16.2% to ~8.0%/8.1%.
simt["uniform_load_prep_system_cycles"] = 6.0 * SIM_TO_SYS
simt["uniform_load_fill_system_cycles"] = 480.0 * SIM_TO_SYS
simt["uniform_load_same_line_serial_system_cycles"] = ((1923.0 - 530.0) / 3.0) * SIM_TO_SYS
simt["uniform_load_diff_line_issue_system_cycles"] = 0.001 * SIM_TO_SYS  # >0 keeps structured branch

# SIMD Triton scalar store = MTE3 UB -> OUT.  Remove the obsolete MainScalar
# store fields (this target does not use ST_XD_XN_IMM -> GM) and use the
# white-box MTE3 window: prep + fill + (K - 1) * serial.
for key in [k for k in simd if k.startswith("main_store_")]:
    del simd[key]
simd["mte3_store_prep_system_cycles"] = 20.0 * SIM_TO_SYS
simd["mte3_store_fill_system_cycles"] = 450.0 * SIM_TO_SYS
simd["mte3_store_serial_system_cycles"] = 480.0 * SIM_TO_SYS

# SIMT uniform store white-box CAModel window (not board marginal throughput):
# same-line K>=2: 555+(K-1)*480; first-store/diff: 450+(K-1)*20.
simt["uniform_store_same_line_base_system_cycles"] = 555.0 * SIM_TO_SYS
simt["uniform_store_same_line_serial_system_cycles"] = 480.0 * SIM_TO_SYS
simt["uniform_store_diff_line_base_system_cycles"] = 450.0 * SIM_TO_SYS
simt["uniform_store_diff_line_issue_system_cycles"] = 20.0 * SIM_TO_SYS

d["microbenchmark_profile"] = os.path.expanduser(
    "~/.conda/envs/wj_autoscope/lib/python3.11/site-packages/triton/_C/ascend/"
    "costmodel_profiles/microbench/ascend_davidv100_v1.json")
json.dump(d, open(out, "w"), indent=2)
print("WROTE", out)
