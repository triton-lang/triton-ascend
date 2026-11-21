# Final paired-comparison statistics

`paired_stats.py` implements the reporting rules for the eleven original
BS128/H8 attention shapes. The caller supplies six fresh alternating AB/BA
rounds with ten selected hardware operation durations per arm in microseconds,
the exact case, stable input fingerprints and stable compiler/reference
identities. It checks consistency and reports the median of six arithmetic
block means. Both the median and all six pairs must satisfy
`CVPipeline < AscendC / 0.9` for a strict pass.

The N2048/HD128 exception stays labelled separately and does not become a
strict pass. Positive target margins below 2% request an independent complete
repeat. Missing/duplicate matrix shapes, mixed inputs/images/geometry, wrong
units and reported simulator/host timings are rejected.

The module does not collect timings or certify the caller's evidence. The
hardware collector, normal compiler path, full O/M checks, 64-forward stress and
input/binary identity still need separate verification. The tests use synthetic
samples to check these statistics; they are not measured performance results.
# Aggregate hardware CSV input

`op_statistic.py` selects exactly one row by physical device ID, full operator
name and core type. It requires microseconds, exactly ten selected attention
operations, a consistent total/count/average and valid extrema. It preserves
the raw CSV hash, resolved path, data-record index and original row.

The paired reporter also accepts `<arm>_op_statistic` blocks produced by that
reader. It uses total divided by the actual count as the block mean and keeps
the six-round median and six-of-six target checks. It does not invent ten raw
samples by repeating an aggregate average. Duplicate source records, changed
devices/operators and mixed input representations are rejected.

All fixtures in the tests are synthetic. These modules check selection,
consistency and arithmetic; the final collector and independent raw-evidence
audit must still establish actual hardware collection, fresh inputs/binaries,
correctness and admission. No hardware timing is claimed by these tests.

# Hardware collector

`run_comparison.py` uses the original attention kernel through the normal Triton
host/JIT and a matching pinned B060 wheel. `runtime_common.py`, `device_checks.py`
and `admission_gate.py` provide the original-host adapter, output checks and
fresh physical-host admission. The kernel is supplied by path and must match
the recorded original checksum; no replacement kernel or bitcode is injected.

Each of six alternating AB/BA rounds profiles both the selected CVPipeline
recipe and fresh AscendC on identical seeded Q/K/V. Each block uses exactly one
initialization and thirteen profiled-loop calls: wait1, warmup1, active10,
skip_first1, repeat1, with synchronization and a one-second sleep after each
call. Collection uses NPU activity only, Level1/PipeUtilization and l2_cache=False.
One additional untimed AscendC call prepares the shared O/M reference before
the cohort; it is explicitly recorded and never counted as a timing sample.

Every timed block receives complete finite O/M and unchanged-input checks after
the profiler exits. Its CSV must contain exactly ten operations in one exact
operator/device/core row. The collector preserves the actual aggregate and raw
CSV identity; it does not manufacture per-task samples. A fresh mapping canary
must establish the CSV device ID, which can differ from the physical NPU mask.

Example interface, after the experiment controller supplies the selected stack,
physical visibility, fresh private cache and live admission/stop files:

```sh
python run_comparison.py --config timing-config.json --recipes recipes.json \
  --qualified-result /path/to/passed-case/RESULT.json \
  --qualified-sha256 ACTUAL_SHA256 --case-id n16384-hd64 \
  --output /path/to/fresh/cohort --admission /path/to/ADMISSION.json \
  --stop-marker /path/to/STOP --csv-device-id ACTUAL_CSV_DEVICE_ID
```

The sealed timing configuration must name the same source/tool/wheel/runtime
pins and options as the case's complete three-seed and64-stress qualification.
Enabled-query profiles require the protected no-TCB-reuse policy. The default
core selections are AI_CORE; exact overrides must match actual mapping evidence.
An absent/partial qualification, changed native image, missing or ambiguous CSV,
wrong task count, nonfinite output or admission failure stops the cohort.

Use a separate sealed `mapping-calibration` configuration with
`--calibration-only` before a timing cohort. It collects one pair, verifies O/M
and input integrity, and records the actual CSV device ID for both operators
under the selected physical mask. Calibration durations are explicitly excluded
from performance adoption. Pass that verified ID with `--csv-device-id` to the
six-round cohort; do not assume that logical and physical device IDs coincide.

The CPU tests verify call counts, schedule/options, source selection and failure
handling with synthetic inputs. They provide no NPU correctness or timing credit.

`run_host_comparison.py` runs on the physical server host and reuses the numerical
controller's PID/start/boot admission monitor. It requires the complete admitted
25-case qualification, exact wheel/frontend and numerical-result hashes, and a
fresh physical/logical/CSV mapping before the eleven ordered timing cases.
Each positive result with less than2% target headroom schedules a separate
six-round repeat in a new process/output/cache. Cohorts are retained separately.
The child collector independently rechecks full numerical/stress/native evidence.

Keep `kernel_path` identical to qualification so source locations do not change
the qualified native image. The profiler packet and cache directories are private;
its kernel input is the same read-only original source used in qualification.

Set workspace_root and container in the sealed host configuration; the host controller has no experiment-path default. Qualification matches the full declared preset saved by Runtime; filtered launch kwargs are recorded separately.
