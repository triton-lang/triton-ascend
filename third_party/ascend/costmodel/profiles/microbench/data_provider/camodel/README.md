# CAModel data source workflow

This directory documents how to generate CAModel simulator data and how to
promote the parsed result into the microbenchmark profile data source used by
`../../ascend_davidv100_v1.json`.

## Directory layout

```text
data_provider/
  *.cce / *_host.cpp                 # runnable microbenchmark sources
  build_and_run.sh                   # build + on-board run helper
  build_commands.md                  # normal build/run commands
  camodel/
    README.md                        # this workflow
    camodel_experiment_matrix.json   # planned CAModel experiment coverage
    extract_camodel_system_cycle_profile.py
```

## 1. Build the probe

Run on the Ascend board from `data_provider/`:

```bash
cd /data/kaixin/triton-ascend/third_party/ascend/costmodel/profiles/microbench/data_provider
bash -lc './build_and_run.sh simt_memory'
```

`build_and_run.sh` expands to:

```bash
ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
  -I"$INC" "$name.cce" -o "$name.o"

g++ -O2 "${name}_host.cpp" -o "${name}_host" \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" \
  -lruntime -lascendcl
```

If the template headers are not in the default path:

```bash
INC=/data/kaixin/AscendNPU-IR/bishengir/lib/Template/include \
bash -lc './build_and_run.sh simt_memory'
```

## 2. Generate CAModel simulator output

After the host binary is built, run CAModel through `msopprof simulator`:

```bash
msopprof simulator --soc-version=Ascend950PR ./simt_memory_host
```

If the local CANN package uses davinci naming, use:

```bash
msopprof simulator --soc-version=dav-c310 ./simt_memory_host
```

The simulator creates an output directory similar to:

```text
OPPROF_YYYYMMDDHHMMSS_xxx/
  device0/
    ...
```

Keep the raw `OPPROF_*` directory as the primary evidence artifact.  The
profile JSON should not point only to an opaque number; it should point to the
source probe, host launcher, raw CAModel output, parser, and derived result.

## 3. Parse CAModel counts into SYS_CNT-domain rates

The parser expects a normalized JSON count file with this shape:

```json
{
  "per_unit": {
    "aiv0": {
      "span": {"delta": 1000},
      "group_counts": {
        "memory": 100,
        "float_alu": 200,
        "shuffle": 0,
        "predicate": 10,
        "control": 20,
        "int_alu": 30
      }
    }
  }
}
```

Then run:

```bash
python3 camodel/extract_camodel_system_cycle_profile.py parsed_camodel_counts.json \
  --simulator-clock-mhz 1650.0 \
  --sys-cnt-mhz 988.9 \
  --scope simt_memory
```

The parser emits JSON like:

```json
{
  "unit": "system_cycles",
  "scope": "simt_memory",
  "rates": {
    "memory": {
      "warp_instructions_per_system_cycle": 0.5
    }
  },
  "confidence": "low"
}
```

Important: these are workload-effective CAModel rates.  They are useful as
calibration evidence, but they are not bare hardware peak throughput or
isolated instruction latency.

## 4. Promote the result into the microbenchmark profile

When a CAModel-derived number is used by `ascend_davidv100_v1.json`, record the
complete provenance in the measurement `source` field.

Recommended source format:

```json
"source": "data_provider/simt_memory.cce; data_provider/simt_memory_host.cpp; data_provider/camodel/OPPROF_xxx; data_provider/camodel/parsed_simt_memory.json; data_provider/camodel/extract_camodel_system_cycle_profile.py; data_provider/build_commands.md"
```

The measurement should also state:

- `source_kind`: `camodel_simulator` or `isolated_microbenchmark_with_camodel`
- `cycle_domain`: normally `SYS_CNT` after parser conversion
- `confidence`: usually `low` or `medium`, unless independently validated
- `description`: whether the value is workload-effective or isolated

## 5. Validation checklist

Before treating a CAModel result as a data source:

1. Source `.cce` and `_host.cpp` are checked into `data_provider/`.
2. Exact build command is covered by `build_commands.md`.
3. Raw CAModel `OPPROF_*` artifact or a documented extracted subset is saved.
4. Parser command and output JSON are saved under `data_provider/camodel/`.
5. `ascend_davidv100_v1.json` `source` points to all relevant artifacts.
6. The profile description does not claim hardware peak if the measurement is
   only workload-effective.
