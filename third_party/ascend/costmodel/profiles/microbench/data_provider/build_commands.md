# Microbenchmark data provider

This directory contains the source CCE probes and host launchers used as
evidence for `../ascend_davidv100_v1.json`.

Run on the Ascend board, from this directory:

```bash
cd /data/kaixin/triton-ascend/third_party/ascend/costmodel/profiles/microbench/data_provider
bash -lc './build_and_run.sh tput concur2 meas simt_memory simt_gm_memory simt_shuffle'
```

If the template headers are not under the default path, override `INC`:

```bash
INC=/data/kaixin/AscendNPU-IR/bishengir/lib/Template/include \
bash -lc './build_and_run.sh tput concur2 meas simt_memory simt_gm_memory simt_shuffle'
```

The device build command used by `build_and_run.sh` is:

```bash
ccec -c -std=c++17 -O2 --cce-aicore-only --cce-aicore-arch=dav-c310 \
  -I"$INC" "$name.cce" -o "$name.o"
```

The host build command used by `build_and_run.sh` is:

```bash
g++ -O2 "${name}_host.cpp" -o "${name}_host" \
  -I"$ASCEND_TOOLKIT_HOME/x86_64-linux/pkg_inc" \
  -I"$ASCEND_TOOLKIT_HOME/include" \
  -L"$ASCEND_TOOLKIT_HOME/lib64" \
  -lruntime -lascendcl
```

## Source mapping

| JSON evidence source | Local files | Related measurements |
|---|---|---|
| `triton_cases/SIMT_Test/tput.cce` | `tput.cce`, `tput_host.cpp` | `simt.f32.add.throughput` |
| `triton_cases/SIMT_Test/tput.cce; concur2.cce` | `tput.cce`, `tput_host.cpp`, `concur2.cce`, `concur2_host.cpp` | `simd.f32.add.throughput`, `simd.f32.add.dependent_latency` |
| `triton_cases/SIMT_Test/meas.cce` | `meas.cce`, `meas_host.cpp` | `simt.setup.empty`, `simt.setup.empty_with_barrier` |
| `triton_cases/SIMT_Test/simt_memory_david_v100_20260725.csv` | `simt_memory.cce`, `simt_memory_host.cpp` | SIMT UB load/store throughput and bandwidth |
| `triton_cases/SIMT_Test/simt_gm_memory_david_v100_20260725.csv` | `simt_gm_memory.cce`, `simt_gm_memory_host.cpp` | SIMT GM load/store throughput and bandwidth |
| `triton_cases/SIMT_Test/simt_shuffle_david_v100_20260725.csv` | `simt_shuffle.cce`, `simt_shuffle_host.cpp` | SIMT shuffle throughput and dependent latency |
| `simt_transition_microbench_tail16_barrier_20260713.txt` | `transition.cce`, `transition_host.cpp`, `run_transition.remote.sh` | SIMT transition harness setup proxies |

`build_and_run.sh` sources `/data/kaixin/set_env.sh` or `/home/kaixin/set_env.sh`,
then builds `<name>.cce -> <name>.o` and `<name>_host.cpp -> <name>_host`.

## CAModel / msopprof simulator

The full workflow for generating CAModel data and promoting it into
`../ascend_davidv100_v1.json` as a data source is documented in
`camodel/README.md`.

The simulator command follows the Ascend devkit documentation pattern:

```bash
msopprof simulator --soc-version=Ascend950PR ./${name}_host
```

If the local CANN package uses the davinci-style SOC name, use the matching
target name instead, for example:

```bash
msopprof simulator --soc-version=dav-c310 ./${name}_host
```

The output directory is usually named `OPPROF_*`.  Per-instruction/per-unit
simulator data is under:

```text
OPPROF_*/device0/
```

The current parser for converted CAModel counts is:

```bash
python3 camodel/extract_camodel_system_cycle_profile.py parsed_camodel_counts.json \
  --simulator-clock-mhz 1650.0 \
  --sys-cnt-mhz 988.9 \
  --scope <experiment-name>
```

Files:

| Purpose | File |
|---|---|
| CAModel experiment plan | `camodel/camodel_experiment_matrix.json` |
| CAModel count-to-SYS_CNT parser | `camodel/extract_camodel_system_cycle_profile.py` |
