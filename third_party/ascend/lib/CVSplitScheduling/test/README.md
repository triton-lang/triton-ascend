# CVSplitScheduling Flash-Attention validation

`test_fa_accuracy.py` is the single kernel source for all comparisons. The
variants change only compiler policy switches:

| Variant | DynamicCVPipeline | CVSplitScheduling |
| --- | --- | --- |
| `default` | target default | target default |
| `baseline` | off | off |
| `dcvp` | on | off |
| `cvsplit` | off | on, unroll 4 |
| `auto` | on (fallback) | on (first attempt) |

The explicit `baseline` is important because an empty launch option set uses
the A5 production default (`auto`) and is therefore not a plain baseline.

The names used in older experiments map to this table as follows:

- **pass off / SSBUF:** `baseline` (both high-level CV schedulers disabled);
- **SSBUF pipeline:** `dcvp` (legacy DynamicCVPipeline enabled);
- **CVSS:** `cvsplit` (CVSplitScheduling explicitly enabled);
- **pass on:** `auto` (CVSS first, SSBUF/DCVP only as transactional fallback);
- **production invocation:** `default` (no policy keyword arguments).

## Pinned five-way reproduction

The one-command reproduction checks the exact embedded AscendNPU-IR revision
and the SHA256 of `bishengir-compile`, validates full output for every policy,
then profiles the same full BM128 grid for `baseline`, `dcvp`, `cvsplit`,
`auto`, and `default`:

```bash
source /path/to/ascend-toolkit/set_env.sh
export ASCEND_RT_VISIBLE_DEVICES=6
PYTHON=/path/to/python MSPROF=/path/to/msprof \
  ./test/reproduce_fa_matrix.sh /tmp/fa-repro
```

`fa_repro_lock.json` records the compiler implementation commit, embedded
AscendNPU-IR commit, BishengIR revision and binary SHA256, hardware, shape,
warmup count, and measured iteration count. `comparison.csv` is the compact
result; the output directory also contains raw msprof data and accuracy logs.

The clean hardware run on 2026-08-11 produced:

| Policy | Mean AI_CORE time (us) | Speedup vs pass off |
| --- | ---: | ---: |
| pass off / `baseline` | 2539.363 | - |
| legacy SSBUF pipeline / `dcvp` | 2387.597 | 5.98% |
| explicit CVSS / `cvsplit` | 1882.369 | 25.87% |
| pass on / `auto` | 1882.399 | 25.87% |
| production invocation / `default` | 1882.416 | 25.87% |

Every row used 13 captured launches, discarded the first three, and averaged
the remaining ten. Full-output correctness passed for every policy with
maximum absolute error `0.00003052`. The machine-readable result is
`fa_reference_result_20260811.csv`.

## Correctness

Run all query tiles with one batch and one head so every output can be checked
against the PyTorch reference:

```bash
for block_m in 64 128; do
  for variant in default baseline dcvp cvsplit auto; do
    python test/profile_fa.py \
      --variant "$variant" --sequence-length 1024 \
      --batch-size 1 --num-heads 1 --block-m "$block_m" --block-n 128 \
      --active-blocks 0 --warmup 1 --iterations 1
  done
done
```

## Hardware performance

Use the same full production grid for every variant.  `msprof` records three
warmups followed by ten measured launches:

```bash
msprof --output=./msprof_bm64_cvsplit --task-time=on --ai-core=on \
  --aic-mode=task-based --application="python \
  test/profile_fa.py --variant cvsplit --sequence-length 1024 \
  --batch-size 128 --num-heads 8 --block-m 64 --block-n 128 \
  --active-blocks 0 --warmup 3 --iterations 10 --skip-accuracy"

python test/summarize_msprof.py ./msprof_bm64_cvsplit --warmup 3
```

Repeat the command without changing shape, grid, inputs, warmup, or iteration
count for `default`, `baseline`, `dcvp`, and `auto`. Use
`auto --unroll-factor 3` to prove
that CV rejection falls back to DCVP with correct output. A performance result
is reportable only when:

1. all variants use this same Python kernel and input distribution;
2. all variants run the full query grid on the same device;
3. profiler `AI_CORE` task time is used, not wall-clock time;
4. the same number of warmups is discarded;
5. the corresponding full-output correctness runs pass.

The parser is covered by a host-only test:

```bash
python test/test_summarize_msprof.py
```
