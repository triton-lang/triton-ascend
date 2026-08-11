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
