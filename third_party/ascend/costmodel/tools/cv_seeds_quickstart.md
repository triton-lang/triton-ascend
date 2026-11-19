# CV seed search: quickstart

Find the kernels where the cost model prefers an operation ordering other than
the compiler's own, then replay exactly those orderings on hardware.

> **Experimental, and you are switching it on.** The cost model ranks IR
> variants with a model that has no measured acceptance criterion and partly
> placeholder hardware numbers. A seed it likes is a candidate to measure, not a
> result: step 3 below is the point of the whole exercise, not a formality.
> Nothing searches unless one of the variables here is set.

Full reference: `docs/en/dynamic_cv_pipeline_cost_model.md`.

## 1. Search

Run your test scripts with a cache directory of their own, empty, so the table
holds only this run:

```bash
rm -rf ./tr_cache && mkdir ./tr_cache
TRITON_ALWAYS_COMPILE=1 \
TRITON_CACHE_DIR=./tr_cache \
TRITON_ASCEND_CV_VARIANTS=200 \
python your_test.py
```

`TRITON_ALWAYS_COMPILE=1` is **required**. None of these variables invalidate
the compilation cache, so without it a kernel compiled once is served from the
cache: nothing is recorded and no seed is applied.

`TRITON_ASCEND_CV_VARIANT_THREADS=8` scores that many orderings at once.

## 2. Collect

Each compilation records its decision in `<cache>/<hash>/<kernel>.json`. Turn
those into one table (pass the cache root, not a per-kernel directory):

```bash
python collect_cv_seeds.py --cache ./tr_cache
```

```
status         seed  kernel_name            hash          cost  second_bound  ub_peak  pipeline_stages  usable  total
seed_found     7     attn_fwd               9f2c4e1a77b3  28857  15479        180224   3                82      200
baseline_best  0     flex_attention_kernel  38dce1d004a1  4534   1656         786432   0                265     2000
```

| status | meaning |
|---|---|
| `seed_found` | an ordering beat the compiler's own — worth measuring |
| `baseline_best` | the search ran and nothing beat the untouched pipeline |
| `none_usable` | the search ran and no ordering could be scored |
| `ignored` | the CV pipeline does not apply to this kernel |
| `fallback` | the pipeline gave up; the seed is moot |
| `no_search` | no search was asked for |

Write the ones worth measuring to a file:

```bash
python collect_cv_seeds.py --cache ./tr_cache --only-found --format jsonl --out seeds.jsonl
```

`--only-found` keeps only `seed_found` rows, so an empty result means the cost
model preferred the baseline everywhere — run without it to see what was
recorded. `--format csv` and `--status NAME` are also available.

## 3. Replay

```bash
TRITON_ALWAYS_COMPILE=1 \
TRITON_CACHE_DIR=./tr_cache_replay \
TRITON_ASCEND_CV_SEED_TABLE=seeds.jsonl \
python your_test.py
```

Naming the file is what turns replay on; there is no mode variable. A kernel in
the table is compiled with its seed and **no search runs**, even if
`TRITON_ASCEND_CV_VARIANTS` is still set. A kernel not in the table compiles
exactly as it would with no table at all, and both hit and miss are logged:

```
[cv-seed-table] attn_fwd (hash 9f2c...) found in seeds.jsonl, mtime ...: compiling seed 7, no search
[cv-seed-table] gemm (hash cc10...) not found in seeds.jsonl (12 row(s), mtime ...); compiling as if no table had been given
```

Then compare timings against a baseline run — the same command with
`TRITON_ASCEND_CV_SEED_TABLE` unset and no `TRITON_ASCEND_CV_VARIANTS`.

## Nothing recorded?

```bash
python -c "import triton.backends.ascend.cv_seed_table as m; print('python side OK')"
python -c "from triton._C.libtriton import ascend; print('cpp side OK:', 'reorder_seed' in (ascend.passes.ttir.add_dynamic_cv_pipeline.__doc__ or ''))"
```

An `ImportError` or `False` means the installed wheel predates this feature.
Otherwise check `enable_dynamic_cv_pipeline` and `arch` in the kernel's JSON:
the pipeline only runs on `Ascend910_95*` / `Ascend950*`.
