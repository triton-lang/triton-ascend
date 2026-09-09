# task_0005 test harness

This directory owns frozen DSL fixtures and the standalone baseline protocol
for the Juhe graph-rule work. The fixture loader resolves files relative to
this directory, so a test does not import the task-book checkout or another
developer worktree.

The grid-specialization observation is deliberately different: it directly
imports the supplied `example1/kernel_ori.py` through a visible
`TASK0005_KERNEL_ORI_PATH` override (or the source-worktree sibling input),
then calls its existing `merge_split_states`, `indexer_norm_rope`, and
`indexer_logits` wrappers. It never copies those kernels or adds shape
parameters to their JIT signatures.

## Contracts frozen here

- merge_split: partial output [splits,8,64,192] BF16 and partial LSE
  [splits,8,64] FP32 for splits 2 and 4. The before grid is (tokens,heads);
  output uses rtol=1e-2, atol=4e-3, LSE uses rtol=atol=1e-3, and outputs must
  be finite.
- norm+RoPE: BF16 tokens 1, 16, 512, 4096, 8192, and 32768, Q
  [tokens,4096], K/Z [tokens,256], and cos/sin [4096,16]. Q/K have a
  two-BF16-peak-ULP maximum absolute error bound; Z is bitwise passthrough.
- indexer_logits: independent Step4-derived candidate Q [8,4,4,256],
  weights [8,4,4], K [64,1,256], with the original three-dimensional
  (Q tile,K tile,group) grid. Its precision oracle is rtol=1e-2, atol=5e-2
  before StaticProgramAxisFusionRule is evaluated.

The after DSL describes target structure only. In particular, its merge
BLOCK_D=head_dim, norm overlapping stores, and logits num_stages=2 are not
semantic or legality oracles.

## Test layers

From the source worktree, use:

    PYTHONPATH="$PWD/third_party/ascend/unittest${PYTHONPATH:+:$PYTHONPATH}" \
      /home/w00609825/miniconda/envs/pta210_ta36_0902_juhe_test_fallback/bin/python \
      -m pytest -q third_party/ascend/unittest/pytest_ut/test_task_0005_harness_unit.py

    TRITON_OPT="$(
      /home/w00609825/miniconda/envs/pta210_ta36_0902_juhe_test_fallback/bin/python \
      -c 'import pathlib, triton; print(pathlib.Path(triton.__file__).resolve().parent / "_C" / "triton-opt")'
    )"
    FILECHECK="$(
      /home/w00609825/miniconda/envs/pta210_ta36_0902_juhe_test_fallback/bin/python \
      -c 'import pathlib, triton; print(pathlib.Path(triton.__file__).resolve().parent / "FileCheck")'
    )"
    for fixture in third_party/ascend/unittest/Conversion/General/TritonToGraph/task-0005-*.mlir; do
      "$TRITON_OPT" "$fixture" --verify-each -graph-optimize='rule-mask=0' -o - |
        "$FILECHECK" "$fixture" || exit $?
    done

    TASK0005_ENABLE_E2E=1 TASK0005_NORM_TOKENS=1,16 \
      PYTHONPATH="$PWD/third_party/ascend/unittest${PYTHONPATH:+:$PYTHONPATH}" \
      /home/w00609825/miniconda/envs/pta210_ta36_0902_juhe_test_fallback/bin/python \
      -m pytest -q third_party/ascend/unittest/pytest_ut/test_task_0005_harness_e2e.py

    # This is a separate full-runtime layer.  It uses the unchanged original
    # wrappers for merge H=64/65, norm Q/K H=16/1 at tokens 1/16/512, and
    # logits groups=1/2/4.  Use a fresh task-owned TRITON_CACHE_DIR.
    TASK0005_ENABLE_E2E=1 TASK0005_ENABLE_GRID_E2E=1 \
      PYTHONPATH="$PWD/third_party/ascend/unittest${PYTHONPATH:+:$PYTHONPATH}" \
      /home/w00609825/miniconda/envs/pta210_ta36_0902_juhe_test_fallback/bin/python \
      -m pytest -q third_party/ascend/unittest/pytest_ut/test_task_0005_harness_e2e.py \
      -k unchanged_dsl_grid_specialization

## Paired baseline/control protocol

Use physical card 7 and the external BiSheng compiler. A baseline run enables
GraphOptimize; a control run disables it. Clear only the exact dedicated cache
before every independent invocation. The runner writes correctness, grid,
cache JSON, TTIR and TTAdapter golden files. Under msprof it emits five
warmups followed by at least 20 active launches; the ingest command selects
only the final 20 rows whose Name exactly equals the target kernel.

    export ASCEND_RT_VISIBLE_DEVICES=7
    export PATH=/home/w00609825/triton_workspace/.tasks/0904/0904_bisheng/bishengir/bin:$PATH
    export PYTHONPATH="$PWD/third_party/ascend/unittest${PYTHONPATH:+:$PYTHONPATH}"
    export TASK0005_ARTIFACT_ROOT=/home/w00609825/triton_workspace/.tasks/0905/juhe/develop/artifacts/task_0005
    export TASK0005_CACHE=/home/w00609825/triton_workspace/.tasks/0905/juhe/caches/task_0005/merge-r1-baseline
    mkdir -p "$TASK0005_CACHE"
    cache_root=$(realpath "$TASK0005_CACHE")
    case "$cache_root" in
      /|/home/w00609825|/home/w00609825/triton_workspace) exit 1 ;;
    esac
    find "$cache_root" -depth -mindepth 1 -delete
    TRITON_CACHE_DIR="$cache_root" msprof \
      --output="$TASK0005_ARTIFACT_ROOT/profiler/merge-r1-baseline" --export=on --type=text \
      /home/w00609825/miniconda/envs/pta210_ta36_0902_juhe_test_fallback/bin/python \
      -m task_0005_harness.run_baselines run \
      --operator merge_split --splits 2 --variant baseline \
      --artifact-root "$TASK0005_ARTIFACT_ROOT" --run-label merge-r1-baseline

    /home/w00609825/miniconda/envs/pta210_ta36_0902_juhe_test_fallback/bin/python \
      -m task_0005_harness.run_baselines ingest-profile \
      --artifact-root "$TASK0005_ARTIFACT_ROOT" \
      --run-record "$TASK0005_ARTIFACT_ROOT/runs/merge-r1-baseline/run_record.json" \
      --profiler-root "$TASK0005_ARTIFACT_ROOT/profiler/merge-r1-baseline" --round 1

Repeat in paired baseline/control order for rounds 1, 2 and 3. Use fresh cache
and profiler paths for every command; never select a latest CSV from a shared
directory.

## Grid-specialization baseline/off observation

The `grid-observation` command is intentionally not a performance run.  It
uses a clean cache and emits a reusable JSON contract for task_0002 v2.  The
baseline records that the current legacy JIT evaluates grid after cache lookup,
has no grid-specialized extent attr, publishes no transform metadata, and sends
the original grid to the generated launcher.  It also contains the enabled-bit
assertion contract: grid must move before cache lookup, enter cache identity,
be represented by a canonical original-extent attr, and reconcile with the
actual launcher grid.

    export TASK0005_KERNEL_ORI_PATH="$PWD/../../example1/kernel_ori.py"
    export TASK0005_CACHE=/home/w00609825/triton_workspace/.tasks/0905/juhe/caches/task_0005/grid-specialization-off
    mkdir -p "$TASK0005_CACHE"
    cache_root=$(realpath "$TASK0005_CACHE")
    case "$cache_root" in
      /|/home/w00609825|/home/w00609825/triton_workspace) exit 1 ;;
    esac
    find "$cache_root" -depth -mindepth 1 -delete
    TRITON_CACHE_DIR="$cache_root" \
      /home/w00609825/miniconda/envs/pta210_ta36_0902_juhe_test_fallback/bin/python \
      -m task_0005_harness.run_baselines grid-observation \
      --artifact-root /home/w00609825/triton_workspace/.tasks/0905/juhe/develop/artifacts/task_0005 \
      --label grid-specialization-off

The report is written to `grid_specialization/<label>.json` and copies its
exact cache JSON/TTIR/TTAdapter inputs under `golden/grid_specialization/`.

## Existing smoke regressions

- third_party/ascend/unittest/pytest_ut/test_graph_optimize.py
- third_party/ascend/unittest/pytest_ut/test_row_coalescing_e2e.py
- third_party/ascend/unittest/pytest_ut/test_task_0005_grid_specialization.py
- third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-load-store.mlir
- third_party/ascend/unittest/Conversion/General/TritonToGraph/graph-optimize-store-coalescing.mlir

These remain stage-smoke coverage. New rules must add their own positive,
negative and transactional tests rather than weakening these controls.
