# CV-split lane-safe UB reuse

## Scope

This change has two related, but distinct, parts:

1. A generic Triton buffer-language operation, `reinterpret_view`, creates a
   static strided logical view of an existing local allocation.
2. CVSplitScheduling recognizes the supported four-lane QK/P/PV handoff
   structure and may reuse one lane-local UB allocation for that lane's QK and
   PV results while preserving cross-iteration ownership.

The manually scheduled Flash Attention U4 experiment is not evidence that the
automatic CV-split pass produced the schedule. The manual Python kernel writes
the Cube/Vector scopes, unrolling, transfers, and synchronization explicitly.
It requires the generic `reinterpret_view` frontend/compiler support, but it
runs with both automatic CVSplitScheduling and DynamicCVPipeline disabled.

## Why a reinterpret view is required

For the validated `BLOCK_M=128`, `BLOCK_N=128`, `HEAD_DIM=64` case, each
unrolled lane owns one physical UB allocation for a `64x128xf32` QK result. The
later PV result needs only `64x64xf32`. Allocating a separate PV buffer raises
peak UB use, while treating the smaller tensor as if it had the QK row pitch
gives the wrong physical layout.

The manual kernel therefore expresses:

```python
lane_ub = bl.alloc(tl.float32, [64, 128], al.ascend_address_space.UB)
pv_ub = bl.reinterpret_view(lane_ub, [64, 64], [64, 1])
```

`pv_ub` aliases `lane_ub`. It does not allocate storage and does not copy data.
The compiler lowers the operation to `memref.reinterpret_cast` with the same
allocation root.

## Safety contract

`reinterpret_view` accepts only a direct, statically shaped local allocation
with identity layout. Shape, strides, and offset must be compile-time positive
integers (the offset may be zero), and the maximum referenced element must fit
inside the source allocation. The frontend and C++ builder both enforce the
footprint check. These restrictions prevent the operation from silently
extending an allocation or hiding an unknown alias chain.

The operation changes logical indexing only. It does not prove temporal
non-overlap. The caller or scheduling pass must separately prove that the old
QK contents are dead before the PV view is written.

## Automatic CV-split handling

The automatic pass uses a conservative matcher for the supported four-lane
QK/P/PV structure. It verifies the lane relationship, creates one UB allocation
per lane, gives the QK and PV transfers views of that same allocation root, and
adds an ownership token around the reuse interval. If the structure cannot be
proved, the pass falls back to independent transfer buffers instead of forcing
the alias.

This ownership token is different from the twelve data-handoff flags used by
U4. The handoff flags establish Cube-to-Vector and Vector-to-Cube producer and
consumer ordering. The ownership token prevents a following loop iteration
from overwriting a lane allocation before the current iteration's final Vector
consumer has finished with it.

## Manual experiment boundary

The separate manual checkpoint uses:

```python
enable_cv_split_scheduling = False
enable_dynamic_cv_pipeline = False
```

Consequently:

- the Python kernel is responsible for the manual U4 schedule;
- the compiler is responsible for validating and lowering the buffer view and
  the explicitly written scopes, transfers, and synchronization;
- the automatic CVSplitScheduling ownership/reuse matcher is not exercised by
  that manual performance comparison.

A stock compiler without this change rejects the manual kernel because
`triton.extension.buffer.language` has no `reinterpret_view` operation. The
experiment must therefore pin a compiler build containing this change.

## Validation coverage

`python/unittest/pytest_ut/test_reinterpret_view.py` covers valid lowering and
rejects invalid rank, shape, stride, offset, footprint, non-allocation source,
and non-identity source-layout cases. The CVSplitScheduling MLIR tests cover
the lane-local allocation/view structure, ownership synchronization, fallback,
and rollback behavior.

The recorded hardware experiment separately validates the manually authored
kernel's correctness and device-task timing. Because that kernel is manually
rewritten, its timing is a schedule proof of concept, not an automatic-pass
performance result.
