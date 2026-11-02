# TLA buffer subviews and reshape

```python
tla.subview(src, offsets, sizes, strides=None)
```

Create a rank-preserving alias of a `tla.buffer`. The operation lowers directly
to `memref.subview`; it does not allocate or copy data. The result keeps the
source element type and memory space.

- `offsets` are scalar integers or scalar integer tensors, relative to the
  current source view.
- `sizes` and `strides` are positive compile-time integers with one entry per
  source dimension. The default strides are all one.
- Result strides are source strides multiplied by the supplied strides. A
  subview does not make a sliced matrix compact or change its row pitch.
- Statically out-of-bounds slices are rejected. The caller must keep dynamic
  offsets in bounds; the operation does not insert runtime bounds checks.
- Rank reduction and dynamic sizes/strides are not supported by this interface.

The existing `tla.view(buffer, shape, strides=..., offsets=...)` spelling remains
compatible and delegates to the same implementation. `tla.view(pointer, ...)`
continues to create a GM pointer view and is not a subview of a known allocation.

## Compact prefixes without a special-purpose view operation

`reshape` preserves the element count and supports a contiguous row-major
buffer or subview. It preserves a nonzero or dynamic source offset through
`memref.collapse_shape` / `memref.expand_shape`; it never resets that offset to
the allocation origin. Non-contiguous slices cannot be reshaped.

A compact attention tail can share a larger UB allocation:

```python
p = tla.alloc((8, 2064), tl.bfloat16, space="ub")
flat = tla.reshape(p, (16512,))
prefix = tla.subview(flat, offsets=(0,), sizes=(8320,))
tail = tla.reshape(prefix, (8, 1040))
```

`tail` has strides `(1040, 1)` and aliases the first 8320 elements of `p`.
In contrast, `tla.subview(p, (0, 0), (8, 1040))` retains strides `(2064, 1)`.
Both are aliases, but their rows address different elements.

IR, compilation and exact device checks are in
`third_party/ascend/unittest/tla_ut/vector/subview.py`. They cover FP16/BF16,
compact and pitched layouts, static/runtime/nested offsets, and write-through
aliasing with untouched allocation guards. Host argument/layout contracts are
in `view_frontend_contract.py` alongside the other TLA frontend contract tests.
