"""Experimental runtime-versioned SIMD column packing.

The original DSL body is unchanged. ExperimentalColumnPack.cpp replaces the
opt-in column rule in an ISOLATED triton-opt build. It additionally requires the
ascend.experimental_checked_full_rows function attribute, which is a HARNESS
assertion, not a compiler-derived safety certificate. Do not add it to arbitrary
user kernels. Normal TA builds do not compile this experimental source.

The native guard covers only kernel_full.py's exact affine memory contract:
positive B/D, power-of-two W in [2,64], disjoint write envelopes, unit channel
stride, non-overlapping batch rows, zero chunk offset, FP16/FP32 homogeneous
arguments, and integer/grid limits. It rejects unproved inputs. dispatch.py's
callables must bind exactly these inputs and grid; automatic JIT binding and
compiler certificate extraction remain TODO. No shape-only guard caching.

Tests check host storage legality, not compiler reorder legality for arbitrary
programs. W>4 device correctness/performance is pending at initial publication.
An optimized kernel must never be called directly on unproved inputs. This
prototype is not ready for merge or default enablement.
"""
