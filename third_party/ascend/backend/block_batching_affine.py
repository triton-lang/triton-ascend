# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
"""Integer arithmetic shared by the element and byte domain proofs."""
from dataclasses import dataclass


class GuardFailure(ValueError):
    """An unestablished proof. Guard entry points turn it into a refusal.

    Defined here rather than in block_batching_guard so that this leaf module
    can refuse too; block_batching_guard re-exports it for its own importers.
    """


@dataclass(frozen=True)
class Affine:
    offset: int
    # The v1 element domain is fixed at [grid0, grid1, grid2, extent]. Byte
    # domains are N-dimensional and must always pass their own width.
    coefficients: tuple = (0, 0, 0, 0)

    @property
    def uniform(self):
        return not any(self.coefficients)

    def add(self, other, sign=1):
        # zip would truncate to the shorter operand and silently drop the
        # remaining coefficients. Under-counting a dimension under-estimates
        # the proved address range, so refuse the proof instead.
        if len(self.coefficients) != len(other.coefficients):
            raise GuardFailure("AFFINE_DIMENSION_MISMATCH")
        return Affine(self.offset + sign * other.offset,
                      tuple(a + sign * b for a, b in zip(self.coefficients, other.coefficients)))

    def scale(self, scale):
        return Affine(self.offset * scale, tuple(c * scale for c in self.coefficients))

    def bounds(self, counts):
        if len(counts) != len(self.coefficients):
            raise GuardFailure("AFFINE_DIMENSION_MISMATCH")
        lo = hi = self.offset
        for coefficient, count in zip(self.coefficients, counts):
            delta = coefficient * (count - 1)
            lo += min(0, delta)
            hi += max(0, delta)
        return lo, hi
