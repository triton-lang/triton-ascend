"""Signature-independent experimental dispatch; not a production TA launcher.

A trusted producer must bind a guard to the exact two variants. The guard must
cover memory safety AND transformation preconditions. This module does not
infer that proof, and never treats argument names as proof.
"""
import inspect
from types import MappingProxyType


class CheckedKernel:
    """Bundle same-signature variants with their trusted launch guard.

    guard(values, grid, options) receives values in formal parameter order,
    including constexprs and defaults. No guard means original-only execution.
    Guard producers must regenerate the guard when the kernel signature changes.
    """

    def __init__(self, original, fast, guard=None):
        self.original, self.fast, self.guard = original, fast, guard
        self.signature = inspect.signature(original.fn)
        if self.signature != inspect.signature(fast.fn):
            raise ValueError("Checked variants must have identical signatures")
        if any(p.kind in (p.VAR_POSITIONAL, p.VAR_KEYWORD)
               for p in self.signature.parameters.values()):
            raise ValueError("Variadic kernel signatures are unsupported")

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            parameters = self.signature.parameters
            inputs = {k: v for k, v in kwargs.items() if k in parameters}
            options = {k: v for k, v in kwargs.items() if k not in parameters}
            bound = self.signature.bind(*args, **inputs)
            bound.apply_defaults()
            # Resolve once: the guard and the selected launch see the same grid.
            resolved = grid({**bound.arguments, **options}) if callable(grid) else grid
            resolved = tuple(resolved)
            values = tuple(bound.arguments[name] for name in parameters)
            valid = False
            if self.guard is not None:
                try:
                    valid = self.guard(values, resolved, MappingProxyType(options)) is True
                except (TypeError, ValueError, OverflowError):
                    pass  # Unrepresentable checks cannot authorize widening.
            kernel = self.fast if valid else self.original
            return kernel[resolved](*args, **kwargs)
        return launch
