"""Host-only dispatch contract tests; no compiler or device initialization."""
import unittest
from dispatch import CheckedKernel


class Kernel:
    def __init__(self, fn):
        self.fn = fn
        self.calls = []

    def __getitem__(self, grid):
        def launch(*args, **kwargs):
            self.calls.append((grid, args, kwargs))
            return self
        return launch


class DispatchTests(unittest.TestCase):
    def test_signatures_and_exact_forwarding(self):
        def small(renamed): pass
        def extra(first, additional, size, default=7): pass
        def reordered(size, buffer, *, stride=1): pass
        cases = [(small, (object(),), {}),
                 (extra, (object(), object()), {"size": 19}),
                 (reordered, (), {"buffer": object(), "size": 23, "stride": 2})]
        for fn, args, inputs in cases:
            for decision in (True, False):
                original, fast = Kernel(fn), Kernel(fn)
                seen = []
                def guard(values, grid, options):
                    seen.append((values, grid, dict(options)))
                    return decision
                checked = CheckedKernel(original, fast, guard)
                kwargs = {**inputs, "num_warps": 8, "compile_mode": "simd_simt_template"}
                chosen = checked[(3, 2)](*args, **kwargs)
                self.assertIs(chosen, fast if decision else original)
                self.assertEqual(chosen.calls, [((3, 2), args, kwargs)])
                bound = checked.signature.bind(*args, **inputs)
                bound.apply_defaults()
                self.assertEqual(seen, [(tuple(bound.arguments.values()), (3, 2),
                                         {"num_warps": 8, "compile_mode": "simd_simt_template"})])

    def test_callable_grid_evaluated_once_and_defaults_bound(self):
        def fn(a, count=7): pass
        original, fast = Kernel(fn), Kernel(fn)
        grids = []
        def grid(meta):
            grids.append(meta)
            return (meta["count"],)
        def guard(values, actual_grid, options):
            self.assertEqual(values, ("tensor", 7))
            self.assertEqual(actual_grid, (7,))
            return True
        CheckedKernel(original, fast, guard)[grid]("tensor")
        self.assertEqual(len(grids), 1)
        self.assertEqual(fast.calls[0][0], (7,))

    def test_reject_mismatched_variants_and_bad_binding(self):
        def fn(a): pass
        def different(a, b): pass
        with self.assertRaises(ValueError):
            CheckedKernel(Kernel(fn), Kernel(different))
        original, fast = Kernel(fn), Kernel(fn)
        checked = CheckedKernel(original, fast, lambda *args: True)
        for args, kwargs in [((), {}), ((1, 2), {}), ((1,), {"a": 2})]:
            with self.assertRaises(TypeError):
                checked[(1,)](*args, **kwargs)
        self.assertFalse(original.calls or fast.calls)

    def test_missing_failed_or_non_boolean_guard_falls_back(self):
        def fn(a): pass
        def failed(*args): raise OverflowError("range")
        for guard in (None, failed, lambda *args: 1, lambda *args: False):
            original, fast = Kernel(fn), Kernel(fn)
            self.assertIs(CheckedKernel(original, fast, guard)[(1,)](5), original)
            self.assertFalse(fast.calls)

    def test_launch_error_not_retried(self):
        def fn(a): pass
        class Broken(Kernel):
            def __getitem__(self, grid):
                def fail(*args, **kwargs): raise RuntimeError("launch failed")
                return fail
        original = Kernel(fn)
        with self.assertRaises(RuntimeError):
            CheckedKernel(original, Broken(fn), lambda *args: True)[(1,)](5)
        self.assertFalse(original.calls)


if __name__ == "__main__":
    unittest.main()
