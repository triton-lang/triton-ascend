"""CPU-only checks: python -m unittest test_native -v (build guard first)."""
import random
import unittest
from unittest.mock import patch
import torch
import native_guard
import dispatch


class GuardTests(unittest.TestCase):
    def test_storage_and_width_policy(self):
        rng = random.Random(42)
        for i in range(1000):
            D, N = rng.choice([1, 3, 17, 127, 257]), rng.choice([1, 3, 5])
            W, BD = rng.choice([2, 4, 8, 16, 32, 64]), rng.choice([8, 32, 128, 256])
            dtype = rng.choice([torch.float16, torch.float32])
            tensors = [torch.empty(n, dtype=dtype) for n in (N*D, N*D*W, D*W, D, N*D)]
            kind = i % 5
            if kind == 1:
                tensors[1] = tensors[1][1:]  # offset makes the envelope too short
            if kind == 2:
                tensors[2] = tensors[1]  # weight aliases writable cache
            if kind == 3:
                tensors[4] = tensors[0]  # output aliases input
            if kind == 4:
                storage = torch.empty(sum(t.numel() for t in tensors), dtype=dtype)
                tensors = list(storage.split([t.numel() for t in tensors]))
            args = (*tensors, D, N, W, BD, (D+BD-1)//BD, N, D, D)
            allowed = W <= 16 or (W == 32 and dtype == torch.float32 and BD <= 128)
            self.assertEqual(native_guard.check_cpu_test(*args), kind in (0, 4) and allowed)
            self.assertFalse(native_guard.check(*args))  # CPU cannot authorize NPU launch

    def test_invalid_launch(self):
        tensors = [torch.empty(n) for n in (16, 64, 32, 8, 16)]
        scalars = [8, 2, 4, 8, 1, 2, 8, 8]
        for index, value in [(0, 0), (1, 0), (2, 3), (2, 64), (3, 3),
                             (4, 2), (5, 3), (6, 7), (7, 7), (0, 2**40)]:
            changed = scalars.copy()
            changed[index] = value
            self.assertFalse(native_guard.check_cpu_test(*tensors, *changed))

    def test_dispatch_binds_checked_arguments(self):
        objects = [object() for _ in range(5)]
        calls = []

        class Kernel:
            def __init__(self, name):
                self.name = name
            def __getitem__(self, grid):
                return lambda *args, **kwargs: calls.append((self.name, grid, args, kwargs))

        for allowed in (True, False):
            with patch.object(dispatch.native_guard, "check", return_value=allowed) as check:
                dispatch.launch_checked(Kernel("fast"), Kernel("original"), *objects,
                                        D=17, N=3, W=4, BD=8, stride_x_n=20, stride_y_n=24)
                x, cache, y, weight, bias = objects
                check.assert_called_once_with(x, cache, weight, bias, y, 17, 3, 4, 8, 3, 3, 20, 24)
                name, grid, args, meta = calls[-1]
                self.assertEqual(name, "fast" if allowed else "original")
                self.assertEqual(grid, (3, 3))
                self.assertEqual(args, (*objects, 20, 1, 24, 1))
                self.assertEqual((meta["W"], meta["BD"], meta["CHUNK_OFFSET"]), (4, 8, 0))
        with patch.object(dispatch, "native_guard", None):
            dispatch.launch_checked(Kernel("fast"), Kernel("original"), *objects,
                                    D=17, N=3, W=4, BD=8, stride_x_n=20, stride_y_n=24)
        self.assertEqual(calls[-1][0], "original")
        with patch.object(dispatch, "native_guard", None):
            dispatch.launch_checked(Kernel("fast"), Kernel("original"), *objects[:3], None, None,
                                    D=17, N=3, W=4, BD=8, stride_x_n=20, stride_y_n=24)
        self.assertFalse(calls[-1][3]["HAS_WEIGHT"])
        self.assertFalse(calls[-1][3]["HAS_BIAS"])
