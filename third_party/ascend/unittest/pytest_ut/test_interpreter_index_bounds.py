import numpy as np
import pytest

import triton.language as tl
from triton.backends.ascend import ascend_interpreter
from triton.runtime.interpreter import TensorHandle

pytestmark = pytest.mark.backend("native")


@pytest.mark.parametrize("operation", ["gather", "scatter", "index_put"])
def test_masked_index_lanes(monkeypatch, operation):
    builder = ascend_interpreter.AscendInterpreterBuilder()
    storage = np.zeros(5, dtype=np.int32)
    ptr = TensorHandle(np.array([storage.ctypes.data], dtype=np.uint64), tl.pointer_type(tl.int32))
    indices = TensorHandle(np.array([-1, 0, 2, 3], dtype=np.int32), tl.int32)
    values = TensorHandle(np.arange(4, dtype=np.int32), tl.int32)
    expected_mask = [False, True, True, False]

    def check_load(addresses, mask, other, dtype):
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(other, -7)
        return other

    def check_store(addresses, data, mask):
        np.testing.assert_array_equal(mask, expected_mask)
        np.testing.assert_array_equal(data, values.data)

    if operation == "gather":
        monkeypatch.setattr(ascend_interpreter._interpreter, "load", check_load)
        builder.create_gather_out_to_ub(ptr, indices, 3, 0, [1], [4], [1], other=-7)
    else:
        monkeypatch.setattr(ascend_interpreter._interpreter, "store", check_store)
        if operation == "scatter":
            builder.create_scatter_ub_to_out(ptr, values, indices, 3, 0, [1], [4], [1])
        else:
            builder.create_index_put(ptr, indices, values, 0, 3, [4], [1], [1])


@pytest.mark.parametrize("index", [-1, 2])
def test_index_select_rejects_invalid_row(monkeypatch, index):
    builder = ascend_interpreter.AscendInterpreterBuilder()
    storage = np.zeros((2, 2), dtype=np.int32)
    ptr = TensorHandle(np.array([storage.ctypes.data], dtype=np.uint64), tl.pointer_type(tl.int32))
    indices = TensorHandle(np.array([index], dtype=np.int32), tl.int32)
    monkeypatch.setattr(ascend_interpreter._interpreter, "load", lambda *args: pytest.fail("unexpected load"))
    with pytest.raises(IndexError, match="out of bounds"):
        builder.create_index_select_simd(ptr, indices, 0, [2, 2], [-1, 0], [-1, 2], [1, 2])
