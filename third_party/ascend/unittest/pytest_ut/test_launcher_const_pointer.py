import pytest

from triton.backends.ascend import launcher

pytestmark = pytest.mark.backend("native")


@pytest.mark.parametrize("dtype, expected", [("fp16", 1), ("bf16", 27), ("fp32", 0), ("i32", 3), ("u32", 8), ("i1", 12),
                                             ("fp8e4b15", -1)])
def test_launcher_reports_const_pointer_dtype(dtype, expected):
    types = launcher.argument_types({0: "i32", 1: f"*k{dtype}", 2: f"*{dtype}"})
    assert types == [(launcher.I32, -1), (launcher.POINTER, expected), (launcher.POINTER, expected)]


@pytest.mark.parametrize("signature", ["*unknown", "*kunknown"])
def test_launcher_rejects_unknown_pointer_dtype(signature):
    with pytest.raises(ValueError, match="Unsupported data type: unknown"):
        launcher.argument_types({0: signature})
