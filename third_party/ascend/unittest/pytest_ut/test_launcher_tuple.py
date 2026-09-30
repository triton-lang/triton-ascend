import importlib.util
from collections import namedtuple
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest


def _load_driver():
    path = Path(__file__).resolve().parents[2] / "backend" / "driver.py"
    spec = importlib.util.spec_from_file_location("ascend_tuple_driver", path)
    driver = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(driver)
    return driver


driver = _load_driver()
PREFIX = tuple(object() for _ in range(driver._BASE_ARGS_FORMAT_LEN))
Pair = namedtuple("Pair", ["pointer", "size"])


@pytest.mark.parametrize("signature,values,expected", [
    ({0: ()}, ((), ), ()),
    ({0: ("i32", )}, ((7, ), ), (7, )),
    ({0: ("i32", ("fp32", "constexpr")), 1: "i32"}, ((7, (2.5, None)), 9), (7, 2.5, None, 9)),
    ({0: "constexpr", 1: ("i32", "i32")}, ((8, 8), (2, 3)), ((8, 8), 2, 3)),
    ({0: ("*fp32", "i32")}, (Pair(1024, 7), ), (1024, 7)),
])
def test_launcher_flattens_tuple_values(signature, values, expected):
    launch = driver.wrap_handle_tensordesc(lambda *args: args, signature)
    assert launch(*PREFIX, *values) == PREFIX + expected


def test_launcher_keeps_scalar_fast_path():
    launch = lambda *args: args
    assert driver.wrap_handle_tensordesc(launch, {0: "*fp32", 1: "i32"}) is launch


@pytest.mark.parametrize("nested", [False, True])
def test_launcher_expands_descriptor_at_tuple_leaf(nested):
    descriptor = SimpleNamespace(base=1024, shape=[8, 16], strides=[16, 1], padding="nan")
    signature = ("i32", "tensordesc<fp32[8,16]>") if nested else "tensordesc<fp32[8,16]>"
    value = (7, descriptor) if nested else descriptor
    launch = driver.wrap_handle_tensordesc(lambda *args: args, {0: signature, 1: "i32"})
    expected = (1024, 8, 16, 16, 1, True, 8, 16, 16, 1, 23)
    assert launch(*PREFIX, value, 23) == PREFIX + ((7, ) if nested else ()) + expected


def test_nested_descriptor_signature_matches_flat_signature():
    metadata = SimpleNamespace(target=driver.GPUTarget("npu", "Ascend910B3", 0), workspace_size=0, bs_task_type=0,
                               mix_mode="aiv", shared=0, compile_on_910_95=False, parallel_mode="", is_pure_simt=False,
                               debug=False, program_grid_mapping_applied=False, program_grid_transforms=None,
                               auto_blockify_enabled=False, ptsm_cap_authorized=False, row_coalescing_applied=False,
                               coalesce_factor=1, coalesce_axis=-1, coalesce_grid_ceil_div=False,
                               has_auto_blockify_blacklist_op=False)
    nested = {0: ("i32", ("tensordesc<fp32[8,16]>", "constexpr")), 1: "i32"}
    flat = {0: "i32", 1: "tensordesc<fp32[8,16]>", 2: "constexpr", 3: "i32"}
    with patch.object(driver, "NPUUtils") as utils, \
         patch.object(driver, "is_ffts_supported", return_value=True), \
         patch.object(driver, "force_disable_ffts", return_value=False), \
         patch.object(driver, "get_backend_func", return_value=""):
        utils.return_value.get_aivector_core_num.return_value = 40
        utils.return_value.get_aicore_num.return_value = 20
        utils.return_value.get_so_path.return_value = "/tmp/npu_utils.so"
        assert driver.make_launcher({}, nested, metadata) == driver.make_launcher({}, flat, metadata)
