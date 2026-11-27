# Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
# SPDX-License-Identifier: MIT
"""Pure-Python TLA contracts: no Triton installation or NPU is required."""
import importlib.util
from pathlib import Path
import sys
import unittest

_PATH = Path(__file__).resolve().parents[2] / "language/tla/_type_contract.py"
_SPEC = importlib.util.spec_from_file_location("tla_type_contract_test", _PATH)
contracts = importlib.util.module_from_spec(_SPEC)
sys.modules[_SPEC.name] = contracts
_SPEC.loader.exec_module(contracts)

# Override pytest_ut's device-binding fixture while retaining unittest support.
try:
    import pytest

    @pytest.fixture(scope="module", autouse=True)
    def assign_npu():
        yield
except ImportError:
    pass


class TestTypeContract(unittest.TestCase):

    def test_signedness_and_float_formats_remain_distinct(self):
        contract = contracts.TypeContract(signatures=(("i32", ), ("f16", )))
        for dtype in ("i32", "f16"):
            contracts.check_type_contract("op", (dtype, ), contract)
        for dtype in ("u32", "bf16"):
            with self.subTest(dtype=dtype), self.assertRaisesRegex(ValueError, f"element type {dtype}"):
                contracts.check_type_contract("op", (dtype, ), contract)

    def test_parameter_constraints_apply_to_the_whole_type_signature(self):
        contract = contracts.TypeContract(signatures=(("f32", "i32"), ("f32", "f16")),
                                          parameters=(contracts.ParameterRule(("f32", "i32"), "round", ("R", "Z")), ))
        contracts.check_type_contract("cast", ("f32", "i32"), contract, options={"round": "R"})
        contracts.check_type_contract("cast", ("f32", "f16"), contract, options={"round": "O"})
        with self.assertRaisesRegex(ValueError, "f32 -> i32 on Ascend950.*round=O.*R, Z"):
            contracts.check_type_contract("cast", ("f32", "i32"), contract, options={"round": "O"}, target="Ascend950")
        with self.assertRaisesRegex(ValueError, "type combination f32 -> u32"):
            contracts.check_type_contract("cast", ("f32", "u32"), contract, options={"round": "R"})

    def test_all_applicable_parameters_are_checked(self):
        contract = contracts.TypeContract(parameters=(
            contracts.ParameterRule(("i32", ), "mode", ("ZEROING", )),
            contracts.ParameterRule(("i32", ), "dist", ("NORM_B32", )),
        ))
        with self.assertRaisesRegex(ValueError, "dist=NORM_B16"):
            contracts.check_type_contract("store", ("i32", ), contract, options={"mode": "ZEROING", "dist": "NORM_B16"})
        with self.assertRaisesRegex(ValueError, "requires dist"):
            contracts.check_type_contract("store", ("i32", ), contract, options={"mode": "ZEROING"})

    def test_empty_type_list_is_not_an_unspecified_contract(self):
        contracts.check_type_contract("op", ("f32", ), contracts.TypeContract())
        with self.assertRaisesRegex(ValueError, "supported: none"):
            contracts.check_type_contract("op", ("f32", ), contracts.TypeContract(signatures=()))


if __name__ == "__main__":
    unittest.main()
