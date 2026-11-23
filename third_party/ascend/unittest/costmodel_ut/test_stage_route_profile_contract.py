"""Contract tests for the 950PR Stage Route CostModel profile.

These deliberately avoid importing ``triton.backends.ascend.compiler`` (whose
relative imports need an installed Triton package).  The default-profile
function is extracted with ``ast`` and executed in isolation, so the test runs
with a plain ``python -m unittest`` on a source checkout.
"""

import ast
import json
import unittest
from pathlib import Path

ASCEND = Path(__file__).resolve().parents[2]
PROFILES = ASCEND / "costmodel" / "profiles" / "simd_simt"
COMPILER = ASCEND / "backend" / "compiler.py"

CALIBRATION_KEYS = (
    "completion_inclusive_groups_cover_boundary",
    "completion_inclusive_minimum_stage_count",
    "reduction_resident_tile_max_fraction",
    "simt_whole_kernel_completion_scale",
    "simt_whole_kernel_factor_limit",
)


def _default_profile_name(arch):
    tree = ast.parse(COMPILER.read_text())
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "_default_auto_simt_profile_name":
            namespace = {}
            exec(compile(ast.Module(body=[node], type_ignores=[]), str(COMPILER), "exec"), namespace)
            return namespace["_default_auto_simt_profile_name"](arch)
    raise AssertionError("_default_auto_simt_profile_name not found in compiler.py")


class StageRouteProfileContractTest(unittest.TestCase):

    def test_default_950pr_profile_is_calibrated_v154(self):
        name = _default_profile_name("Ascend950PR")
        self.assertEqual(name, "simd_simt/ascend_950pr_simd_simt_v154.json")
        self.assertTrue((PROFILES / Path(name).name).is_file())

    def test_other_arch_defaults_are_unchanged(self):
        self.assertEqual(_default_profile_name("Ascend950DT"), "simd_simt/ascend_950dt_simd_simt_v1.json")
        self.assertEqual(_default_profile_name("david_v100"), "simd_simt/david_v100_simd_simt_v1.json")

    def test_v154_profile_carries_stage_route_calibration(self):
        profile = json.loads((PROFILES / "ascend_950pr_simd_simt_v154.json").read_text())
        timing = profile["kernel_timing"]
        self.assertTrue(timing["completion_inclusive_groups_cover_boundary"])
        self.assertEqual(timing["completion_inclusive_minimum_stage_count"], 12)
        self.assertAlmostEqual(timing["reduction_resident_tile_max_fraction"], 0.6)
        self.assertAlmostEqual(timing["simt_whole_kernel_completion_scale"], 0.44)
        self.assertEqual(timing["simt_whole_kernel_factor_limit"], 4)
        for mode in ("simd", "simt"):
            self.assertAlmostEqual(profile[mode]["stage_resources"]["scalar_memory"]["in_kernel_transfer_scale"], 0.45)

    def test_v1_profile_keeps_new_calibration_disabled(self):
        timing = json.loads((PROFILES / "ascend_950pr_simd_simt_v1.json").read_text()).get("kernel_timing", {})
        for key in CALIBRATION_KEYS:
            self.assertNotIn(key, timing)


if __name__ == "__main__":
    unittest.main()
