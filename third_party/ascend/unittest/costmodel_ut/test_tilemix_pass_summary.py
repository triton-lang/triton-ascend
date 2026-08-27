import importlib.util
import unittest
from pathlib import Path


def load_module(name, filename):
    root = Path(__file__).resolve().parents[4]
    path = root / "third_party" / "ascend" / "backend" / "runtime" / filename
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(module)
    return module


class TileMixPassSummaryTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.summary = load_module("tilemix_pass_summary_under_test", "tilemix_pass_summary.py")

    def test_capture_is_optional(self):
        self.assertFalse(self.summary.should_capture_tilemix_pass_summary(4, 4, {}))
        self.assertTrue(
            self.summary.should_capture_tilemix_pass_summary(4, 4, {"TRITON_CAPTURE_TILE_MIX_PASS_SUMMARY": "1"}))
        options = []
        self.assertTrue(
            self.summary.add_tilemix_pass_dump_options(options, 4, 2,
                                                       env={"TRITON_CAPTURE_TILE_MIX_PASS_SUMMARY": "true"}))
        self.assertEqual(len(options), 2)

    def test_real_pass_warning_and_ir_diff(self):
        output = """
// -----// IR Dump Before TileCubeVectorLoop (tile-cube-vector-loop) //----- //
scf.for %i = %c0 to %c8 step %c1 {
} {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
scf.for %j = %c0 to %c8 step %c1 {
} {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
// -----// IR Dump After TileCubeVectorLoop (tile-cube-vector-loop) //----- //
scf.for %i = %c0 to %c8 step %c1 {
} {hivm.loop_core_type = #hivm.tcore_type<CUBE>}
scf.for %j = %c0 to %c4 step %c1 {
  hivm.hir.wait_flag
} {hivm.loop_core_type = #hivm.tcore_type<VECTOR>}
loc("kernel.py":1:1): warning: Ignoring candidate cube loop trip count because it's suboptimal
"""
        result = self.summary.parse_tilemix_pass_summary(output, cube_loop=4, vector_loop=4)
        self.assertTrue(result["valid"])
        self.assertTrue(result["changed"])
        self.assertFalse(result["cube_applied"])
        self.assertEqual(result["cube_skip_reason"], "pass_rejected_suboptimal")
        self.assertTrue(result["vector_applied"])
        self.assertEqual(result["vector_segments"], 4)
        self.assertEqual(result["sync_ops_before"], 0)
        self.assertEqual(result["sync_ops_after"], 1)

    def test_missing_dump_fails_closed(self):
        result = self.summary.parse_tilemix_pass_summary("warning only", cube_loop=2, vector_loop=4)
        self.assertFalse(result["valid"])
        self.assertFalse(result["cube_applied"])
        self.assertEqual(result["vector_skip_reason"], "missing_pass_dump")

    def test_flatten_contract(self):
        flat = self.summary.summary_to_compile_params({
            "source": "pass",
            "valid": True,
            "cube_applied": False,
            "vector_applied": True,
            "vector_segments": 4,
            "cube_skip_reason": "fits",
            "vector_skip_reason": "none",
        })
        self.assertEqual(flat["tile_mix_summary_valid"], 1)
        self.assertEqual(flat["tile_mix_vector_segments"], 4)


if __name__ == "__main__":
    unittest.main()
