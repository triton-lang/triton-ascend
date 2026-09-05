"""Check calibration units; this does not establish hardware accuracy."""

import json
from pathlib import Path
import unittest


class ScanProfileCalibrationTest(unittest.TestCase):
    def test_recorded_ratio_accounts_for_base_shuffle_rates(self):
        ascend = Path(__file__).resolve().parents[2]
        path = ascend / "costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json"
        profile = json.loads(path.read_text())
        measurements = json.loads(
            (path.parent / profile["microbenchmark_profile"]).read_text()
        )["measurements"]
        simd_rate = measurements["simd.vector_width_bits"]["value"] / 32
        simt_rate = (measurements["simt.warp_size"]["value"] *
                     measurements["simt.shuffle.throughput"]["value"])
        simd_factor = profile["simd"]["stage_resources"]["prefix_scan"]["dependency_factor"]
        simt_factor = profile["simt"]["stage_resources"]["prefix_scan"]["dependency_factor"]
        # Equal lane-step workloads cancel. Exclude scalar/setup/issue terms.
        ratio = (simd_factor / simd_rate) / (simt_factor / simt_rate)
        self.assertAlmostEqual(ratio, 143.398 / 8.267, places=8)


if __name__ == "__main__":
    unittest.main()
