import json
import tempfile
import unittest
from pathlib import Path

from compare_fa_results import load_results


class CompareFaResultsTest(unittest.TestCase):

    def test_computes_speedup_against_named_reference(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            baseline = root / "baseline.json"
            cvsplit = root / "cvsplit.json"
            baseline.write_text(json.dumps({"mean_us": 2500}), encoding="utf-8")
            cvsplit.write_text(json.dumps({"mean_us": 1875}), encoding="utf-8")

            results = load_results([f"baseline={baseline}", f"cvsplit={cvsplit}"], "baseline")
            self.assertEqual(results[0]["speedup_vs_reference_pct"], 0)
            self.assertEqual(results[1]["speedup_vs_reference_pct"], 25)


if __name__ == "__main__":
    unittest.main()
