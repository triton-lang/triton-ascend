import csv
import tempfile
import unittest
from pathlib import Path

from summarize_msprof import summarize


class SummarizeMsprofTest(unittest.TestCase):
    def test_rejects_negative_warmup(self):
        with self.assertRaisesRegex(ValueError, "warmup must be nonnegative"):
            summarize("unused.csv", warmup=-1)

    def test_discards_warmup_and_ignores_non_kernel_tasks(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "task_time_test.csv"
            with path.open("w", newline="", encoding="utf-8") as stream:
                writer = csv.DictWriter(
                    stream,
                    fieldnames=("kernel_name", "kernel_type", "task_time(us)"))
                writer.writeheader()
                writer.writerow({
                    "kernel_name": "N/A", "kernel_type": "PROFILING_ENABLE",
                    "task_time(us)": "1"})
                for value in (99, 10, 12, 14):
                    writer.writerow({
                        "kernel_name": "_attn_fwd_0", "kernel_type": "AI_CORE",
                        "task_time(us)": str(value)})

            result = summarize(path, warmup=1)
            self.assertEqual(result["captured_launches"], 4)
            self.assertEqual(result["measured_launches"], 3)
            self.assertEqual(result["mean_us"], 12)
            self.assertEqual(result["median_us"], 12)
            self.assertEqual(result["min_us"], 10)
            self.assertEqual(result["max_us"], 14)


if __name__ == "__main__":
    unittest.main()
