"""Synthetic unit samples test the reporting rules; these are not NPU timings."""
import copy
import unittest
from paired_stats import PRODUCTION_SHAPES, summarize_matrix, summarize_paired_rounds


def cohort(shape=(128, 8, 16384, 64), cvp=100.0, ascend=95.0):
    case = {"shape": list(shape), "block_m": 128, "block_n": 256 if shape[-1] == 64 else 128,
            "dtype": "float16", "layout": "BNSD", "causal": False, "scale": 0.5}
    rows = [{"round_index": i, "order": ["cvpipeline", "ascendc"] if i % 2 == 0 else ["ascendc", "cvpipeline"],
             "measurement_mode": "hardware-operation-duration", "timing_unit": "us", "case": copy.deepcopy(case),
             "input_fingerprint": {"synthetic_unit_inputs": "same"},
             "arm_identities": {"cvpipeline": "synthetic-cvp", "ascendc": "synthetic-ascendc"},
             "cvpipeline_durations_us": [cvp] * 10, "ascendc_durations_us": [ascend] * 10} for i in range(6)]
    return {"case": case, "rounds": rows}


class PairedStatisticsTest(unittest.TestCase):
    def summarize(self, data):
        return summarize_paired_rounds(data["case"], data["rounds"])

    def test_six_paired_wins_and_median(self):
        result = self.summarize(cohort())
        self.assertTrue(result["strict_target_met"])
        self.assertEqual(result["paired_target_wins"], 6)
        self.assertEqual(result["ascendc_over_cvp_percent"], 95)

    def test_one_bad_pair_cannot_hide_behind_median(self):
        data = cohort(); data["rounds"][0]["cvpipeline_durations_us"] = [120] * 10
        result = self.summarize(data)
        self.assertTrue(result["median_target_met"])
        self.assertFalse(result["strict_target_met"])
        self.assertEqual(result["paired_target_wins"], 5)

    def test_equality_fails_strict_target(self):
        self.assertFalse(self.summarize(cohort(cvp=100, ascend=90))["strict_target_met"])

    def test_arithmetic_task_mean_not_task_median(self):
        data = cohort(ascend=100)
        for row in data["rounds"]:row["cvpipeline_durations_us"] = [80] * 9 + [400]
        result = self.summarize(data)
        self.assertEqual(result["median_us"]["cvpipeline"], 112)
        self.assertFalse(result["strict_target_met"])

    def test_small_target_margin_requires_independent_repeat(self):
        result = self.summarize(cohort(cvp=110, ascend=100))
        self.assertTrue(result["strict_target_met"])
        self.assertTrue(result["independent_repeat_needed"])

    def test_accepted_shape_does_not_forge_strict_pass(self):
        result = self.summarize(cohort(shape=(128, 8, 2048, 128), cvp=100, ascend=89.8))
        self.assertTrue(result["accepted_exception_case"])
        self.assertFalse(result["strict_target_met"])

    def test_missing_or_extra_samples_reject(self):
        for count in (9, 11):
            data = cohort();data["rounds"][0]["cvpipeline_durations_us"] = [100] * count
            with self.assertRaises(ValueError):self.summarize(data)

    def test_nonfinite_zero_negative_and_boolean_reject(self):
        for value in (float("nan"), float("inf"), 0, -1, True):
            data = cohort();data["rounds"][0]["ascendc_durations_us"][0] = value
            with self.assertRaises(ValueError):self.summarize(data)

    def test_changed_geometry_inputs_or_identity_reject(self):
        for field, value in [("case", {}), ("input_fingerprint", {"changed": True}),
                             ("arm_identities", {"cvpipeline": "different", "ascendc": "synthetic-ascendc"})]:
            data = cohort();data["rounds"][2][field] = value
            with self.assertRaises(ValueError):self.summarize(data)

    def test_simulator_host_time_and_wrong_units_reject(self):
        for field, value in [("measurement_mode", "simulator"), ("measurement_mode", "host-elapsed"), ("timing_unit", "ms")]:
            data = cohort();data["rounds"][0][field] = value
            with self.assertRaises(ValueError):self.summarize(data)

    def test_duplicate_unbalanced_or_missing_round_reject(self):
        for change in ("index", "order", "length"):
            data = cohort()
            if change == "index":data["rounds"][1]["round_index"] = 0
            elif change == "order":data["rounds"][1]["order"] = ["cvpipeline", "ascendc"]
            else:data["rounds"].pop()
            with self.assertRaises(ValueError):self.summarize(data)

    def test_matrix_requires_exact_eleven_and_sorts_large_first(self):
        inputs = [cohort(shape) for shape in reversed(PRODUCTION_SHAPES)]
        result = summarize_matrix(inputs)
        self.assertEqual(tuple(result["cases"][0]["case"]["shape"]), PRODUCTION_SHAPES[0])
        self.assertEqual(result["strict_target_count"], 11)
        with self.assertRaises(ValueError):summarize_matrix(inputs[:-1])
        inputs[0] = inputs[1]
        with self.assertRaises(ValueError):summarize_matrix(inputs)


if __name__ == "__main__":
    unittest.main()
