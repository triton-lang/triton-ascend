import copy
import subprocess
import sys
from pathlib import Path

import pytest

from evaluate_vv_contention import metric, verified_times
from cpu_resource_gate import sha256


def records():
    prediction = {
        "candidates": {
            "a0_f2": {
                "status": "STATIC_COST_ESTIMATED", "resource_checked": True, "factor": 2, "axis": 0, "binary":
                {"sha256": "binary-A"}
            }
        }
    }
    measurement = {
        "status": "COMPLETE", "reference": {"passed": True}, "input_shape_dtype_grid_match_static": True,
        "post_profile_correctness": {"passed": True}, "candidate_results": {
            "a0_f2": {
                "status": "MEASURED", "binary_sha256": "binary-A", "static_binary_loaded": True,
                "loaded_and_synchronized": True, "correctness": {"passed": True}, "last_launch":
                {"runtime_status": "APPLIED", "applied_factor": 2, "selected_axis":
                 0}, "samples_us": [4, 5, 6, 7, 100], "median_us": 6
            }
        }
    }
    return prediction, measurement


def test_evaluation_uses_five_sample_median_and_preserves_prediction():
    prediction, measurement = records()
    before = copy.deepcopy(prediction)
    assert verified_times(prediction, measurement) == {"a0_f2": 6}
    assert prediction == before


@pytest.mark.parametrize("field,value,error", [("binary_sha256", "other-binary", "different binary"),
                                               ("correctness", {"passed": False}, "correctness evidence"),
                                               ("last_launch", {"runtime_status": "FALLBACK"}, "not executed"),
                                               ("samples_us", [4, 5, 6], "five valid"),
                                               ("median_us", 5, "differs from samples")])
def test_invalid_device_evidence_is_not_counted_as_a_hit(field, value, error):
    prediction, measurement = records()
    measurement["candidate_results"]["a0_f2"][field] = value
    with pytest.raises(ValueError, match=error):
        verified_times(prediction, measurement)


def test_unsupported_or_unchecked_candidate_is_not_an_evaluated_selection():
    prediction, measurement = records()
    prediction["candidates"]["a0_f2"]["resource_checked"] = False
    assert verified_times(prediction, measurement) == {}


def test_only_one_available_choice_does_not_inflate_accuracy():
    rows = [{
        "multiple_measured_choices": False, "old_within_3pct": True, "new_within_3pct": True, "old_exact_best": True,
        "new_exact_best": True
    }]
    assert metric(rows) == {
        "cases": 0, "old_within_3pct": 0, "new_within_3pct": 0, "old_exact_best": 0, "new_exact_best": 0
    }


def test_full_output_fp64_fixture_schema_requires_reference_and_guards(tmp_path):
    prediction, measurement = records()
    source = tmp_path / "case.py"
    source.write_text("# reference fixture identity\n")
    measurement["reference"] = {
        "policy": "full_output_CPU_FP64_original_case", "source": str(source), "sha256": sha256(source)
    }
    accuracy = {"passed": True, "guards": True, "outputs": [{"passed": True}]}
    measurement["post_profile_correctness"] = accuracy
    measurement["candidate_results"]["a0_f2"]["correctness"] = copy.deepcopy(accuracy)
    assert verified_times(prediction, measurement) == {"a0_f2": 6}
    measurement["candidate_results"]["a0_f2"]["correctness"]["guards"] = False
    with pytest.raises(ValueError, match="full-output FP64"):
        verified_times(prediction, measurement)
    source.write_text("# changed reference\n")
    assert verified_times(prediction, measurement) == {}


def test_selection_imports_and_scoring_need_no_device_packages():
    code = '''
import sys
def deny_runtime(event, args):
    if event == 'import' and args[0].split('.')[0] in {'torch', 'torch_npu', 'triton', 'pynvml'}:
        raise RuntimeError('device runtime import')
    if event in ('subprocess.Popen', 'os.system'):
        raise RuntimeError('external process')
sys.addaudithook(deny_runtime)
import run_vv_contention_experiment
from backend_cost_model import ASSUMPTIONS
from vv_contention_model import DependencySchedule, schedule
s = DependencySchedule(ASSUMPTIONS)
s.dma('load', 256)
assert schedule([s.result()], 1)['score'] > 0
'''
    subprocess.run([sys.executable, "-c", code], cwd=Path(__file__).parent, check=True)
