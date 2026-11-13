import json
import math
from pathlib import Path
import subprocess
import sys

import pytest

import backend_cost_model as backend
import run_vv_skill_selector
from cpu_resource_gate import sha256
from run_vv_contention_experiment import verify_artifacts
from run_vv_skill_selector import select
from vv_candidate_plan import plan_candidates
from vv_skill_model import InstructionModel, SkillVectorSchedule, ini_rates, DATA


@pytest.fixture(scope="module")
def model():
    return InstructionModel()


def candidate(axis=-1, factor=1, score=10., **extra):
    return dict(axis=axis, factor=factor, score=score, status="STATIC_COST_ESTIMATED", resource_checked=True,
                binary={"sha256": "binary"}, **extra)


def test_three_candidates_choose_axis_from_grid_not_names_or_scores():
    choices = {
        "unmerged": candidate(), "a0_f2": candidate(0, 2, .01), "a1_f2": candidate(1, 2, 900), "a1_f4":
        candidate(1, 4, 900), "a1_f8": candidate(1, 8, 0.)
    }
    plan = plan_candidates([2, 16, 1], choices)
    assert plan["keys"] == ["unmerged", "a1_f2", "a1_f4"]
    assert plan["omitted"] == ["a0_f2", "a1_f8"]


def test_tail_padding_and_small_grid_do_not_invent_candidates():
    choices = {"unmerged": candidate(), "a0_f4": candidate(0, 4), "a1_f4": candidate(1, 4)}
    assert plan_candidates([9, 8, 1], choices)["keys"] == ["unmerged", "a1_f4"]
    assert plan_candidates([2, 1, 1], choices)["keys"] == ["unmerged"]
    with pytest.raises(ValueError):
        plan_candidates([0, 1, 1], choices)


def test_selection_ties_and_unsafe_scores():
    choices = {"unmerged": candidate(), "a0_f2": candidate(0, 2)}
    assert select(choices)["selected"] == "unmerged"
    choices["a0_f2"]["score"] = 9.
    assert select(choices)["selected"] == "a0_f2"
    for score in (math.inf, math.nan, -1., True):
        choices["a0_f2"]["score"] = score
        assert select(choices)["selected"] == "unmerged"
        assert not select(choices)["model_complete"]


def test_unknown_model_falls_back_only_to_resource_verified_baseline():
    choices = {"unmerged": candidate(), "a0_f2": candidate(0, 2, .01)}
    choices["a0_f2"]["status"] = "STATIC_ANALYSIS_UNSUPPORTED"
    result = select(choices)
    assert result["selected"] == "unmerged" and not result["model_complete"]
    choices["unmerged"]["resource_checked"] = False
    assert select(choices)["selected"] is None
    choices["unmerged"]["resource_checked"] = True
    choices["a0_f2"].update(status="UB_REJECTED", resource_checked=False)
    assert select(choices)["model_complete"]


def test_native_width_dtype_and_single_issue(model):
    inst, rate = model.instruction("llvm.hivm.vmul.s.x.v64f32")
    assert inst["elements"] == 64 and rate["issue_slots"] == 2
    _, rate = model.instruction("llvm.hivm.vcadd.s.x.v64f32.v64f32")
    assert rate["issue_slots"] == 1
    inst, _ = model.instruction("llvm.hivm.vcvtff.bf162f32.x")
    assert inst["elements"] == 64 and inst["src_dtype"] == "bf16"
    inst, _ = model.instruction("llvm.hivm.vcmp.v256i1.v64f32")
    assert inst["dtype"] == "fp32"
    with pytest.raises(backend.Unsupported):
        model.instruction("llvm.hivm.vexp.x.v64i32")


def test_unknown_native_instruction_retains_explicit_nonzero_cost(model):
    s = SkillVectorSchedule({}, backend.ASSUMPTIONS, model)
    s.add("llvm.hivm.pge.b32", "call @predicate()", "%x")
    assert s.finish > 0 and s.counts["cost_source:retained_fallback"] == 1


def test_dependencies_and_transcendental_overlap(model):
    table = {("VFMulOp", "F32", "F32"): (1., 5., 2.)}
    seq = SkillVectorSchedule(table, backend.ASSUMPTIONS, model)
    seq.add("llvm.hivm.vmul.s.x.v64f32", "call @mul(%input)", "%a")
    seq.add("llvm.hivm.vmul.s.x.v64f32", "call @mul(%a)", "%b")
    assert seq.finish == 10.
    independent = SkillVectorSchedule(table, backend.ASSUMPTIONS, model)
    independent.add("llvm.hivm.vmul.s.x.v64f32", "call @mul(%input)", "%a")
    independent.add("llvm.hivm.vmul.s.x.v64f32", "call @mul(%other)", "%b")
    assert independent.finish < seq.finish
    overlap = SkillVectorSchedule({}, backend.ASSUMPTIONS, model)
    overlap.add("llvm.hivm.vexp.x.v64f32", "call @exp(%x)", "%a")
    overlap.add("llvm.hivm.vmul.s.x.v64f32", "call @mul(%y)", "%b")
    assert overlap.ready["%b"] < overlap.ready["%a"]


def test_analyzers_use_instance_factory_without_changing_legacy(model):
    text = '''
define internal ptc_simdvf void @vf() {
  %1 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> undef)
  %2 = call <64 x float> @llvm.hivm.vcadd.s.x.v64f32.v64f32(<64 x float> undef)
  ret void
}
'''
    old = backend.Analyzer(text, {}).trace("vf", [], 0)
    new = backend.Analyzer(text, {}, vector_factory=model.vector_factory).trace("vf", [], 0)
    assert new["cycles"] > old["cycles"]
    assert backend.Analyzer(text, {}).trace("vf", [], 0) == old


def test_ini_units_and_invalid_rates(tmp_path):
    rates = ini_rates(DATA / "Ascend950PR_9579.ini")
    assert rates["core_count"] == 56
    assert rates["local_bytes_per_cycle"] == pytest.approx(16e9 / 1.65e9)
    path = tmp_path / "bad.ini"
    path.write_text(
        "[SoCInfo]\nvector_core_cnt=56\n[VectorCoreSpec]\nvec_freq=nan\n[VectorCoreMemoryRates]\nddr_rate=16\n")
    with pytest.raises(ValueError):
        ini_rates(path)


def resource_candidate(tmp_path):
    ir = '''module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">,
      spec = ["UB_SIZE", 2031616 : i32]} {
func.func @kernel() {
  %0 = llvm.mlir.constant(0 : i64)
  %1 = hivm.hir.pointer_cast(%0) : memref<64xf32, #hivm.address_space<ub>>
}}'''
    files = {}
    for name, contents in {
            "final_ir": ir, "binary": "binary", "metadata": json.dumps({"target": {"arch": "Ascend950PR_9579"}})
    }.items():
        path = tmp_path / name
        path.write_text(contents)
        files[name] = dict(path=str(path), sha256=sha256(path))
    proof = dict(status="UB_RESOURCE_CHECKED", resource_checked=True, artifact=files["binary"],
                 final_ir_sha256=files["final_ir"]["sha256"])
    path = tmp_path / "resource"
    path.write_text(json.dumps(proof))
    files["resource"] = dict(path=str(path), sha256=sha256(path))
    return dict(resource_checked=True, factor=1, files=files)


def test_resource_target_from_hashed_metadata_and_tampering(tmp_path):
    c = resource_candidate(tmp_path)
    assert verify_artifacts(c)[3]["address_extent_bits"] == 64 * 32
    Path(c["files"]["binary"]["path"]).write_text("changed")
    with pytest.raises(ValueError, match="binary changed"):
        verify_artifacts(c)


def test_resource_target_mismatch_rejected_even_with_valid_hashes(tmp_path):
    c = resource_candidate(tmp_path)
    path = Path(c["files"]["resource"]["path"])
    proof = json.loads(path.read_text())
    proof["target"] = "Ascend910B"
    path.write_text(json.dumps(proof))
    c["files"]["resource"]["sha256"] = sha256(path)
    with pytest.raises(ValueError, match="targets differ"):
        verify_artifacts(c)


def test_cpu_guard_blocks_timings_and_device_imports(tmp_path):
    output = tmp_path / "out"
    output.mkdir()
    forbidden = tmp_path / "timings.json"
    forbidden.write_text("{}")
    script = '''
from pathlib import Path
from run_vv_skill_selector import install_cpu_guard
events = install_cpu_guard(Path('manifest.json'), {'cases': []}, Path('cost.def'), Path(OUT))
for operation in (lambda: Path(FORBIDDEN).read_text(), lambda: __import__('torch_npu')):
    try:
        operation()
    except RuntimeError:
        pass
    else:
        raise AssertionError('prohibited access succeeded')
assert len(events['denied']) == 2
'''.replace("OUT", repr(str(output))).replace("FORBIDDEN", repr(str(forbidden)))
    process = subprocess.run([sys.executable, "-B", "-c", script], cwd=Path(run_vv_skill_selector.__file__).parent,
                             capture_output=True, text=True)
    assert process.returncode == 0, process.stderr


def test_post_selection_evaluation_can_include_fallback_without_changing_predictions():
    from copy import deepcopy
    from evaluate_vv_contention import verified_times
    from test_vv_contention_evaluation import records
    prediction, measurement = records()
    prediction["candidates"]["a0_f2"]["status"] = "STATIC_ANALYSIS_UNSUPPORTED"
    before = deepcopy(prediction)
    assert verified_times(prediction, measurement) == {}
    assert verified_times(prediction, measurement, require_static_score=False) == {"a0_f2": 6}
    assert prediction == before
    prediction["candidates"]["a0_f2"]["resource_checked"] = False
    assert verified_times(prediction, measurement, require_static_score=False) == {}
