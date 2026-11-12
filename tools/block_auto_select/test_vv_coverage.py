import json

import pytest

from backend_cost_model import Analyzer, Unsupported
from cpu_resource_gate import inspect_final_ub, sha256
from vv_library_model import LibraryModel, DATA
from vv_resource_gate import inspect_vv_ub
from vv_skill_model import InstructionModel


def ir(dtype="i1", count=8, offset=0, mode="MIX"):
    return f'''module attributes {{hacc.target = #hacc.target<"Ascend950PR_9579">,
spec = ["UB_SIZE", 2031616 : i32, "UB_ALIGN_SIZE", 256 : i32,
"MINIMAL_D_CACHE_SIZE", 262144 : i32, "MAXIMUM_D_CACHE_SIZE", 983040 : i32]}} {{
func.func @kernel() attributes {{hacc.entry, hivm.vf_mode = #hivm.vf_mode<{mode}>, hivm.func_core_type = #hivm.func_core_type<AIV>}} {{
  %0 = llvm.mlir.constant({offset} : i64)
  %1 = hivm.hir.pointer_cast(%0) : memref<{count}x{dtype}, #hivm.address_space<ub>>
}}
}}'''


def evidence(tmp_path):
    defaults = tmp_path / "defaults.txt"
    defaults.write_text("--simt-vf-dynamic-size=<int> Default is 216.\n")
    meta = dict(parallel_mode="mix_simd_simt", mix_mode="aiv", shared_mem_dynamic_size=216 * 1024)
    files = dict(metadata=dict(sha256="metadata"), backend_defaults=dict(path=str(defaults), sha256=sha256(defaults)))
    proof = dict(metadata=files["metadata"], ub=dict(capacity_bits=216 * 1024 * 8), command=[],
                 default_option_evidence_sha256=files["backend_defaults"]["sha256"])
    return meta, proof, files


def test_i1_upper_bound_is_aligned_and_does_not_relax_legacy_parser(tmp_path):
    meta, proof, files = evidence(tmp_path)
    usage = inspect_vv_ub(ir(count=33), "Ascend950PR_9579", meta, proof, files)
    assert usage["address_extent_bits"] == 64 * 8
    assert usage["conservative_upper_bound"]
    assert usage["capacity_bits"] == 216 * 1024 * 8
    with pytest.raises(ValueError, match="unsupported UB element"):
        inspect_final_ub(ir(), "Ascend950PR_9579")


def test_i1_upper_bound_overflow_is_unknown_not_proven_overflow(tmp_path):
    meta, proof, files = evidence(tmp_path)
    with pytest.raises(ValueError, match="actual overflow unconfirmed"):
        inspect_vv_ub(ir(count=64, offset=216 * 1024 - 32), "Ascend950PR_9579", meta, proof, files)


def test_simd_alignment_and_mode_mismatch(tmp_path):
    meta = dict(parallel_mode="simd", mix_mode="aiv")
    assert inspect_vv_ub(ir(dtype="f32", count=10, mode="SIMD"), "Ascend950PR_9579", meta, {},
                         {})["address_extent_bits"] == 64 * 8
    with pytest.raises(ValueError, match="vector modes differ"):
        inspect_vv_ub(ir(), "Ascend950PR_9579", meta, {}, {})


@pytest.mark.parametrize("change", ["cache", "runtime", "metadata", "default", "proof_capacity", "command"])
def test_mixed_capacity_and_provenance_mismatches_fail_closed(tmp_path, change):
    meta, proof, files = evidence(tmp_path)
    if change == "cache": meta["shared_mem_dynamic_size"] = 248 * 1024
    if change == "runtime": meta["shared_mem_dynamic_size"] = 208 * 1024
    if change == "metadata": proof["metadata"] = dict(sha256="different")
    if change == "default": proof["default_option_evidence_sha256"] = "different"
    if change == "proof_capacity": proof["ub"]["capacity_bits"] = 248 * 1024 * 8
    if change == "command": proof["command"] = ["--simt-vf-dynamic-size=208"]
    with pytest.raises(ValueError):
        inspect_vv_ub(ir(), "Ascend950PR_9579", meta, proof, files)


@pytest.fixture(scope="module")
def library():
    model = InstructionModel()
    provenance = json.loads((DATA / "provenance.json").read_text())
    return LibraryModel({}, {}, model.vector_factory, {"sha256": provenance["libdevice_sha256"]})


def test_real_library_body_counts_full_vectors_and_masked_tail(library):
    one = library.pow([None, None, 0, 64, 1] * 3)
    two = library.pow([None, None, 0, 128, 1] * 3)
    tail = library.pow([None, None, 0, 65, 1] * 3)
    load = "llvm.hivm.vldsx1.v64f32"
    assert one["counts"][load] == 2
    assert two["counts"][load] == tail["counts"][load] == 4
    assert one["cycles"] < two["cycles"] and one["cycles"] < tail["cycles"]
    assert library.pow([None, None, 0, 64, 1] * 3) == one
    assert one["counts"]["llvm.hivm.vexp.x.v64f32"] > 0


def test_library_requires_matching_binary_and_proven_shapes(library):
    with pytest.raises(Unsupported, match="pinned libdevice"):
        LibraryModel({}, {}, InstructionModel().vector_factory, {"sha256": "wrong"})
    with pytest.raises(Unsupported, match="unknown pow extent"):
        library.pow([None, None, 0, None, 1] * 3)
    with pytest.raises(Unsupported, match="contiguous"):
        library.pow([None, None, 0, 64, 2] * 3)


def test_optimized_model_applies_same_scheduler_to_pow_library(monkeypatch):
    import backend_cost_model as backend
    import vv_optimized_ir
    import vv_skill_model
    from vv_ready_schedule import ReadyVectorSchedule

    model = InstructionModel()
    provenance = json.loads((DATA / "provenance.json").read_text())
    monkeypatch.setattr(vv_optimized_ir, "verified_vectors", lambda text, evidence: (text, {"verified": True}))

    class Checked(Exception):
        pass

    def check_library(*args, **kwargs):
        library = kwargs["kernel_factory"]({}).library
        assert library.analyzer.vector_factory is kwargs["vector_factory"]
        assert isinstance(library.analyzer.vector_factory({}, backend.ASSUMPTIONS), ReadyVectorSchedule)
        full = library.pow([None, None, 0, 64, 1] * 3)
        tail = library.pow([None, None, 0, 65, 1] * 3)
        assert full["counts"]["llvm.hivm.vldsx1.v64f32"] == 2
        assert tail["counts"]["llvm.hivm.vldsx1.v64f32"] == 4
        assert 0 < full["cycles"] < tail["cycles"]
        raise Checked

    monkeypatch.setattr(backend, "analyze_candidate", check_library)
    with pytest.raises(Checked):
        vv_skill_model.analyze_candidate("call void @pow_fp32()", "", "",
                                         {"target": {"arch": "Ascend950PR_9579"}, "parallel_mode": "simd"}, {}, 0, 1,
                                         {}, model=model, diagnostic=False,
                                         library_evidence={"sha256": provenance["libdevice_sha256"]},
                                         optimized_evidence={"test_proof": True})


def test_freeze_does_not_invent_tensor_data():
    text = '''
define void @k(i32 %0) {
  %2 = freeze i32 %0
  %3 = icmp eq i32 %2, 0, !loop.guard !1
  br i1 %3, label %4, label %5, !loop.guard !1
4:
  ret void
5:
  ret void
}
'''
    assert Analyzer(text, {}).trace("k", [0], 0)["block_visits"]["4"] == 1
    with pytest.raises(Unsupported, match="data-dependent"):
        Analyzer(text, {}).trace("k", [None], 0)
