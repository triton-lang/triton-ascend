"""CPU model tests runnable without importing Triton or initializing a device."""
import importlib
import copy
import json
import os
from pathlib import Path
import re
import sys
from types import ModuleType

import pytest

BACKEND = Path(__file__).resolve().parents[2] / "backend"
PACKAGE = "_ascend_block_auto_model_test"


@pytest.fixture(scope="module")
def model():
    package = ModuleType(PACKAGE)
    package.__path__ = [str(BACKEND)]
    sys.modules[PACKAGE] = package
    return importlib.import_module(PACKAGE + ".block_auto_model")


def test_packaged_cost_and_instruction_assets(model):
    table = model._cost_table()
    assert table[("VFMulOp", "F32", "F32")][0] > 0
    instruction, rate = model._instruction_model().instruction("llvm.hivm.vcvtff.bf162f32.x")
    assert instruction["elements"] == 64
    assert rate["issue_slots"] in (1, 2)


def test_post_constexpr_argument_binding_keeps_hidden_grid_separate(model):
    ttir = "tt.func public @f(%arg0: !tt.ptr<f32>, %arg1: i32, %__hacc_original_grid_x: i32) {}"
    contract = dict(grid=[17, 1, 1], arguments={"n": dict(kind="scalar", value=13), "out": dict(kind="tensor")},
                    argument_order=["out", "n"])
    bound = model._contract_for_ttir(contract, ttir)
    backend = importlib.import_module(PACKAGE + ".block_auto_models.backend_cost_model")
    args, grid = backend.launch_arguments(ttir, bound, 0, 4)
    assert args == [None, None, None, 13, 17, 5, 1, 1]
    assert grid == [5, 1, 1]


@pytest.mark.parametrize(
    "argument", [dict(kind="unsupported"),
                 dict(kind="scalar", value=float("nan")),
                 dict(kind="scalar", value=None)])
def test_unknown_runtime_argument_never_becomes_a_model_constant(model, argument):
    with pytest.raises(ValueError):
        model._contract_for_ttir(dict(grid=[1, 1, 1], arguments={"x": argument}), "tt.func public @f(%arg0: i32) {}")


def test_actual_model_rejects_data_dependent_control_flow(model):
    backend = importlib.import_module(PACKAGE + ".block_auto_models.backend_cost_model")
    llvm = """
define void @main(ptr addrspace(1) %0) {
  %2 = load i32, ptr addrspace(1) %0
  %3 = icmp sgt i32 %2, 0
  br i1 %3, label %4, label %5
4:
  ret void
5:
  ret void
}
"""
    with pytest.raises(ValueError, match="data-dependent branch"):
        backend.Analyzer(llvm, model._cost_table()).trace("main", [None], 0)


def _canonical_target_candidate():
    return dict(
        metadata=dict(target=dict(arch="Ascend950PR_9579"), mix_mode="aiv", parallel_mode="simd",
                      block_auto_evidence=dict(target="Ascend950PR_9579", command=["compiler", "--target=Ascend950"])),
        ttadapter='module attributes {hacc.target = #hacc.target<"Ascend950PR_9579">} {}',
        final_ir='''module attributes {hacc.target = #hacc.target<"Ascend950">,
          limits = ["UB_SIZE", 2031616 : i32, "UB_ALIGN_SIZE", 256 : i32]} {
          func.func @main() attributes {hacc.entry, hivm.vf_mode<SIMD>, hivm.func_core_type<AIV>} {
            %base = llvm.mlir.constant(0 : i64)
            %buffer = hivm.hir.pointer_cast(%base) : memref<16xf32, #hivm.address_space<ub>>
          }
        }''', files=dict(metadata=dict(sha256="metadata")))


@pytest.mark.parametrize("mode", ["aiv", "mix"])
def test_canonical_950_target_has_explicit_9579_compile_proof(model, mode):
    candidate = _canonical_target_candidate()
    candidate["metadata"]["mix_mode"] = mode
    before = candidate["final_ir"]
    model._resource_check(candidate)
    assert candidate["resource_checked"] and candidate["ub_usage"]["address_extent_bits"] == 512
    assert candidate["metadata"]["target"]["arch"] == "Ascend950PR_9579"
    assert candidate["final_ir"] == before
    assert candidate["target_binding"]["compiler"] == "Ascend950"


@pytest.mark.parametrize("changed", ["frontend", "command", "recorded_target", "duplicate_option", "final_target"])
def test_canonical_target_mapping_rejects_unbound_or_other_architecture(model, changed):
    candidate = copy.deepcopy(_canonical_target_candidate())
    if changed == "frontend":
        candidate["ttadapter"] = candidate["ttadapter"].replace("Ascend950PR_9579", "Ascend950PR_9589")
    elif changed == "command":
        candidate["metadata"]["block_auto_evidence"]["command"] = []
    elif changed == "recorded_target":
        candidate["metadata"]["block_auto_evidence"]["target"] = "Ascend950PR_9589"
    elif changed == "duplicate_option":
        candidate["metadata"]["block_auto_evidence"]["command"].append("--target=Ascend950")
    else:
        candidate["final_ir"] = candidate["final_ir"].replace('target<"Ascend950">', 'target<"Ascend910B">')
    with pytest.raises(ValueError):
        model._resource_check(candidate)


def test_missing_same_compile_evidence_keeps_original(model):

    class Metadata:

        def _asdict(self):
            return {}

    class Kernel:
        metadata = Metadata()

    factor, report = model.select_candidates(
        [dict(factor=1, axis=-1, kernel=Kernel()),
         dict(factor=2, axis=0, kernel=Kernel())], {})
    assert factor == 1
    assert not report["model_complete"]
    assert report["reason"] == "BASELINE_RESOURCE_UNCONFIRMED"


def test_changed_packaged_table_cannot_silently_supply_costs(model, tmp_path, monkeypatch):
    data = tmp_path / "data"
    data.mkdir()
    (data / "AVECostModelInfo.def").write_text("changed")
    (data / "provenance.json").write_text(json.dumps(dict(schema_version=1, sha256="not-the-file")))
    model._cost_table.cache_clear()
    try:
        with monkeypatch.context() as patch:
            patch.setattr(model, "_MODELS", tmp_path)
            with pytest.raises(ValueError, match="identity mismatch"):
                model._cost_table()
    finally:
        model._cost_table.cache_clear()


def test_vv_incomplete_compiled_family_retains_baseline(model, monkeypatch):
    skill = importlib.import_module(PACKAGE + ".block_auto_models.vv_skill_model")
    monkeypatch.setattr(model, "_optimized_evidence", lambda item: {"compiler": {"path": "/installed/bin/compiler"}})
    monkeypatch.setattr(model, "_instruction_model", lambda: None)
    monkeypatch.setattr(skill, "analyze_candidate",
                        lambda *args, **kwargs: dict(status="STATIC_COST_ESTIMATED", score=7.))
    items = [
        dict(key="unmerged", factor=1, axis=-1, resource_checked=True, metadata={}, ttir="tt.func public @f() {}",
             llvm="", final_ir=""),
        dict(key="a0_f2", factor=2, axis=0, resource_checked=False, status="RESOURCE_UNCONFIRMED")
    ]
    result = model._rank_vv(items, dict(grid=[128, 1, 1], arguments={}), model._cost_table())
    assert result["selected_key"] == "unmerged" and not result["model_complete"]


def test_compiled_core_floor_excludes_small_launch_before_ranking(model, monkeypatch):
    skill = importlib.import_module(PACKAGE + ".block_auto_models.vv_skill_model")
    monkeypatch.setattr(model, "_optimized_evidence", lambda item: {"compiler": {"path": "/installed/bin/compiler"}})
    monkeypatch.setattr(model, "_instruction_model", lambda: None)
    monkeypatch.setattr(skill, "analyze_candidate",
                        lambda *args, **kwargs: dict(status="STATIC_COST_ESTIMATED", score=100. / args[6]))
    items = [
        dict(key="unmerged" if factor == 1 else f"a0_f{factor}", factor=factor, axis=-1 if factor == 1 else 0,
             resource_checked=True, metadata={}, ttir="tt.func public @f() {}", llvm="",
             final_ir='"VECTOR_CORE_COUNT", 56 : i32') for factor in (1, 2, 4)
    ]
    contract = dict(grid=[32, 1, 1], arguments={})
    result = model._rank_vv(items, contract, model._cost_table())
    assert result["selected_key"] == "unmerged" and result["model_complete"]
    assert result["candidates"]["a0_f2"]["error"] == "GRID_BELOW_CORE_FLOOR"
    override = model._rank_vv(items, dict(contract, core_floor="0"), model._cost_table())
    assert override["selected_key"] == "a0_f4"


def test_frozen_vv_artifact_replay(model, monkeypatch):
    """Optional external sealed-artifact replay; benchmark data is not packaged."""
    root = os.environ.get("BLOCK_AUTO_REPLAY_ROOT")
    if not root:
        pytest.skip("set BLOCK_AUTO_REPLAY_ROOT to a sealed CPU evidence directory")
    root = Path(root)
    manifest = json.loads((root / "optimized/manifest.json").read_text())
    expected = {item["id"]: item for item in json.loads((root / "frozen-selections.json").read_text())}
    # Existing optimized evidence is still validated inside analyze_candidate;
    # replay need not run the native compiler a second time.
    monkeypatch.setattr(model, "_optimized_evidence", lambda item: dict(item["files"]))
    for case in manifest["cases"]:
        items = []
        for key, candidate in case["candidates"].items():
            if "files" not in candidate:
                items.append(
                    dict(key=key, factor=candidate["factor"], axis=candidate["axis"], status=candidate["status"],
                         error=candidate.get("error"), resource_checked=False))
                continue
            files = candidate["files"]
            for role, record in files.items():
                assert model.sha256(record["path"]) == record["sha256"], role
            metadata = json.loads(Path(files["metadata"]["path"]).read_text())
            proof = json.loads(Path(files["resource"]["path"]).read_text())
            metadata["block_auto_evidence"] = dict(command=proof.get("command", []))
            item = dict(key=key, factor=candidate["factor"], axis=candidate["axis"], metadata=metadata, files=files,
                        **{role: Path(files[role]["path"]).read_text()
                           for role in ("ttir", "llvm", "final_ir")})
            model._resource_check(item)
            items.append(item)
        baseline = next(item for item in items if item["factor"] == 1)
        signature = next(line for line in baseline["ttir"].splitlines() if "tt.func public @" in line)
        order = re.findall(r"%([\w.$-]+):", signature)
        contract = dict(case["contract"], arguments={name: case["contract"]["arguments"][name]
                                                     for name in order}, argument_order=order)
        result = model._rank_vv(items, contract, model._cost_table())
        assert result["model_complete"], result
        assert result["selected_key"] == expected[case["id"]]["selected"], (case["id"], result)
