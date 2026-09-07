#!/usr/bin/env python3
"""Source-contract checks for Ownership reachability."""
from pathlib import Path
import re

TEST = Path(__file__).resolve().parent
LIB = TEST.parent
ROOT = TEST.parents[2]
PIPE = ROOT / "include" / "CVSplitScheduling" / "CrossCorePipelinePlan.h"
RESOURCE = ROOT / "include" / "CVSplitScheduling" / "CrossCoreResourcePlan.h"
PROOF = LIB / "CrossCoreOwnershipProof.cpp"
PLANNER = LIB / "CrossCoreResourcePlan.cpp"
CMAKE = LIB / "CMakeLists.txt"

def test_inputs_and_results_are_explicit() -> None:
    pipe, resource = PIPE.read_text(), RESOURCE.read_text()
    for token in ("EngineType engine", "PrincipalResource resource",
                  "unsigned order"):
        assert token in pipe
    for token in ("SameEngineOrder", "usesCrossCorePath",
                  "unresolvedOwnershipEdges", "loopCarriedOwnershipEdges",
                  "seedRequirements"):
        assert token in resource

def test_graph_is_conservative_and_two_iteration() -> None:
    text = PROOF.read_text()
    required = ("std::array<llvm::SmallVector<const PipelineResourceUse *>",
                "it->second->engine == use.engine",
                "consumer->second->engine == producer->second->engine",
                "llvm::SmallVector<uint8_t> seen(n * 4, 0)",
                "if (iter == 0 && g.wrap[node] >= 0)",
                "edge.loopCarried ? 1u : 0u")
    missing = [token for token in required if token not in text]
    assert not missing, f"missing ownership proof rules: {missing}"
    assert "same engine program order" not in text.lower()

def test_release_and_fallback_are_conservative() -> None:
    text = PROOF.read_text()
    assert "edge.loopCarried && delayed.contains(edge.physicalGroup)" in text
    assert "edge.needsSeed = false" in text
    assert "ResourceOwnershipOrdering::Unresolved" in text
    assert "unresolvedOwnershipEdges == 0" in text
    planner = PLANNER.read_text()
    assert "if (plan.anchorsComplete)" in planner
    assert "proveCrossCoreResourceOwnership" in planner

def test_proof_is_generic_and_does_not_mutate_ir() -> None:
    text = PROOF.read_text()
    forbidden = (r"_attn_fwd", r"flash[_ -]?attention", r"head[_ -]?dim",
                 r"\bHD(?:64|128)\b", r"lane\s*==\s*[0-9]+",
                 r"unrollFactor\s*==\s*[0-9]+")
    assert not [p for p in forbidden if re.search(p, text, re.I)]
    mutations = ("builder.create", "moveBefore", "moveAfter",
                 "replaceAllUses", "setAttr(", "erase()")
    assert not [token for token in mutations if token in text]
    assert "CrossCoreOwnershipProof.cpp" in CMAKE.read_text()

if __name__ == "__main__":
    test_inputs_and_results_are_explicit()
    test_graph_is_conservative_and_two_iteration()
    test_release_and_fallback_are_conservative()
    test_proof_is_generic_and_does_not_mutate_ir()
    print("Ownership-reachability source contract: PASS")
