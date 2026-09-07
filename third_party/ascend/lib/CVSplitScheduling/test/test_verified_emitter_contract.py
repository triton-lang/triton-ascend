#!/usr/bin/env python3
"""Source-contract checks for Verified-plan consumption."""
from pathlib import Path
import re

TEST = Path(__file__).resolve().parent
LIB = TEST.parent
ROOT = TEST.parents[2]
HEADER = ROOT / "include" / "CVSplitScheduling" / "CrossScopeTransfers.h"
SOURCE = LIB / "CrossScopeTransfers.cpp"
PASS = LIB / "CVSplitScheduling.cpp"

def test_materialized_plan_reaches_emitter() -> None:
    header, source, caller = HEADER.read_text(), SOURCE.read_text(), PASS.read_text()
    assert "const CrossCorePipelinePlan *materializedPlan" in header
    assert "const CrossCoreResourcePlan *resourcePlan" in header
    call = caller[caller.index("cv_split::insertCrossScopeTransfers"):]
    assert "materializedPlan ? &*materializedPlan" in call
    assert "materializedResources ? &*materializedResources" in call
    assert "hasMaterializedPlan != hasResourcePlan" in source

def test_shadow_gate_runs_before_ir_emission() -> None:
    text = SOURCE.read_text()
    gate = text.index("const bool useVerifiedPlan")
    allocation = text.index("BufferPool bufferPool", gate)
    assert gate < allocation
    required = ("ValidUnknownCapacity", "ownershipResolved",
                "unresolvedOwnershipEdges != 0", "seedRequirements != 0",
                "firstAvailableFlagId", "assignments.size() != transfers.size()")
    assert not [token for token in required if token not in text[gate:allocation]]

def test_emission_uses_verified_records() -> None:
    text = SOURCE.read_text()
    required = ("verifiedAssignmentByProducer.lookup(xfer.producer)",
                "slot = assignment.slot", "forwardFlagId = assignment.forwardFlagId",
                "groupKey = assignment.physicalGroup",
                "group.releaseFlagId", "verifiedGroupByOrigin")
    assert not [token for token in required if token not in text]
    assert "verified emission plan matched" in text

def test_policy_is_generic() -> None:
    source = SOURCE.read_text()
    text = source[source.index("const bool useVerifiedPlan"):
                  source.index("BufferPool bufferPool")]
    forbidden = (r"_attn_fwd", r"flash[_ -]?attention", r"\bHD(?:64|128)\b",
                 r"lane\s*==\s*[0-9]+", r"unrollFactor\s*==\s*[0-9]+")
    assert not [p for p in forbidden if re.search(p, text, re.I)]

if __name__ == "__main__":
    test_materialized_plan_reaches_emitter()
    test_shadow_gate_runs_before_ir_emission()
    test_emission_uses_verified_records()
    test_policy_is_generic()
    print("Verified-emitter source contract: PASS")
