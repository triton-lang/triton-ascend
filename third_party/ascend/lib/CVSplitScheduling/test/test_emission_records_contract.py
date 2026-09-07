#!/usr/bin/env python3
"""Source-contract checks for Verified emission records."""
from pathlib import Path
import re

TEST = Path(__file__).resolve().parent
LIB = TEST.parent
ROOT = TEST.parents[2]
HEADER = ROOT / "include" / "CVSplitScheduling" / "CrossCoreResourcePlan.h"
SOURCE = LIB / "CrossCoreResourcePlan.cpp"

def test_plan_has_explicit_flag_records() -> None:
    header = HEADER.read_text()
    assert "std::optional<unsigned> releaseFlagId" in header
    assert "unsigned forwardFlagId" in header

def test_flag_ids_are_deterministic_prefixes() -> None:
    source = SOURCE.read_text()
    required = (
        "unsigned nextFlag = plan.firstAvailableFlagId",
        "buildEmissionLineageOrder(pipelinePlan)",
        "boundary.producerOrder < *lastVectorToCubeOrder",
        "llvm::stable_sort(boundaries",
        "lineageFlagBase[lineageIndex] = nextFlag",
        "nextFlag += lineage.slotCount",
        "llvm::sort(delayedReleaseGroups)",
        "releaseFlagId = nextFlag++",
        "lineageFlagBase[lineageIndex]",
        "nextFlag - plan.firstAvailableFlagId != plan.requiredFlags",
    )
    missing = [token for token in required if token not in source]
    assert not missing, f"missing deterministic emission record: {missing}"

def test_records_are_generic_and_analysis_only() -> None:
    text = HEADER.read_text() + SOURCE.read_text()
    forbidden = (r"_attn_fwd", r"flash[_ -]?attention", r"\bHD(?:64|128)\b",
                 r"lane\s*==\s*[0-9]+", r"unrollFactor\s*==\s*[0-9]+")
    assert not [p for p in forbidden if re.search(p, text, re.I)]
    mutations = ("builder.create", "moveBefore", "replaceAllUses", "erase()")
    assert not [token for token in mutations if token in SOURCE.read_text()]

if __name__ == "__main__":
    test_plan_has_explicit_flag_records()
    test_flag_ids_are_deterministic_prefixes()
    test_records_are_generic_and_analysis_only()
    print("Verified emission-record source contract: PASS")
