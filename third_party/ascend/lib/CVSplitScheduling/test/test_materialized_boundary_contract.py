#!/usr/bin/env python3
"""Source-contract checks for Materialized boundaries."""

from pathlib import Path


TEST_DIR = Path(__file__).resolve().parent
CVSPLIT_LIB = TEST_DIR.parent
ASCEND_ROOT = TEST_DIR.parents[2]
PIPELINE_HEADER = (
    ASCEND_ROOT / "include" / "CVSplitScheduling" / "CrossCorePipelinePlan.h"
)
RESOURCE_HEADER = (
    ASCEND_ROOT / "include" / "CVSplitScheduling" / "CrossCoreResourcePlan.h"
)
UNFUSE_HEADER = (
    ASCEND_ROOT / "include" / "CVSplitScheduling" / "UnfusePVMatmuls.h"
)
PIPELINE_SOURCE = CVSPLIT_LIB / "CrossCorePipelinePlan.cpp"
UNFUSE_SOURCE = CVSPLIT_LIB / "UnfusePVMatmuls.cpp"
RESOURCE_SOURCE = CVSPLIT_LIB / "CrossCoreResourcePlan.cpp"
PASS = CVSPLIT_LIB / "CVSplitScheduling.cpp"


def test_rewrite_returns_real_in_memory_bindings() -> None:
    header = UNFUSE_HEADER.read_text()
    source = UNFUSE_SOURCE.read_text()
    required = {
        "AccumulatorJoinBinding",
        "matmulProducer",
        "vectorJoin",
        "AccumulatorJoinRewriteResult",
        "unfuseVectorAccumulatorMatmuls",
    }
    missing = sorted(token for token in required if token not in header)
    assert not missing, f"missing rewrite-result API: {missing}"
    assert "rewriteResult.bindings.push_back" in source
    assert "matmulOp.getOperation()" in source
    assert "addOp.getOperation()" in source


def test_binder_requires_real_consumers_and_refreshes_order() -> None:
    header = PIPELINE_HEADER.read_text()
    source = PIPELINE_SOURCE.read_text()
    assert "bindCrossCorePipelinePlan" in header
    required = {
        "joinByProducer",
        "uniqueJoins",
        "directConsumers.size() != 1",
        "directConsumers.front() != join",
        "join->getOperands()",
        "boundary.consumers = {join}",
        "boundary.lastReader = join",
        "operationOrder[&op] = nextOrder++",
        "boundary.producerOrder = operationOrder.lookup",
        "boundary.lastReaderOrder = operationOrder.lookup",
        "usedBindings.size() != joinByProducer.size()",
    }
    missing = sorted(token for token in required if token not in source)
    assert not missing, f"missing materialized-boundary validation: {missing}"
    forbidden_mutations = [
        "builder.create",
        "moveBefore",
        "moveAfter",
        "replaceAllUsesWith",
        "replaceUsesOfWith",
        "setAttr(",
        "erase()",
    ]
    present = [token for token in forbidden_mutations if token in source]
    assert not present, f"binder mutates IR: {present}"


def test_binding_runs_after_scheduler_and_feeds_verified_emitter() -> None:
    text = PASS.read_text()
    scheduler = text.index("scheduler.run")
    rewrite = text.index("unfuseVectorAccumulatorMatmuls")
    binder = text.index("bindCrossCorePipelinePlan")
    bound_resource = text.index("resourceResult =", binder)
    transfer = text.index("cv_split::insertCrossScopeTransfers")
    assert scheduler < rewrite < binder < bound_resource < transfer
    scheduler_call = text[scheduler:text.index("return failure();", scheduler)]
    transfer_call = text[transfer:text.index("if (failed(transferInfo))", transfer)]
    assert "materializedPlan" not in scheduler_call
    assert "materializedResources" not in scheduler_call
    assert "materializedPlan" in transfer_call
    assert "materializedResources" in transfer_call


def test_unresolved_ownership_remains_non_selectable() -> None:
    header = RESOURCE_HEADER.read_text()
    source = RESOURCE_SOURCE.read_text()
    assert "UnresolvedOwnership" in header
    assert "else if (!plan.ownershipResolved)" in source
    assert "logMaterializedCrossCoreResourcePlan" in header
    assert 'logResourcePlan(plan, "bound-resource")' in source


if __name__ == "__main__":
    test_rewrite_returns_real_in_memory_bindings()
    test_binder_requires_real_consumers_and_refreshes_order()
    test_binding_runs_after_scheduler_and_feeds_verified_emitter()
    test_unresolved_ownership_remains_non_selectable()
    print("Materialized-boundary source contract: PASS")
