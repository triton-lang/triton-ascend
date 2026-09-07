#!/usr/bin/env python3
"""Source contracts for Atomic schedule materialization and publication."""

from pathlib import Path

ROOT = Path(__file__).resolve().parents[5]
INCLUDE = ROOT / "third_party/ascend/include/CVSplitScheduling"
LIB = ROOT / "third_party/ascend/lib/CVSplitScheduling"
BACKEND = ROOT / "third_party/ascend/backend/compiler.py"


def read(path: Path) -> str:
    return path.read_text()


def test_materialize_mode_is_transactional() -> None:
    passes = read(INCLUDE / "Passes.td")
    cpp = read(LIB / "CVSplitScheduling.cpp")
    backend = read(BACKEND)
    assert 'Option<"postSplitScheduleMode"' in passes
    assert "PostSplitScheduleMode::Materialize" in cpp
    assert "materializePostSplitSchedule" in cpp
    assert 'cv_split_post_split_schedule_mode: str = "disabled"' in backend
    assert "bindingReady" in cpp
    assert "structuralCandidateReady" in cpp
    assert "materializationSchedule" in cpp
    assert "kPreserveExplicitScheduleAttr" in cpp
    assert "cube=yes vector=yes" in cpp
    assert "OwningOpRef<ModuleOp> transformedModule = moduleOp.clone()" in cpp
    assert "commitModuleClone(moduleOp, *transformedModule)" in cpp
    assert "enable_materialization_cube_only" not in backend
    assert "enable_materialization_vector_only" not in backend


def test_materializer_outlines_all_probability_regions() -> None:
    header = read(INCLUDE / "ScopeSeparation.h")
    source = read(LIB / "ScopeSeparation.cpp")
    assert "materializePostSplitSchedule" in header
    for token in (
            "materializeVectorLaneRegion",
            "materializeVectorLaneRegions",
            "VectorToCubePack &pack",
            "vector_mode",
            'StringAttr::get(context, "simd")',
            'BoolAttr::get(context, true)',
            "outputs.insert(pack.pSrc)",
            "pack.pSrc = result->packedProbability",
            "publication=detached",
    ):
        assert token in source
    outline = source.split("materializeVectorLaneRegion", 1)[1]
    assert "scope::ScopeOp" in outline
    assert "scope::ReturnOp" in outline
    assert "SyncBlockWaitOp" in outline
    assert "SyncBlockSetOp" in outline


def test_no_textual_or_shape_identity_policy() -> None:
    cpp = read(LIB / "CVSplitScheduling.cpp")
    scope = read(LIB / "ScopeSeparation.cpp")
    combined = (cpp + scope).lower()
    for forbidden in (
            "_attn_fwd",
            "flash_attention",
            "native_0316",
            "head_dim == 128",
            "logicalunrollfactor == 4",
            "scoreoriginid == 15",
            "productoriginid == 34",
    ):
        assert forbidden not in combined


def test_generated_row_loops_handle_empty_scf_bodies() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeOnlineSoftmaxLane", 1)[1]
    assert "if (!maxBody->empty())\n    maxBody->back().erase();" in online
    assert "if (!expBody->empty())\n    expBody->back().erase();" in online
    assert "b.create<arith::ConstantIntOp>(loc, 0, 32)" in online
    assert "b.create<arith::ConstantIntOp>(loc, rows, 32)" in online
    assert online.count("create<arith::IndexCastOp>") >= 2


def test_row_reduction_identities_are_materialized_inside_simd_scope() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeOnlineSoftmaxLane", 1)[1]
    assert "createReductionInit" in online
    assert "getDefiningOp<linalg::FillOp>()" in online
    assert "createReductionInit(mb, maxReduce, rowScalarType)" in online
    assert "createReductionInit(eb, sumReduce, rowScalarType)" in online


def test_loop_storage_is_explicitly_ub_backed_before_the_simd_scope() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeOnlineSoftmaxLane", 1)[1]
    scope = online.index("builder.create<scope::ScopeOp>")
    for token in (
            "sumRowsInit",
            "scaledRowsInit",
            "packedRowsInit",
            "maxRowsInit",
            '"cvsplit.softmax.max-rows"',
            '"cvsplit.softmax.sum-rows"',
            '"cvsplit.softmax.scaled-rows"',
            '"cvsplit.softmax.packed-rows"',
    ):
        assert online.index(token) < scope
    storage = online.split("auto createLoopStorage", 1)[1]
    storage = storage.split("auto createReductionInit", 1)[0]
    for token in (
            "hivm::AddressSpace::UB",
            "MemRefType::get",
            "memref::AllocOp",
            "annotation::MarkOp",
            'mark->setAttr("effects"',
            'getStringAttr("write")',
            'getStringAttr("read")',
            "memref::MemorySpaceCastOp",
            "bufferization::ToTensorOp",
            "/*restrict=*/true",
            "/*writable=*/true",
    ):
        assert token in storage
    assert "tensor::EmptyOp" not in storage
    assert "createLoopStorage(b," not in online
    assert 'getStringAttr("cvsplit.softmax.deferred-add-row")' in online
    assert "deferredAddInit = builder.create<tensor::EmptyOp>" in online


def test_full_scale_fill_is_narrowed_once_and_shared_across_lanes() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeOnlineSoftmaxLane", 1)[1]
    online = online.split("materializeVectorLaneRegions", 1)[0]
    for token in (
            "Value &sharedScaleScalar",
            "Value &sharedScaleRow",
            "scale.getDefiningOp<linalg::FillOp>()",
            "scaleFill.getInputs().size() != 1",
            "scaleFill.getResult(0) != scale",
            "sharedScaleScalar != scaleScalar",
            'getStringAttr("cvsplit.softmax.shared-scale-row")',
            "scaleBuilder.create<tensor::EmptyOp>",
            "scaleBuilder",
            ".create<linalg::FillOp>",
            "mb.create<arith::MulFOp>(loc, scoreChunk, sharedScaleRow)",
            "max-shared-scale-row-ready",
    ):
        assert token in online
    assert "extractRowChunk(mb, scale," not in online

    caller = source.split("materializeVectorLaneRegions", 1)[1]
    caller = caller.split("buildBufferOwnershipPlan", 1)[0]
    assert caller.index("Value sharedScaleScalar") < caller.index(
        "for (auto [lane, pack]")
    assert caller.index("Value sharedScaleRow") < caller.index(
        "for (auto [lane, pack]")
    assert "sharedScaleScalar,\n            sharedScaleRow" in caller


def test_product_operand_prefetch_consumes_the_structural_candidate_limit() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    header = read(INCLUDE / "ScopeSeparation.h")
    cpp = read(LIB / "CVSplitScheduling.cpp")
    for text in (header, source):
        assert "const CrossCoreScheduleCandidate *" in text
        assert "scheduleCandidate" in text
    assert "forcedScheduleCandidate" in cpp.split(
        "createScopeSeparation(", 1)[1]

    prefetch = source.split("sinkCubeLoadChainsToMatmul", 2)[2]
    prefetch = prefetch.split("ROW_SPLIT vector re-tile", 1)[0]
    for token in (
            "productOperandPrefetchDepth",
            "hivm::PIPE::PIPE_MTE3",
            "hivm::PIPE::PIPE_MTE1",
            "probabilityWaits",
            "productMatmuls",
            "lastScoreMatmul",
            "collectOperandChain",
            "memref::CopyOp",
            "copiedOperandChains != 1",
            "prefetchDistance = productOperandPrefetchDepth - 1",
            "lane < prefetchDistance",
            "productMatmuls[lane - prefetchDistance]",
            "anchor->isBeforeInBlock(definition)",
            "operation->moveBefore(anchor)",
            "product-operand-prefetch",
    ):
        assert token in prefetch
    for forbidden in ("lane == 0", "lane == 3", "flag == 4", "flag == 7"):
        assert forbidden not in prefetch

    callsite = source.split("// Step 6b:", 1)[1]
    callsite = callsite.split("// Step 7:", 1)[0]
    for token in (
            "materializePostSplitSchedule",
            "scheduleCandidate->logicalLaneCount",
            "detachedSchedule->logicalLaneCount",
            "scheduleCandidate->prefetchLimit",
            "productOperandPrefetchDepth",
            "failed(sinkCubeLoadChainsToMatmul(",
    ):
        assert token in callsite


def test_online_softmax_marks_the_inter_loop_vector_dependency() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeOnlineSoftmaxLane", 1)[1]
    marker = 'syncMark->setAttr("SYNC_IN_VF", StringAttr::get(context, "VST_VLD"))'
    assert marker in online
    assert online.index("Value maximum =") < online.index(marker)
    assert online.index(marker) < online.index("auto expLoop =")


def test_final_maximum_preserves_original_tensor_semantics() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeOnlineSoftmaxLane", 1)[1]
    assert "b.create<arith::MaximumFOp>(loc, oldMaximum, maxLoop.getResult(0))" in online
    assert '"cvsplit.softmax.maximum"' not in online


def test_direct_nz_pack_reshapes_f32_before_truncation() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeOnlineSoftmaxLane", 1)[1]
    reshape = "Value packedFloatChunk = eb.create<tensor::ReshapeOp>"
    truncate = "Value packedChunk = eb.create<arith::TruncFOp>"
    assert reshape in online
    assert truncate in online
    assert online.index(reshape) < online.index(truncate)
    assert "packedFloatChunkType, exponential, shape" in online
    assert "packedChunkType, packedFloatChunk" in online


def test_lane_scope_defers_only_the_last_logical_lane_sum() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    online = source.split("materializeOnlineSoftmaxLane", 1)[1]
    assert "struct OnlineSoftmaxLaneState" in source
    for token in (
            "bool deferLaneSum",
            "deferLaneSum ? maxLoop.getResult(1) : sumRowsInit",
            "if (deferLaneSum)",
            "simdScope->getResult(deferLaneSum ? 1 : 0)",
            "deferredBuilder.setInsertionPointAfter(anchor)",
            '"cvsplit.softmax.deferred-add-row"',
            "rb.create<linalg::AddOp>",
            "ValueRange{deferredAddInit}",
            "deferredScope",
            "deferredLoop",
            'deferLaneSum ? "deferred" : "inline"',
    ):
        assert token in online
    assert "ValueRange{expLoop.getResult(0), maximum, expLoop.getResult(1)}" in online
    assert "ValueRange{maximum, expLoop.getResult(0), expLoop.getResult(1)}" in online
    scope = online.index("builder.create<scope::ScopeOp>")
    assert scope < online.index('simdScope->setAttr("noinline"')
    assert online.index('simdScope->setAttr("noinline"') < online.index(
        'simdScope->setAttr("outline"')
    assert "setNoInline(true)" not in source
    worklist = online.split("SmallVector<Operation *> worklist", 1)[1].split("};", 1)[0]
    assert "alpha.getOperation()" not in worklist
    assert "newDenominator.getOperation()" not in worklist
    replacements = online.split("SmallVector<std::pair<Value, Value>> replacements", 1)[1]
    replacements = replacements.split("};", 1)[0]
    assert "newMaximum.getResult()" in replacements
    assert "sumReduce.getResult(0)" in replacements
    assert "newDenominator.getResult()" not in replacements
    assert "alpha.getResult()" not in replacements
    outline = source.split("materializeVectorLaneRegions", 1)[1]
    assert "lane + 1 == packs.size()" in outline


def test_grouped_recurrence_is_lane_count_driven_and_balanced() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    grouped = source.split("materializeGroupedSoftmaxRecurrence", 1)[1]
    grouped = grouped.split("materializeVectorLaneRegions", 1)[0]
    for token in (
            "alphaResultTypes(lanes.size(), rowType)",
            "for (OnlineSoftmaxLaneState state : lanes)",
            "previousMaximum = state.maximum",
            "while (segments.size() > 1)",
            "index + 1 == segments.size()",
            "left.scale, right.scale",
            "left.offset, right.scale",
            "scaledLeft, right.offset",
            "firstState.oldDenominator",
            "segments.front().scale",
            "segments.front().offset",
            "materialized-grouped-recurrence",
    ):
        assert token in grouped
    assert "lanes.size() == 4" not in grouped
    assert "lanes.size() != 4" not in grouped


def test_grouped_recurrence_replaces_alpha_and_final_denominator_atomically() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    grouped = source.split("materializeGroupedSoftmaxRecurrence", 1)[1]
    grouped = grouped.split("materializeVectorLaneRegions", 1)[0]
    assert "use->set(alphaScope->getResult(lane))" in grouped
    assert "use->set(affineScope->getResult(0))" in grouped
    for token in (
            "state.newDenominator.erase()",
            "state.scaledDenominator.erase()",
            "state.alpha.erase()",
            "state.alphaDifference.erase()",
    ):
        assert token in grouped
    outline = source.split("materializeVectorLaneRegions", 1)[1]
    assert "SmallVector<OnlineSoftmaxLaneState> lanes" in outline
    assert "return materializeGroupedSoftmaxRecurrence(lanes, groupedAlphaScope);" in outline


def test_release_protocol_is_driven_by_detached_schedule_roles_and_slots() -> None:
    header = read(INCLUDE / "ScopeSeparation.h")
    source = read(LIB / "ScopeSeparation.cpp")
    cpp = read(LIB / "CVSplitScheduling.cpp")
    assert "const PostCVSplitDetachedSchedule *" in header
    assert "detachedSchedule" in header
    assert "materializationSchedule.emplace(detachedSchedule)" in cpp
    assert "materializationSchedule ? &*materializationSchedule : nullptr" in cpp
    protocol = source.split("buildReleaseProtocolPlan", 1)[1]
    protocol = protocol.split("retileVectorScopeForRowSplit", 1)[0]
    for token in (
            "PostCVSplitDetachedCommandKind::ScorePublish",
            "PostCVSplitDetachedCommandKind::ScoreReleaseWait",
            "PostCVSplitDetachedCommandKind::ScoreRelease",
            "PostCVSplitDetachedCommandKind::ProbabilityPublish",
            "PostCVSplitDetachedCommandKind::ProductPublish",
            "PostCVSplitDetachedCommandKind::ProductReleaseWait",
            "PostCVSplitDetachedCommandKind::ProductRelease",
            "release->slot",
            "release->logicalFlagId",
            "initial.signalingResource",
            "initial.waitingResource",
            "pipeForResource",
            "legacyVectorSet->erase()",
            "legacyCubeWait->erase()",
            "scoreCubeBuilder.create<hivm::SyncBlockWaitOp>",
            "scoreVectorBuilder.create<hivm::SyncBlockSetOp>",
            "productCubeBuilder.create<hivm::SyncBlockWaitOp>",
            "productVectorBuilder.create<hivm::SyncBlockSetOp>",
            "cubeLoop->getParentOfType<scf::ForOp>()",
            "vectorLoop->getParentOfType<scf::ForOp>()",
            "cubeOuterLoop.getOperation() != vectorOuterLoop.getOperation()",
            "vectorOuterLoop.getInductionVar()",
            "vectorOuterLoop.getLowerBound()",
            "arith::CmpIPredicate::eq",
            "initialBuilder.create<scf::IfOp>",
            "seedIf.getThenBodyBuilder()",
            "seedBuilder.create<hivm::SyncBlockSetOp>",
            "initial-once-per-outer-loop=yes",
            "materialized-release-protocol",
    ):
        assert token in protocol
    assert "initialBuilder.create<hivm::SyncBlockSetOp>" not in protocol
    for forbidden in (
            "scoreReleaseFlag = 12",
            "scoreReleaseFlag = 13",
            "productReleaseFlag = 14",
            "productReleaseFlag = 15",
    ):
        assert forbidden not in protocol


def test_buffer_ownership_is_rematerialized_by_role_and_plan_slot() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    ownership = source.split("buildBufferOwnershipPlan", 1)[1]
    ownership = ownership.split("buildReleaseProtocolPlan", 1)[0]
    for token in (
            "transferInfo.cubeToVectorChains",
            "PostCVSplitDetachedCommandKind::ScorePublish",
            "PostCVSplitDetachedCommandKind::ProductPublish",
            "publish->lane",
            "publish->slot",
            "plan.scoreSlotCount",
            "plan.productSlotCount",
            "plan->scoreBufferType, plan->scoreSlotCount",
            "plan->productBufferType, plan->productSlotCount",
            "lane.cubeDrain->replaceUsesOfWith",
            "lane.vectorCast->setOperand(0, replacement)",
            "allocation.erase()",
            "materialized-buffer-ownership",
    ):
        assert token in ownership
    assert "scoreSlotCount = 2" not in ownership
    assert "productSlotCount = 2" not in ownership
    retile = source.split("retileVectorScopeForRowSplit", 2)[2]
    assert retile.index("materializeBufferOwnership(") < retile.index(
        "materializeReleaseProtocol(")


def test_cube_drain_clusters_follow_detached_command_order() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    ordering = source.split("orderCubeDrainsBySchedule", 1)[1]
    ordering = ordering.split("retileVectorScopeForRowSplit", 1)[0]
    for token in (
            "llvm::enumerate(schedule.cubeCommands)",
            "PostCVSplitDetachedCommandKind::ScorePublish",
            "PostCVSplitDetachedCommandKind::ProductPublish",
            "ordinalByForwardFlag",
            "groups[end - 1].publish->getNextNode()",
            "left.ordinal < right.ordinal",
            "group.releaseWait->moveBefore(afterCluster)",
            "group.drain->moveBefore(afterCluster)",
            "group.publish->moveBefore(afterCluster)",
            "ordered-cube-drain-clusters",
    ):
        assert token in ordering
    for forbidden in ("lane == 2", "lane == 3", "flag == 10", "flag == 11"):
        assert forbidden not in ordering
    retile = source.split("retileVectorScopeForRowSplit", 2)[2]
    assert retile.index("materializeReleaseProtocol(") < retile.index(
        "orderCubeDrainsBySchedule(")


def test_grouped_alpha_precedes_the_first_planned_product_wait() -> None:
    source = read(LIB / "ScopeSeparation.cpp")
    ordering = source.split("orderGroupedAlphaBeforeProductWait", 1)[1]
    ordering = ordering.split("orderCubeDrainsBySchedule", 1)[0]
    for token in (
            "schedule.vectorCommands",
            "PostCVSplitDetachedCommandKind::ProductWait",
            "firstProductWait->logicalFlagId",
            "waitOperation->isBeforeInBlock(groupedAlphaScope)",
            "groupedAlphaScope->moveBefore(waitOperation)",
            "ordered-grouped-alpha-before-product-wait",
    ):
        assert token in ordering
    assert "logicalFlagId == 8" not in ordering
    retile = source.split("retileVectorScopeForRowSplit", 2)[2]
    assert retile.index("materializeReleaseProtocol(") < retile.index(
        "orderGroupedAlphaBeforeProductWait(")


if __name__ == "__main__":
    test_materialize_mode_is_transactional()
    test_materializer_outlines_all_probability_regions()
    test_no_textual_or_shape_identity_policy()
    test_generated_row_loops_handle_empty_scf_bodies()
    test_row_reduction_identities_are_materialized_inside_simd_scope()
    test_loop_storage_is_explicitly_ub_backed_before_the_simd_scope()
    test_full_scale_fill_is_narrowed_once_and_shared_across_lanes()
    test_product_operand_prefetch_consumes_the_structural_candidate_limit()
    test_online_softmax_marks_the_inter_loop_vector_dependency()
    test_final_maximum_preserves_original_tensor_semantics()
    test_direct_nz_pack_reshapes_f32_before_truncation()
    test_lane_scope_defers_only_the_last_logical_lane_sum()
    test_grouped_recurrence_is_lane_count_driven_and_balanced()
    test_grouped_recurrence_replaces_alpha_and_final_denominator_atomically()
    test_release_protocol_is_driven_by_detached_schedule_roles_and_slots()
    test_buffer_ownership_is_rematerialized_by_role_and_plan_slot()
    test_cube_drain_clusters_follow_detached_command_order()
    test_grouped_alpha_precedes_the_first_planned_product_wait()
    print("Atomic schedule materialization source contract: PASS")
