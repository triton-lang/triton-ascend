#include "AscendModel/RouteModel/SimdSimtCostModel.h"

#include <gtest/gtest.h>

using mlir::ascend::estimateSimdSimtCandidates;
using mlir::ascend::SimdSimtCandidateKind;
using mlir::ascend::SimdSimtCostModelOptions;
using mlir::ascend::SimdSimtFeatureSummary;

namespace {

SimdSimtCostModelOptions options(unsigned numWarps) {
  SimdSimtCostModelOptions result;
  result.profilePath = TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH;
  result.actualTarget = "Ascend950PR_9579";
  result.numWarps = numWarps;
  result.compileOn91095 = true;
  return result;
}

SimdSimtFeatureSummary gatherDotFeatures() {
  SimdSimtFeatureSummary f;
  f.loadOps = 3;
  f.storeOps = 1;
  f.dotOps = 1;
  f.broadcastOps = 5;
  f.expandDimsOps = 4;
  f.splatOps = 9;
  f.addPtrOps = 7;
  f.arithOps = 9;
  f.addOps = 2;
  f.mulOps = 5;
  f.scalarOps = 7;
  f.maxTensorRank = 2;
  f.maxTensorNumel = 256;
  f.maxElementBits = 32;
  f.pointerTensorOps = 11;
  f.pointerUnstructuredDims = 18;
  f.laneDependentPointerOps = 9;
  f.vectorPtrSplatOps = 4;
  f.loadBytes = 1088;
  f.storeBytes = 1024;
  f.loadWarpInstructions = 17;
  f.storeWarpInstructions = 8;
  f.dotFlops = 8192;
  f.dotOutputElements = 256;
  f.dotMNK.push_back({16, 16, 16});
  f.hasDot = true;
  f.observedMixedKinds.push_back("conditional_indirect_memory");

  f.weightedOps["add"] = 2;
  f.weightedOps["mul"] = 5;
  f.weightedOps["load"] = 3;
  f.weightedOps["store"] = 1;
  f.opElements["add"] = 32;
  f.opElements["mul"] = 50;
  f.opElements["load"] = 528;
  f.opElements["store"] = 256;
  return f;
}

SimdSimtFeatureSummary fbgemmFeatures() {
  SimdSimtFeatureSummary f;
  f.loadOps = 7;
  f.storeOps = 2;
  f.reduceOps = 1;
  f.broadcastOps = 6;
  f.expandDimsOps = 6;
  f.splatOps = 10;
  f.addPtrOps = 12;
  f.arithOps = 29;
  f.mathOps = 2;
  f.addOps = 1;
  f.mulOps = 7;
  f.divOps = 2;
  f.maxOps = 2;
  f.absOps = 2;
  f.cmpOps = 1;
  f.castOps = 2;
  f.clampOps = 2;
  f.scalarOps = 19;
  f.maxTensorRank = 2;
  f.maxTensorNumel = 64;
  f.maxElementBits = 64;
  f.maskTensorOps = 12;
  f.maskRankSum = 25;
  f.maskBroadcastOps = 4;
  f.pointerTensorOps = 19;
  f.pointerUnstructuredDims = 20;
  f.laneDependentPointerOps = 10;
  f.rowLocalReduceOps = 1;
  f.scalarLoadOps = 2;
  f.vectorPtrSplatOps = 6;
  f.loadBytes = 4152;
  f.storeBytes = 2064;
  f.loadWarpInstructions = 37;
  f.storeWarpInstructions = 17;
  f.staticLoopCount = 2;
  f.staticLoopTripCountSum = 16;
  f.staticLoopTripCountMax = 8;
  f.hasControlFlow = true;
  f.observedMixedKinds.push_back("conditional_indirect_memory");

  f.weightedOps["abs"] = 9;
  f.weightedOps["add"] = 1;
  f.weightedOps["cast"] = 2;
  f.weightedOps["clamp"] = 9;
  f.weightedOps["cmp"] = 1;
  f.weightedOps["div"] = 2;
  f.weightedOps["load"] = 21;
  f.weightedOps["max"] = 16;
  f.weightedOps["mul"] = 14;
  f.weightedOps["reduce"] = 8;
  f.weightedOps["store"] = 9;
  f.opElements["abs"] = 516;
  f.opElements["add"] = 4;
  f.opElements["cast"] = 8;
  f.opElements["clamp"] = 516;
  f.opElements["cmp"] = 4;
  f.opElements["div"] = 8;
  f.opElements["load"] = 1038;
  f.opElements["max"] = 40;
  f.opElements["mul"] = 533;
  f.opElements["reduce"] = 512;
  f.opElements["store"] = 516;
  return f;
}

SimdSimtFeatureSummary outOfCoverageFeatures() {
  SimdSimtFeatureSummary f = gatherDotFeatures();
  // One FLOP above the profile's tiny-dot coverage ceiling.  Since dotFlops is
  // non-zero, neither reduction-only coverage domain can admit this feature.
  f.dotFlops = 16385;
  return f;
}

SimdSimtFeatureSummary rank1IndirectVectorReductionFeatures() {
  SimdSimtFeatureSummary f;
  f.reduceOps = 1;
  f.maxTensorRank = 1;
  f.maxTensorNumel = 256;
  f.maxElementBits = 32;
  f.rank1IndirectVectorReduce = true;
  f.weightedOps["reduce"] = 8;
  return f;
}

SimdSimtFeatureSummary solveTrilBt16Features() {
  SimdSimtFeatureSummary f;
  f.reduceOps = 1;
  f.maxTensorRank = 2;
  f.maxTensorNumel = 256;
  f.maxElementBits = 64;
  f.maskRankSum = 20;
  f.pointerTensorOps = 5;
  f.laneDependentPointerOps = 2;
  f.staticLoopCount = 1;
  f.staticLoopTripCountSum = 1;
  f.staticLoopTripCountMax = 1;
  f.hasControlFlow = true;
  f.weightedOps["reduce"] = 1;
  return f;
}

SimdSimtFeatureSummary triangularUnknownLoopFeatures() {
  SimdSimtFeatureSummary f = solveTrilBt16Features();
  // BT64 has four sibling 16x16 recurrences. Their TTIR upper bounds remain
  // dynamic, but the full-tile structural estimate is 14 trips per loop.
  f.staticLoopCount = 4;
  f.staticLoopTripCountSum = 56;
  f.staticLoopTripCountMax = 14;
  f.modeledDynamicLoopCount = 4;
  f.modeledDynamicLoopTripCountSum = 56;
  f.hasUnknownTripCount = true;
  f.maskRankSum = 62;
  f.weightedOps["reduce"] = 56;
  f.shuffleLaneSteps = 57344;
  f.predicateLaneEvaluations = 35840;
  f.simtAnchors.count = 1;
  f.simtAnchors.recognizedCount = 1;
  f.simtAnchors.reduceOps = 4;
  f.simtAnchors.maxTensorNumel = 256;
  f.simtAnchors.maskRankSum = 62;
  f.simtAnchors.staticLoopCount = 4;
  f.simtAnchors.staticLoopTripCountSum = 56;
  f.simtAnchors.modeledDynamicLoopCount = 4;
  f.simtAnchors.modeledDynamicLoopTripCountSum = 56;
  f.simtAnchors.weightedOps["reduce"] = 56;
  f.simtAnchors.shuffleLaneSteps = 57344;
  f.simtAnchors.predicateLaneEvaluations = 35840;
  f.simtAnchors.mechanismKinds.push_back("triangular_solve_loop");
  return f;
}

} // namespace

TEST(SimdSimtCostModelTest,
     GatherDotGoldenScoresRequireMaterializableMixedPlan) {
  auto modelOptions = options(32);
  modelOptions.scoreOutsideCalibrationCoverage = false;
  auto report = estimateSimdSimtCandidates(gatherDotFeatures(), modelOptions);
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_TRUE(report->candidateCostsEvaluated);
  EXPECT_NEAR(report->breakdown.simdStructuralPenaltyCycles,
              report->breakdown.simdAnalyticalCycles *
                  report->breakdown.structuralPenaltyRatio,
              1.0e-6);
  EXPECT_NEAR(report->uncalibratedCandidateCosts.allSimd,
              report->breakdown.simdAnalyticalCycles +
                  report->breakdown.simdStructuralPenaltyCycles,
              1.0e-6);
  EXPECT_NE(report->uncalibratedCandidateCosts.allSimd,
            report->breakdown.simtAnalyticalCycles *
                (1.0 + report->breakdown.structuralPenaltyRatio));
  EXPECT_NEAR(report->candidateCosts.allSimtOnly, 2389.91061977427, 1.0e-6);
  EXPECT_DOUBLE_EQ(report->uncalibratedCandidateCosts.mixedSimdSimt,
                   std::max(report->uncalibratedCandidateCosts.allSimd,
                            report->uncalibratedCandidateCosts.allSimtOnly) +
                       223.0);
  EXPECT_NEAR(report->candidateCosts.mixedSimdSimt,
              report->uncalibratedCandidateCosts.mixedSimdSimt * 0.666478,
              1.0e-6);
  EXPECT_TRUE(report->eventRouteCalibrationApplied);
  EXPECT_FALSE(report->eventAllSimtOnlyValidated);
  EXPECT_TRUE(report->eventMixedSimdSimtValidated);
  EXPECT_DOUBLE_EQ(report->eventRouteScoreMultipliers.mixedSimdSimt, 0.666478);
  EXPECT_FALSE(report->mixedCandidateLegal);
  EXPECT_TRUE(report->applicability.mechanismDetected);
  EXPECT_FALSE(report->applicability.materializable);
  EXPECT_DOUBLE_EQ(report->breakdown.standaloneSimtSetupCycles, 141.0);
  EXPECT_DOUBLE_EQ(report->breakdown.mixedSetupFallbackCycles, 223.0);
  EXPECT_DOUBLE_EQ(report->breakdown.setupProxyDeltaCycles, 82.0);
  EXPECT_EQ(report->decision, SimdSimtCandidateKind::AllSIMD);
  EXPECT_TRUE(report->selectionScoreValid);
  EXPECT_EQ(report->calibrationDomain, "tiny_irregular_dot");
  EXPECT_EQ(report->rankingConfidence, "low");
  EXPECT_TRUE(report->gatePassed);
  EXPECT_TRUE(report->gateReasons.empty());
  EXPECT_EQ(report->schemaVersion, 10);
  EXPECT_EQ(report->profileVersion, "david-v100-simd-simt-20260806-v11");
  EXPECT_FALSE(report->selectionProfileContentSha256.empty());
  EXPECT_EQ(report->microbenchmarkProfileVersion,
            "david-v100-shared-microbench-20260730-v2");
  EXPECT_EQ(report->microbenchmarkProfileTarget, "Ascend950PR/dav-c310");
  EXPECT_FALSE(report->microbenchmarkProfileContentSha256.empty());
  EXPECT_NE(report->profileContentSha256,
            report->selectionProfileContentSha256);
}

TEST(SimdSimtCostModelTest, FbgemmGoldenScoresRequireMaterializableMixedPlan) {
  auto modelOptions = options(4);
  modelOptions.scoreOutsideCalibrationCoverage = false;
  auto report = estimateSimdSimtCandidates(fbgemmFeatures(), modelOptions);
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_TRUE(report->candidateCostsEvaluated);
  EXPECT_NEAR(report->uncalibratedCandidateCosts.allSimd,
              report->breakdown.simdAnalyticalCycles *
                  (1.0 + report->breakdown.structuralPenaltyRatio),
              1.0e-6);
  EXPECT_NEAR(report->uncalibratedCandidateCosts.allSimtOnly,
              14311.016566489482, 1.0e-6);
  EXPECT_DOUBLE_EQ(report->uncalibratedCandidateCosts.mixedSimdSimt,
                   std::max(report->uncalibratedCandidateCosts.allSimd,
                            report->uncalibratedCandidateCosts.allSimtOnly) +
                       182.0);
  EXPECT_NEAR(report->candidateCosts.allSimd,
              report->uncalibratedCandidateCosts.allSimd *
                  report->eventRouteScoreMultipliers.allSimd,
              1.0e-6);
  EXPECT_NEAR(report->candidateCosts.allSimtOnly,
              report->uncalibratedCandidateCosts.allSimtOnly * 0.807671,
              1.0e-6);
  EXPECT_NEAR(report->candidateCosts.mixedSimdSimt,
              report->uncalibratedCandidateCosts.mixedSimdSimt * 3.537276,
              1.0e-6);
  EXPECT_TRUE(report->eventRouteCalibrationApplied);
  EXPECT_TRUE(report->eventAllSimtOnlyValidated);
  EXPECT_TRUE(report->eventMixedSimdSimtValidated);
  EXPECT_FALSE(report->mixedCandidateLegal);
  EXPECT_TRUE(report->applicability.mechanismDetected);
  EXPECT_FALSE(report->applicability.materializable);
  EXPECT_DOUBLE_EQ(report->breakdown.standaloneSimtSetupCycles, 141.0);
  EXPECT_DOUBLE_EQ(report->breakdown.mixedSetupFallbackCycles, 182.0);
  EXPECT_DOUBLE_EQ(report->breakdown.setupProxyDeltaCycles, 41.0);
  EXPECT_EQ(report->decision, SimdSimtCandidateKind::AllSIMTOnly);
  EXPECT_TRUE(report->selectionScoreValid);
  EXPECT_EQ(report->calibrationDomain, "masked_rowwise_reduction");
  EXPECT_EQ(report->rankingConfidence, "low");
  EXPECT_TRUE(report->gatePassed);
  EXPECT_TRUE(report->gateReasons.empty());
}

TEST(SimdSimtCostModelTest, Rank1IndirectVectorReductionIsCovered) {
  auto modelOptions = options(4);
  modelOptions.scoreOutsideCalibrationCoverage = false;
  auto report = estimateSimdSimtCandidates(
      rank1IndirectVectorReductionFeatures(), modelOptions);
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_TRUE(report->calibrationCovered);
  EXPECT_EQ(report->calibrationDomain, "rank1_indirect_vector_reduction");
  EXPECT_TRUE(report->selectionScoreValid);
  EXPECT_TRUE(report->candidateCostsEvaluated);
}

TEST(SimdSimtCostModelTest, SolveTrilBt16StaysOutsideMaskedRowwiseCalibration) {
  auto modelOptions = options(32);
  modelOptions.scoreOutsideCalibrationCoverage = false;
  auto report =
      estimateSimdSimtCandidates(solveTrilBt16Features(), modelOptions);
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_FALSE(report->calibrationCovered);
  EXPECT_EQ(report->calibrationDomain, "out_of_calibration_domain");
  EXPECT_FALSE(report->selectionScoreValid);
  EXPECT_FALSE(report->candidateCostsEvaluated);
  EXPECT_FALSE(report->gatePassed);
  ASSERT_EQ(report->gateReasons.size(), 1u);
  EXPECT_EQ(report->gateReasons.front(), "selection_score_invalid");
}

TEST(SimdSimtCostModelTest,
     TriangularUnknownLoopUsesBoundedCalibrationException) {
  auto modelOptions = options(32);
  modelOptions.scoreOutsideCalibrationCoverage = false;
  auto report =
      estimateSimdSimtCandidates(triangularUnknownLoopFeatures(), modelOptions);
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_TRUE(report->calibrationCovered);
  EXPECT_EQ(report->calibrationDomain, "triangular_solve_loop");
  EXPECT_TRUE(report->selectionScoreValid);
  EXPECT_TRUE(report->candidateCostsEvaluated);
  EXPECT_NEAR(report->breakdown.simtIssuePayloadCycles,
              report->breakdown.simtComputeCycles +
                  report->breakdown.simtShuffleCycles +
                  report->breakdown.simtDotCycles +
                  report->breakdown.simtMemoryCycles +
                  report->breakdown.simtPredicateCycles,
              1.0e-6);
  EXPECT_NEAR(report->breakdown.mixedSimtAnchorPayloadCycles,
              report->breakdown.mixedSimtAnchorComputeCycles +
                  report->breakdown.mixedSimtAnchorDotCycles +
                  report->breakdown.mixedSimtAnchorShuffleCycles +
                  report->breakdown.mixedSimtAnchorMemoryCycles +
                  report->breakdown.mixedSimtAnchorPredicateCycles,
              1.0e-6);
  EXPECT_GT(report->breakdown.mixedSimtAnchorShuffleCycles, 0.0);
  EXPECT_GT(report->breakdown.mixedSimtAnchorPredicateCycles, 0.0);
}

TEST(SimdSimtCostModelTest,
     UnknownLoopWithoutTriangularEvidenceRemainsRejected) {
  auto features = triangularUnknownLoopFeatures();
  features.simtAnchors.mechanismKinds.clear();
  auto modelOptions = options(32);
  modelOptions.scoreOutsideCalibrationCoverage = false;
  auto report = estimateSimdSimtCandidates(features, modelOptions);
  if (!report)
    FAIL() << llvm::toString(report.takeError());

  EXPECT_FALSE(report->calibrationCovered);
  EXPECT_EQ(report->calibrationDomain, "unknown_loop_trip_count");
  EXPECT_FALSE(report->selectionScoreValid);
  EXPECT_FALSE(report->candidateCostsEvaluated);
}

TEST(SimdSimtCostModelTest,
     TriangularUnknownLoopStillHonorsAnchorLoopAndShapeBounds) {
  auto modelOptions = options(32);
  modelOptions.scoreOutsideCalibrationCoverage = false;

  auto tooFewAnchors = triangularUnknownLoopFeatures();
  tooFewAnchors.simtAnchors.count = 0;
  auto fewAnchorReport =
      estimateSimdSimtCandidates(tooFewAnchors, modelOptions);
  if (!fewAnchorReport)
    FAIL() << llvm::toString(fewAnchorReport.takeError());
  EXPECT_FALSE(fewAnchorReport->calibrationCovered);
  EXPECT_EQ(fewAnchorReport->calibrationDomain, "unknown_loop_trip_count");

  auto tooManyLoops = triangularUnknownLoopFeatures();
  tooManyLoops.simtAnchors.staticLoopCount = 5;
  auto loopReport = estimateSimdSimtCandidates(tooManyLoops, modelOptions);
  if (!loopReport)
    FAIL() << llvm::toString(loopReport.takeError());
  EXPECT_FALSE(loopReport->calibrationCovered);
  EXPECT_EQ(loopReport->calibrationDomain, "unknown_loop_trip_count");

  auto oversized = triangularUnknownLoopFeatures();
  oversized.maxTensorNumel = 257;
  auto oversizedReport = estimateSimdSimtCandidates(oversized, modelOptions);
  if (!oversizedReport)
    FAIL() << llvm::toString(oversizedReport.takeError());
  EXPECT_FALSE(oversizedReport->calibrationCovered);
  EXPECT_EQ(oversizedReport->calibrationDomain, "unknown_loop_trip_count");
}

TEST(SimdSimtCostModelTest, OutOfCoverageAutoSkipsButDiagnosticsStillScore) {
  auto autoOptions = options(32);
  autoOptions.scoreOutsideCalibrationCoverage = false;
  auto skipped =
      estimateSimdSimtCandidates(outOfCoverageFeatures(), autoOptions);
  if (!skipped)
    FAIL() << llvm::toString(skipped.takeError());

  EXPECT_FALSE(skipped->calibrationCovered);
  EXPECT_EQ(skipped->calibrationDomain, "out_of_calibration_domain");
  EXPECT_FALSE(skipped->selectionScoreValid);
  EXPECT_FALSE(skipped->candidateCostsEvaluated);
  EXPECT_FALSE(skipped->gatePassed);
  ASSERT_EQ(skipped->gateReasons.size(), 1u);
  EXPECT_EQ(skipped->gateReasons.front(), "selection_score_invalid");
  EXPECT_DOUBLE_EQ(skipped->candidateCosts.allSimd, 0.0);
  EXPECT_DOUBLE_EQ(skipped->candidateCosts.allSimtOnly, 0.0);
  EXPECT_DOUBLE_EQ(skipped->candidateCosts.mixedSimdSimt, 0.0);

  auto diagnosticOptions = options(32);
  diagnosticOptions.scoreOutsideCalibrationCoverage = true;
  auto diagnostic =
      estimateSimdSimtCandidates(outOfCoverageFeatures(), diagnosticOptions);
  if (!diagnostic)
    FAIL() << llvm::toString(diagnostic.takeError());

  EXPECT_FALSE(diagnostic->calibrationCovered);
  EXPECT_FALSE(diagnostic->selectionScoreValid);
  EXPECT_TRUE(diagnostic->candidateCostsEvaluated);
  EXPECT_GT(diagnostic->candidateCosts.allSimd, 0.0);
  EXPECT_GT(diagnostic->candidateCosts.allSimtOnly, 0.0);
  EXPECT_GT(diagnostic->candidateCosts.mixedSimdSimt, 0.0);
  EXPECT_FALSE(diagnostic->gatePassed);
}
