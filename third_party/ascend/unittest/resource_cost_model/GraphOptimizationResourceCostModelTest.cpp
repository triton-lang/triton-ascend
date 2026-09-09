#include "TritonToGraph/ProgramGridTransform.h"
#include "TritonToGraph/ResourceCostModel.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/Parser/Parser.h"

#include <gtest/gtest.h>

#include <limits>
#include <string>

using namespace mlir;
using namespace mlir::triton::cfg;

namespace {

OwningOpRef<ModuleOp> parseModule(MLIRContext &context, llvm::StringRef text) {
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<func::FuncDialect>();
  context.getOrLoadDialect<scf::SCFDialect>();
  context.getOrLoadDialect<tensor::TensorDialect>();
  return parseSourceString<ModuleOp>(text, &context);
}

CandidateCost makeCandidate(const CandidatePlan &plan = {}) {
  CandidateCost candidate;
  candidate.plan = plan;
  candidate.plan.stableId =
      candidate.plan.stableId.empty() ? "candidate" : candidate.plan.stableId;
  candidate.logicalTasksBefore = 256;
  candidate.logicalTasksAfter = 256;
  candidate.actualProgramsBefore = 256;
  candidate.actualProgramsAfter = 64;
  candidate.launchesBefore = 1;
  candidate.launchesAfter = 1;
  candidate.gmReadBytesBefore = 4096;
  candidate.gmReadBytesAfter = 2048;
  candidate.gmWriteBytesBefore = 4096;
  candidate.gmWriteBytesAfter = 2048;
  candidate.storeCountBefore = 8;
  candidate.storeCountAfter = 4;
  candidate.addressCalculationsBefore = 128;
  candidate.addressCalculationsAfter = 64;
  candidate.baselinePeakLiveBytes = 256;
  candidate.estimatedPeakLiveBytes = 512;
  candidate.hasPeakLiveBytes = true;
  return candidate;
}

ResourceSnapshot knownResources() {
  return ResourceSnapshot::fromExplicit(/*ubCapacityBytes=*/4096,
                                        /*deviceCoreCount=*/8,
                                        /*minProgramsPerCore=*/2,
                                        /*ubSafetyPercent=*/80);
}

} // namespace

TEST(GraphOptimizationResourceCostModelTest,
     SumsSimultaneouslyLiveStaticTensorsFromIR) {
  MLIRContext context;
  auto module = parseModule(context, R"mlir(
module {
  func.func @simultaneous() {
    %a = tensor.empty() : tensor<16xf32>
    %b = tensor.empty() : tensor<16xf32>
    %sum = arith.addf %a, %b : tensor<16xf32>
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  LiveByteEstimate estimate = estimatePeakLiveBytes(module->getOperation());
  ASSERT_TRUE(estimate.known);
  EXPECT_EQ(estimate.peakLiveBytes, 3 * 16 * sizeof(float));
}

TEST(GraphOptimizationResourceCostModelTest,
     HandlesLoopCarrierOnceAndExtendsInvariantLiveRange) {
  MLIRContext context;
  auto module = parseModule(context, R"mlir(
module {
  func.func @loop_once() {
    %c0 = arith.constant 0 : index
    %c4 = arith.constant 4 : index
    %c1 = arith.constant 1 : index
    %init = tensor.empty() : tensor<4xf32>
    %invariant = tensor.empty() : tensor<4xf32>
    %result = scf.for %i = %c0 to %c4 step %c1 iter_args(%acc = %init) -> (tensor<4xf32>) {
      %first = arith.addf %acc, %invariant : tensor<4xf32>
      %tail = tensor.empty() : tensor<4xf32>
      %next = arith.addf %first, %tail : tensor<4xf32>
      scf.yield %next : tensor<4xf32>
    }
    return
  }
}
)mlir");
  ASSERT_TRUE(module);

  LiveByteEstimate estimate = estimatePeakLiveBytes(module->getOperation());
  ASSERT_TRUE(estimate.known);
  // The loop body is evaluated once for liveness: carrier, invariant, first,
  // tail, and next coexist at its peak.  It must not grow with the trip count.
  EXPECT_EQ(estimate.peakLiveBytes, 5 * 4 * sizeof(float));
}

TEST(GraphOptimizationResourceCostModelTest,
     SupportsBF16FP16FP32AndRejectsDynamicOrOverflowShapes) {
  MLIRContext context;
  const auto bf16Bytes = getStaticTensorBytes(
      RankedTensorType::get({16}, BFloat16Type::get(&context)));
  const auto fp16Bytes = getStaticTensorBytes(
      RankedTensorType::get({16}, Float16Type::get(&context)));
  const auto fp32Bytes = getStaticTensorBytes(
      RankedTensorType::get({16}, Float32Type::get(&context)));
  const auto zeroBytes = getStaticTensorBytes(
      RankedTensorType::get({0}, Float32Type::get(&context)));
  ASSERT_TRUE(bf16Bytes);
  ASSERT_TRUE(fp16Bytes);
  ASSERT_TRUE(fp32Bytes);
  ASSERT_TRUE(zeroBytes);
  EXPECT_EQ(*bf16Bytes, 32u);
  EXPECT_EQ(*fp16Bytes, 32u);
  EXPECT_EQ(*fp32Bytes, 64u);
  EXPECT_EQ(*zeroBytes, 0u);
  EXPECT_FALSE(getStaticTensorBytes(RankedTensorType::get(
      {ShapedType::kDynamic}, Float32Type::get(&context))));
  EXPECT_FALSE(getStaticTensorBytes(RankedTensorType::get(
      {std::numeric_limits<int64_t>::max(), 2}, Float32Type::get(&context))));
}

TEST(GraphOptimizationResourceCostModelTest,
     RejectsUnknownUBOverflowAndInsufficientParallelism) {
  CandidateCost candidate = makeCandidate();
  EXPECT_EQ(evaluateCandidateCost({}, candidate).reason,
            ResourceCostRejectReason::UnknownResource);

  CandidateCost overflow = candidate;
  overflow.estimatedPeakLiveBytes = 4096;
  EXPECT_EQ(evaluateCandidateCost(knownResources(), overflow).reason,
            ResourceCostRejectReason::UBOverflow);

  CandidateCost parallel = candidate;
  parallel.actualProgramsAfter = 15;
  EXPECT_EQ(evaluateCandidateCost(knownResources(), parallel).reason,
            ResourceCostRejectReason::InsufficientParallelism);

  CandidateCost parallelAndOverflow = overflow;
  parallelAndOverflow.actualProgramsAfter = 15;
  EXPECT_EQ(evaluateCandidateCost(knownResources(), parallelAndOverflow).reason,
            ResourceCostRejectReason::InsufficientParallelism);

  CandidateCost arithmeticOverflow = candidate;
  arithmeticOverflow.gmReadBytesBefore = std::numeric_limits<uint64_t>::max();
  arithmeticOverflow.gmReadBytesAfter = 0;
  EXPECT_EQ(evaluateCandidateCost(knownResources(), arithmeticOverflow).reason,
            ResourceCostRejectReason::Overflow);
}

TEST(GraphOptimizationResourceCostModelTest,
     MergeSplitSubCorePolicyIsNarrowAndDoesNotRelaxDefaultParallelism) {
  CandidateCost candidate = makeCandidate();
  candidate.logicalTasksAfter = 8;
  candidate.actualProgramsAfter = 8;

  // The ordinary min-programs-per-core contract remains 16 on this fixture.
  EXPECT_EQ(evaluateCandidateCost(knownResources(), candidate).reason,
            ResourceCostRejectReason::InsufficientParallelism);

  candidate.parallelismPolicy =
      ParallelismPolicy::MergeSplitSmallGridAllowSubCore;
  CandidateEvaluation allowed =
      evaluateCandidateCost(knownResources(), candidate);
  ASSERT_TRUE(allowed.accepted);
  EXPECT_EQ(allowed.requiredParallelPrograms, 16u);

  CandidateCost persistent = candidate;
  persistent.persistent = true;
  EXPECT_EQ(evaluateCandidateCost(knownResources(), persistent).reason,
            ResourceCostRejectReason::InvalidCandidate);

  CandidateCost capped = candidate;
  capped.actualProgramsAfter = 7;
  EXPECT_EQ(evaluateCandidateCost(knownResources(), capped).reason,
            ResourceCostRejectReason::InvalidCandidate);

  CandidateCost overCore = candidate;
  overCore.logicalTasksAfter = 9;
  overCore.actualProgramsAfter = 9;
  EXPECT_EQ(evaluateCandidateCost(knownResources(), overCore).reason,
            ResourceCostRejectReason::InvalidCandidate);
}

TEST(GraphOptimizationResourceCostModelTest,
     PersistentCandidatesPriceLogicalTasksAndActualProgramsSeparately) {
  CandidateCost candidate = makeCandidate();
  candidate.plan = CandidatePlan{2, 4, 1, "persistent", 0};
  candidate.persistent = true;
  candidate.logicalTasksBefore = 4096;
  candidate.logicalTasksAfter = 1024;
  candidate.actualProgramsBefore = 4096;
  candidate.actualProgramsAfter = 32;

  CandidateEvaluation evaluation =
      evaluateCandidateCost(knownResources(), candidate);
  ASSERT_TRUE(evaluation.accepted);
  EXPECT_EQ(evaluation.effectiveWorkPerProgram, 32u);
  EXPECT_NE(evaluation.remark.find("logical_programs=4096->1024"),
            std::string::npos);
  EXPECT_NE(evaluation.remark.find("physical_blocks=4096->32"),
            std::string::npos);
}

TEST(
    GraphOptimizationResourceCostModelTest,
    LauncherProjectionKeepsExample1NonpersistentAndExample2PersistentSemantics) {
  const ResourceSnapshot resources = ResourceSnapshot::fromExplicit(
      /*ubCapacityBytes=*/256 * 1024, /*deviceCoreCount=*/56);
  ProgramGridSpecialization example1;
  example1.grid = {8, 64, 1};

  auto baseline = projectProgramMappingLaunch(example1, {}, resources);
  ASSERT_TRUE(baseline);
  EXPECT_EQ(baseline->logicalGrid, (std::array<uint64_t, 3>{8, 64, 1}));
  EXPECT_EQ(baseline->logicalPrograms, 512u);
  EXPECT_EQ(baseline->physicalPrograms, 56u);
  EXPECT_EQ(baseline->physicalWaves, 10u);
  EXPECT_TRUE(baseline->legacyAutoMap);

  const ProgramGridTransform factor2 = {0,
                                        1,
                                        2,
                                        64,
                                        /*persistentCoverage=*/false,
                                        /*gridStrideAbiVerified=*/false};
  auto factor2Projection =
      projectProgramMappingLaunch(example1, {factor2}, resources);
  ASSERT_TRUE(factor2Projection);
  EXPECT_EQ(factor2Projection->logicalGrid,
            (std::array<uint64_t, 3>{8, 32, 1}));
  EXPECT_EQ(factor2Projection->logicalPrograms, 256u);
  EXPECT_EQ(factor2Projection->physicalPrograms, 256u);
  EXPECT_FALSE(factor2Projection->legacyAutoMap);

  const ProgramGridTransform factor8 = {0,
                                        1,
                                        8,
                                        64,
                                        /*persistentCoverage=*/false,
                                        /*gridStrideAbiVerified=*/false};
  auto factor8Projection =
      projectProgramMappingLaunch(example1, {factor8}, resources);
  ASSERT_TRUE(factor8Projection);
  EXPECT_EQ(factor8Projection->logicalGrid, (std::array<uint64_t, 3>{8, 8, 1}));
  EXPECT_EQ(factor8Projection->physicalPrograms, 64u);

  const ProgramGridTransform factor16 = {0,
                                         1,
                                         16,
                                         64,
                                         /*persistentCoverage=*/false,
                                         /*gridStrideAbiVerified=*/false};
  auto factor16Projection =
      projectProgramMappingLaunch(example1, {factor16}, resources);
  ASSERT_TRUE(factor16Projection);
  EXPECT_EQ(factor16Projection->logicalGrid,
            (std::array<uint64_t, 3>{8, 4, 1}));
  EXPECT_EQ(factor16Projection->physicalPrograms, 32u);
  EXPECT_FALSE(factor16Projection->persistentCoverage);
  EXPECT_FALSE(factor16Projection->legacyAutoMap);

  const ProgramGridTransform factor32 = {0,
                                         1,
                                         32,
                                         64,
                                         /*persistentCoverage=*/false,
                                         /*gridStrideAbiVerified=*/false};
  auto factor32Projection =
      projectProgramMappingLaunch(example1, {factor32}, resources);
  ASSERT_TRUE(factor32Projection);
  EXPECT_EQ(factor32Projection->logicalGrid,
            (std::array<uint64_t, 3>{8, 2, 1}));
  EXPECT_EQ(factor32Projection->physicalPrograms, 16u);
  EXPECT_FALSE(factor32Projection->persistentCoverage);
  EXPECT_FALSE(factor32Projection->legacyAutoMap);

  const ProgramGridTransform factor4 = {0,
                                        1,
                                        4,
                                        64,
                                        /*persistentCoverage=*/false,
                                        /*gridStrideAbiVerified=*/false};
  auto factor4Projection =
      projectProgramMappingLaunch(example1, {factor4}, resources);
  ASSERT_TRUE(factor4Projection);
  EXPECT_EQ(factor4Projection->logicalGrid,
            (std::array<uint64_t, 3>{8, 16, 1}));
  EXPECT_EQ(factor4Projection->physicalPrograms, 128u);

  ProgramGridSpecialization q;
  q.grid = {4096, 16, 1};
  const std::array<ProgramGridTransform, 2> qTransforms = {{
      {0, 1, 16, 16, /*persistentCoverage=*/false,
       /*gridStrideAbiVerified=*/false},
      {1, 0, 4, 4096, /*persistentCoverage=*/true,
       /*gridStrideAbiVerified=*/true},
  }};
  auto qProjection = projectProgramMappingLaunch(q, qTransforms, resources);
  ASSERT_TRUE(qProjection);
  EXPECT_EQ(qProjection->logicalGrid, (std::array<uint64_t, 3>{1024, 1, 1}));
  EXPECT_EQ(qProjection->physicalGrid, (std::array<uint64_t, 3>{56, 1, 1}));
  EXPECT_EQ(qProjection->physicalPrograms, 56u);
  EXPECT_EQ(qProjection->physicalWaves, 19u);

  ProgramGridSpecialization k;
  k.grid = {4096, 1, 1};
  const ProgramGridTransform block64 = {0,
                                        0,
                                        64,
                                        4096,
                                        /*persistentCoverage=*/true,
                                        /*gridStrideAbiVerified=*/true};
  auto k64Projection = projectProgramMappingLaunch(k, {block64}, resources);
  ASSERT_TRUE(k64Projection);
  EXPECT_EQ(k64Projection->logicalGrid, (std::array<uint64_t, 3>{64, 1, 1}));
  EXPECT_EQ(k64Projection->physicalGrid, (std::array<uint64_t, 3>{56, 1, 1}));
  EXPECT_EQ(k64Projection->physicalWaves, 2u);
}

TEST(GraphOptimizationResourceCostModelTest,
     QHeadPackingWinsEqualRowCandidateAndK128FailsPhysicalParallelism) {
  const ResourceSnapshot resources = ResourceSnapshot::fromExplicit(
      /*ubCapacityBytes=*/256 * 1024, /*deviceCoreCount=*/56);
  CandidateCost q16x4 = makeCandidate({16, 4, 1, "q16x4", 0});
  q16x4.persistent = true;
  q16x4.logicalTasksBefore = 4096 * 16;
  q16x4.logicalTasksAfter = 1024;
  q16x4.actualProgramsBefore = 56;
  q16x4.actualProgramsAfter = 56;
  q16x4.physicalWavesBefore = 1171;
  q16x4.physicalWavesAfter = 19;
  q16x4.persistentLoopTripsBefore = 1171;
  q16x4.persistentLoopTripsAfter = 19;
  q16x4.workPerProgramBefore = 1171;
  q16x4.workPerProgramAfter = 19;
  q16x4.baselinePeakLiveBytes = 0;
  q16x4.estimatedPeakLiveBytes = 196608;
  q16x4.tokenOnlyRepeatedBytesBefore = 15ull * 4096 * 256 * sizeof(float);
  q16x4.tokenOnlyRepeatedBytesAfter = 0;

  CandidateCost q8x8 = q16x4;
  q8x8.plan = {8, 8, 1, "q8x8", 1};
  q8x8.tokenOnlyRepeatedBytesAfter = 4096ull * 256 * sizeof(float);
  const CandidateEvaluation q16Evaluation =
      evaluateCandidateCost(resources, q16x4);
  const CandidateEvaluation q8Evaluation =
      evaluateCandidateCost(resources, q8x8);
  ASSERT_TRUE(q16Evaluation.accepted);
  ASSERT_TRUE(q8Evaluation.accepted);
  EXPECT_GT(q16Evaluation.benefitScore, q8Evaluation.benefitScore);
  EXPECT_EQ(q16Evaluation.persistentLoopTripsAfter,
            q8Evaluation.persistentLoopTripsAfter);

  CandidateCost k64 = makeCandidate({1, 64, 1, "k64", 0});
  k64.persistent = true;
  k64.logicalTasksBefore = 4096;
  k64.logicalTasksAfter = 64;
  k64.actualProgramsBefore = 56;
  k64.actualProgramsAfter = 56;
  k64.physicalWavesBefore = 74;
  k64.physicalWavesAfter = 2;
  k64.persistentLoopTripsBefore = 74;
  k64.persistentLoopTripsAfter = 2;
  k64.workPerProgramBefore = 74;
  k64.workPerProgramAfter = 2;
  k64.baselinePeakLiveBytes = 0;
  k64.estimatedPeakLiveBytes = 196608;
  const CandidateEvaluation k64Evaluation =
      evaluateCandidateCost(resources, k64);
  ASSERT_TRUE(k64Evaluation.accepted);

  CandidateCost k128 = k64;
  k128.plan = {1, 128, 1, "k128", 1};
  k128.logicalTasksAfter = 32;
  k128.actualProgramsAfter = 32;
  k128.physicalWavesAfter = 1;
  k128.persistentLoopTripsAfter = 1;
  k128.workPerProgramAfter = 1;
  const CandidateEvaluation k128Evaluation =
      evaluateCandidateCost(resources, k128);
  EXPECT_FALSE(k128Evaluation.accepted);
  EXPECT_EQ(k128Evaluation.reason,
            ResourceCostRejectReason::InsufficientParallelism);
}

TEST(GraphOptimizationResourceCostModelTest,
     EnumeratesAndOrdersCandidatesDeterministicallyWithCompleteRemarks) {
  auto first = enumerateCandidatePlans({4, 1, 2, 2}, {2, 1}, {2, 1}, "plan");
  auto second = enumerateCandidatePlans({4, 1, 2, 2}, {2, 1}, {2, 1}, "plan");
  ASSERT_EQ(first.size(), 12u);
  ASSERT_EQ(first.size(), second.size());
  for (size_t index = 0; index < first.size(); ++index) {
    EXPECT_EQ(first[index].stableId, second[index].stableId);
    EXPECT_EQ(first[index].tensorizeFactor, second[index].tensorizeFactor);
    EXPECT_EQ(first[index].blockT, second[index].blockT);
    EXPECT_EQ(first[index].staticAxisFusionFactor,
              second[index].staticAxisFusionFactor);
  }

  CandidateCost lhs = makeCandidate(first.back());
  CandidateCost rhs = makeCandidate(first.front());
  lhs.gmReadBytesAfter = rhs.gmReadBytesAfter;
  lhs.gmWriteBytesAfter = rhs.gmWriteBytesAfter;
  lhs.actualProgramsAfter = rhs.actualProgramsAfter;
  lhs.storeCountAfter = rhs.storeCountAfter;
  lhs.addressCalculationsAfter = rhs.addressCalculationsAfter;
  lhs.estimatedPeakLiveBytes = rhs.estimatedPeakLiveBytes;
  llvm::SmallVector<CandidateEvaluation, 2> evaluations = {
      evaluateCandidateCost(knownResources(), lhs),
      evaluateCandidateCost(knownResources(), rhs)};
  sortCandidateEvaluations(evaluations);
  ASSERT_TRUE(evaluations.front().accepted);
  EXPECT_EQ(evaluations.front().candidate.plan.stableId,
            first.front().stableId);
  EXPECT_NE(evaluations.front().remark.find("tensorize_factor="),
            std::string::npos);
  EXPECT_NE(evaluations.front().remark.find("block_t="), std::string::npos);
  EXPECT_NE(evaluations.front().remark.find("static_axis_fusion_factor="),
            std::string::npos);
  EXPECT_NE(evaluations.front().remark.find("peak_live_bytes="),
            std::string::npos);

  ProfilerCalibrationRecord record;
  record.kernelName = "norm_q";
  record.profilerArtifact = "kernel_details_norm_q.csv";
  record.sampleCount = 20;
  record.medianNanoseconds = 123;
  record.p90Nanoseconds = 145;
  record.evaluation = evaluations.front();
  const std::string calibration = formatProfilerCalibrationRecord(record);
  EXPECT_NE(calibration.find("schema=v1"), std::string::npos);
  EXPECT_NE(calibration.find("median_ns=123"), std::string::npos);
  EXPECT_NE(calibration.find("resource-cost candidate="), std::string::npos);
}
