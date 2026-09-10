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

TEST(GraphOptimizationResourceCostModelTest,
     ProjectLaunchKeepsNonpersistentGridAndCapsVerifiedPersistentAxis) {
  const ResourceSnapshot resources = ResourceSnapshot::fromExplicit(
      /*ubCapacityBytes=*/256 * 1024, /*deviceCoreCount=*/8);
  ProgramGridSpecialization specialization;
  specialization.grid = {19, 16, 1};
  const std::array<ProgramGridTransform, 2> transforms = {{
      {0, 1, 4, 16, /*persistentCoverage=*/false,
       /*gridStrideAbiVerified=*/false},
      {1, 0, 2, 19, /*persistentCoverage=*/true,
       /*gridStrideAbiVerified=*/true},
  }};

  auto projection =
      projectProgramMappingLaunch(specialization, transforms, resources);
  ASSERT_TRUE(projection);
  EXPECT_EQ(projection->logicalGrid, (std::array<uint64_t, 3>{10, 4, 1}));
  EXPECT_EQ(projection->physicalGrid, (std::array<uint64_t, 3>{8, 4, 1}));
  EXPECT_EQ(projection->logicalPrograms, 40u);
  EXPECT_EQ(projection->physicalPrograms, 32u);
  EXPECT_TRUE(projection->persistentCoverage);
  EXPECT_FALSE(projection->legacyAutoMap);
}
