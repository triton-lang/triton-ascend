/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"
#include "llvm/Support/raw_ostream.h"

#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Diagnostics.h"
#include "mlir/IR/DialectRegistry.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "mlir/Support/WalkResult.h"
#include "llvm/Support/Debug.h"
#include <atomic>
#include <cstdint>
#include <optional>
#include <thread>

#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache.h"
#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"
#include "ascend/include/DynamicCVPipeline/EstimateCVPipelineCost.h"
#include "ascend/include/DynamicCVPipeline/MainLoopUnroll.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlock/Passes.h"
#include "ascend/include/DynamicCVPipeline/PlanComputeBlockPass.h"
#include "ascend/include/DynamicCVPipeline/PreCheckAvailable.h"
#include "ascend/include/DynamicCVPipeline/RemoveAttributes.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromComputePass.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflowPass.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <optional>
#include <tuple>
#ifndef _WIN32
#include <fcntl.h>
#include <unistd.h>
#endif
#include "DynamicCVPipeline/Common/FallbackHelper.h"

static constexpr const char *DEBUG_TYPE = "add-dynamic-cv-pipeline";
static constexpr unsigned MAX_RETRY_TIMES = 2;
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(X) LLVM_DEBUG(DBGS() << (X) << "\n")

namespace mlir {
namespace triton {
#define GEN_PASS_DEF_ADDDYNAMICCVPIPELINE
#include "ascend/include/DynamicCVPipeline/Passes.h.inc"
} // namespace triton
} // namespace mlir

namespace {

static std::optional<int64_t> getErrorCode(ModuleOp moduleOp) {
  auto errCodeAttr =
      moduleOp->getAttrOfType<IntegerAttr>(CVPipeline::ERRCODE_ATTR);
  return errCodeAttr ? std::optional<int64_t>(errCodeAttr.getInt())
                     : std::nullopt;
}

//===----------------------------------------------------------------------===//
// Variant search
//===----------------------------------------------------------------------===//
// The pipeline is deterministic apart from one decision: the order the
// operations are linearised in, which ReorderOpsByBlockId makes and which
// ssbuffer.reorder_seed selects. Everything downstream -- where the blocks fall,
// where the barriers land, how deep the software pipeline gets -- follows from
// it, so running the pipeline once per seed and keeping the cheapest result is
// a search over genuinely different IR rather than over cosmetic orderings.
//
// The cost of the search is one full pipeline run per candidate, so it is off
// unless asked for. Candidates that fail are simply skipped: a variant that
// trips a downstream assumption sets the fallback attribute, and the search
// treats that as an infinitely expensive candidate rather than as an error.

constexpr const char *kVariantCountEnvVar = "TRITON_ASCEND_CV_VARIANTS";

/// Two totals this close to each other are treated as the same number.
///
/// The estimate is the larger of two bounds, and the resource bound does not
/// depend on how operations were grouped -- so once a candidate reaches it,
/// every other candidate that also reaches it scores identically, and the
/// search has nothing left to compare. Exact equality is the common case;
/// near-equality happens because the fusion factor rounds per block, and one
/// run had a candidate win by a single cycle out of half a billion. Neither is
/// a finding, so both are called a tie and decided on the second bound.
///
/// 0.0001 is deliberately tight. Widening it starts trading a real difference
/// in the leading bound for a difference in the trailing one, which is the
/// wrong way round.
constexpr int64_t kTieRelativeDenominator = 10000;

bool totalsAreTied(int64_t a, int64_t b) {
  const int64_t diff = a > b ? a - b : b - a;
  const int64_t scale = a > b ? a : b;
  return diff * kTieRelativeDenominator <= scale;
}

constexpr const char *kUBSlackEnvVar = "TRITON_ASCEND_CV_UB_SLACK_BYTES";

/// How much more Unified Buffer a candidate may hold live than the untouched
/// pipeline does, before the search discards it.
///
/// The search has no other way to know that a candidate cannot be built:
/// overrunning UB is diagnosed by the binary compiler, in a separate process,
/// after the whole of this compilation has finished. So an unbuildable
/// candidate looks perfectly healthy here and gets picked -- and it is the good
/// candidates that overrun, because the overlap they win is paid for by
/// materialising the values that cross each new block boundary. Estimate and
/// footprint are the same variable measured twice.
///
/// Measured against the baseline rather than against the capacity, and that is
/// the whole point. The absolute figure this model computes is not the one the
/// binary compiler enforces -- it multi-buffers again on top, aligns to banks,
/// adds temporaries later -- but every one of those applies to the baseline
/// too, so they cancel in a difference. What is left is the increment, which is
/// exactly the buffers the new cuts created, and that this model does know.
/// The baseline compiles by construction: it is what the pipeline emits with
/// none of this, and the search spends its first attempt on it.
///
/// Unset, it is derived rather than defaulted: the room a candidate may spend
/// is exactly what the baseline leaves unused, `capacity - peak(baseline)`,
/// both halves of which the estimate publishes. That is automatic on any
/// kernel -- a baseline that already fills the buffer yields zero and admits
/// nothing, which is the right answer and is what flash attention at 128x128
/// gives. Set, it overrides: 0 admits only candidates holding no more than the
/// baseline, 65536 lets one more score tile through.
constexpr const char *kStageSlackEnvVar = "TRITON_ASCEND_CV_STAGE_SLACK";

/// How much larger a software-pipeline buffer factor a candidate may carry than
/// the untouched pipeline does.
///
/// This guards the requiredBuffers factor of ssbuffer.iter_extension, not the
/// stage count -- the two are different numbers and only this one predicts a
/// module that will not build. When a producer and its consumer sit more stages
/// apart than there are buffers between them, the binary compiler refuses the
/// module for reading a buffer before its first write, and measured, the
/// untouched pipeline runs at 1 while every candidate refused for that reason
/// had gone to 2.
///
/// The stage count is deliberately NOT guarded. It costs time rather than
/// correctness -- a pipeline of N stages spends N iterations filling and
/// draining -- and the estimate charges that directly, so a candidate that buys
/// shorter dependency chains by adding stages pays for them in its own score.
/// Forbidding depth outright would also forbid the case where a long enough
/// loop amortises it, which is a real one on other kernels.
///
/// Defaults to zero, which is not a tuned value but the structural rule: the
/// factor is a small integer the pipeline chose, and a candidate that needs a
/// larger one has asked for a transformation the toolchain then refuses.
/// Nothing here scales with the kernel, so there is nothing to fit per kernel.
/// The variable only widens it.
std::optional<int64_t> getStageSlack() {
  static const std::optional<int64_t> slack = []() -> std::optional<int64_t> {
    const char *env = std::getenv(kStageSlackEnvVar);
    if (!env) {
      return std::nullopt;
    }
    int64_t value = 0;
    if (llvm::StringRef(env).getAsInteger(10, value) || value < 0) {
      llvm::errs() << "[" << DEBUG_TYPE << "] " << kStageSlackEnvVar << "='"
                   << env << "' is not a count; ignoring\n";
      return std::nullopt;
    }
    return value;
  }();
  return slack;
}

std::optional<int64_t> getUBSlackBytes() {
  static const std::optional<int64_t> slack = []() -> std::optional<int64_t> {
    const char *env = std::getenv(kUBSlackEnvVar);
    if (!env) {
      return std::nullopt;
    }
    int64_t value = 0;
    if (llvm::StringRef(env).getAsInteger(10, value) || value < 0) {
      llvm::errs() << "[" << DEBUG_TYPE << "] " << kUBSlackEnvVar << "='" << env
                   << "' is not a byte count; ignoring\n";
      return std::nullopt;
    }
    return value;
  }();
  return slack;
}

/// How many orderings to try. One or fewer means the ordinary single
/// compilation, which is what every build that does not ask for a search gets.
int getVariantCount() {
  const char *env = std::getenv(kVariantCountEnvVar);
  if (!env) {
    return 1;
  }
  int value = 0;
  if (llvm::StringRef(env).getAsInteger(10, value) || value < 1) {
    llvm::errs() << "[" << DEBUG_TYPE << "] " << kVariantCountEnvVar << "='"
                 << env << "' is not a positive number; ignored\n";
    return 1;
  }
  return value;
}

/// How many candidates the search scores at once.
///
/// The candidates are independent, so scoring them at once trades memory for
/// the sum of their run times. One, the default, is the historical serial
/// search, and it stays the path a search is debugged on: with one worker a
/// failure is the one the pipeline ran, and the report comes out in seed
/// order with nothing to replay. A number is a ceiling, not a request -- the
/// search caps it at the number of candidates, since an idle worker is a
/// worker that only costs to start.
constexpr const char *kVariantThreadsEnvVar = "TRITON_ASCEND_CV_VARIANT_THREADS";

int getVariantThreadCount() {
  const char *env = std::getenv(kVariantThreadsEnvVar);
  if (!env) {
    return 1;
  }
  int value = 0;
  if (llvm::StringRef(env).getAsInteger(10, value) || value < 1) {
    llvm::errs() << "[" << DEBUG_TYPE << "] " << kVariantThreadsEnvVar << "='"
                 << env << "' is not a positive number; ignored\n";
    return 1;
  }
  return value;
}

/// Turns the context's multithreading on for its lifetime and restores the
/// flag after. It only ever enables: a context that already had it is left
/// alone.
///
/// Needed because every MLIRContext built from Python is constructed with
/// threading off (upstream Triton's own binding), and with it off the
/// attribute and type uniquers run unlocked, so parallel trials would race on
/// them. Flipping it is safe here: MLIR forbids the change only while one of
/// its own parallel regions is active, and the search uses plain std::threads,
/// so the guard is built before any worker starts and destroyed after the join.
class ScopedContextThreading {
public:
  ScopedContextThreading(MLIRContext *context, bool enable)
      : context(context) {
    if (enable && !context->isMultithreadingEnabled()) {
      context->disableMultithreading(false);
      restoreToDisabled = true;
    }
  }

  ~ScopedContextThreading() {
    if (restoreToDisabled) {
      context->disableMultithreading(true);
    }
  }

  ScopedContextThreading(const ScopedContextThreading &) = delete;
  ScopedContextThreading &operator=(const ScopedContextThreading &) = delete;

private:
  MLIRContext *context;
  bool restoreToDisabled = false;
};

/// Swallows everything written to stdout and stderr while it is alive.
///
/// A trial run drives the whole pipeline, and the pipeline is talkative: debug
/// prints from several passes, MLIR diagnostics from the ones that decline a
/// candidate, and the estimate's own report. Multiplied by the number of
/// candidates that is thousands of lines describing IR that is about to be
/// thrown away. Silencing at the file descriptor rather than by asking each
/// pass to be quiet is the only thing that covers all of them, including
/// prints this pass does not own.
class OutputSilencer {
public:
  OutputSilencer() {
#ifndef _WIN32
    flushAll();
    devNull = ::open("/dev/null", O_WRONLY);
    if (devNull < 0) {
      return; // cannot silence; noisy is better than broken
    }
    savedOut = ::dup(STDOUT_FILENO);
    savedErr = ::dup(STDERR_FILENO);
    ::dup2(devNull, STDOUT_FILENO);
    ::dup2(devNull, STDERR_FILENO);
#endif
  }

  ~OutputSilencer() {
#ifndef _WIN32
    flushAll();
    if (savedOut >= 0) {
      ::dup2(savedOut, STDOUT_FILENO);
      ::close(savedOut);
    }
    if (savedErr >= 0) {
      ::dup2(savedErr, STDERR_FILENO);
      ::close(savedErr);
    }
    if (devNull >= 0) {
      ::close(devNull);
    }
#endif
  }

  OutputSilencer(const OutputSilencer &) = delete;
  OutputSilencer &operator=(const OutputSilencer &) = delete;

private:
  static void flushAll() {
    // Buffered streams must be emptied on both sides of the swap, or their
    // contents surface attributed to the wrong descriptor.
    std::cout.flush();
    std::cerr.flush();
    llvm::outs().flush();
    llvm::errs().flush();
    std::fflush(stdout);
    std::fflush(stderr);
  }

  int devNull = -1;
  int savedOut = -1;
  int savedErr = -1;
};

void restoreModuleFromBackup(ModuleOp moduleOp, ModuleOp moduleBackup) {
  Operation *moduleOperation = moduleOp.getOperation();
  Operation *backupOperation = moduleBackup.getOperation();

  moduleOperation->setLoc(backupOperation->getLoc());
  moduleOperation->setAttrs(backupOperation->getAttrs());
  if (moduleOperation->getPropertiesStorageSize() != 0) {
    moduleOperation->copyProperties(backupOperation->getPropertiesStorage());
  }
  moduleOp.getBodyRegion().takeBody(moduleBackup.getBodyRegion());
}

static inline void addPasses(OpPassManager &pm, int mainLoopUnrollFactor = 0) {
  // Unroll the main loop once the compute blocks are planned but before the
  // dataflow is split, so that the inter core transfers, their sync flags and
  // the multi buffers below are planned for each unrolled copy separately.
  // The pass is a no-op unless a factor > 1 was requested.
  if (mainLoopUnrollFactor > 1) {
    MainLoopUnrollOptions unrollOptions;
    unrollOptions.unrollFactor = mainLoopUnrollFactor;
    pm.addPass(createMainLoopUnrollPass(unrollOptions));
    pm.addPass(createCanonicalizerPass());
  }
  pm.addPass(createPreCheckAvailablePass());
  pm.addPass(createStandardizeOpPass());
  pm.addPass(createPlanComputeBlockPass());
  pm.addPass(createComputeBlockOptPass());
  pm.addPass(createSplitDataflowPass());
  pm.addPass(createAnalyzeDataFlowPass());
  pm.addPass(createAllocMultiCachePass());
  pm.addPass(createAddControlFlowConditionPass());
  pm.addPass(createSeparateMemoryFromComputePass());
  // Must precede createRemoveSsbufAttrPass(): the estimate is driven by the
  // ssbuffer.* attributes that pass strips.
  pm.addPass(createEstimateCVPipelineCostPass());
  pm.addPass(createRemoveSsbufAttrPass());
}

/// The result of one candidate: its score, or the reason it was not scored.
/// Everything here is a plain number, which is what makes the result safe to
/// hand back from a worker to the thread that ranks it.
struct VariantScore {
  int64_t seed = 0;
  bool accepted = false;
  int64_t cost = 0;
  int64_t second = 0;
  int64_t ubBytes = 0;
  int64_t ubCapacity = 0;
  int64_t depth = 0;
  int64_t stages = 0;
  int errCode = 0;
};

/// Load into the context everything a trial's PassManager would register on
/// its own, before the search fans out into threads.
///
/// PassManager::run starts by appending the pipeline's dependent dialects to
/// the context's registry and loading whichever of them are missing. That is
/// the one place a trial writes to state the other trials are reading: the
/// registry is not behind the context's locks, and neither is the table of
/// loaded dialects it feeds. The pipeline is fixed, so one pre-load of
/// everything it declares is enough; after it, each trial's PassManager finds
/// the registry already a superset of its own needs and leaves both
/// structures alone, and what the trials share is only the context's locked
/// state.
void prepareVariantSearchDialects(MLIRContext *context,
                                 int mainLoopUnrollFactor) {
  OpPassManager searchPM(ModuleOp::getOperationName());
  addPasses(searchPM, mainLoopUnrollFactor);
  DialectRegistry dependentDialects;
  searchPM.getDependentDialects(dependentDialects);
  context->appendDialectRegistry(dependentDialects);
  for (StringRef name : dependentDialects.getDialectNames()) {
    context->getOrLoadDialect(name);
  }
}

/// Run the whole pipeline once for one seed, on a private copy of the
/// untouched input, and read back whatever the estimate wrote.
///
/// The trial is the unit the search parallelises: it clones, it builds the
/// pipeline, it scores, and it discards. The IR it works on never leaves the
/// thread that runs it, and the caller's module is never touched, which is
/// what makes several of them legal at once. The measurement is the one the
/// serial search took by hand: the total, the bound that decided ties, and
/// the two figures the baseline guards are written against.
VariantScore runVariantTrial(MLIRContext *context, ModuleOp moduleBackup,
                            int mainLoopUnrollFactor, int64_t seed) {
  VariantScore score;
  score.seed = seed;

  OpBuilder builder(context);
  ModuleOp attempt(moduleBackup->clone());
  attempt->setAttr(CVPipeline::kReorderSeed, builder.getI64IntegerAttr(seed));
  attempt->setAttr(mlir::triton::kCVPipelineCostQuiet, builder.getUnitAttr());

  PassManager trial(context, attempt.getOperationName());
  addPasses(trial, mainLoopUnrollFactor);
  const bool ran =
      succeeded(trial.run(attempt.getOperation())) &&
      !CVPipeline::hasFallbackAttr(attempt);
  if (!ran) {
    auto codeAttr =
        attempt->getAttrOfType<IntegerAttr>(CVPipeline::ERRCODE_ATTR);
    // -1: the pipeline declined without saying why.
    score.errCode = codeAttr ? static_cast<int>(codeAttr.getInt()) : -1;
  } else {
    auto costAttr = attempt->getAttrOfType<IntegerAttr>(
        mlir::triton::kCVPipelineEstimatedCycles);
    if (!costAttr) {
      // -2: it compiled, but produced no estimate to rank it by.
      score.errCode = -2;
    } else {
      score.accepted = true;
      score.cost = costAttr.getInt();
      auto resourceAttr = attempt->getAttrOfType<IntegerAttr>(
          mlir::triton::kCVPipelineCostResource);
      auto recurrenceAttr = attempt->getAttrOfType<IntegerAttr>(
          mlir::triton::kCVPipelineCostRecurrence);
      // Falling back to the total itself makes every tie compare equal,
      // which is exactly the behaviour before the second bound existed.
      score.second =
          resourceAttr && recurrenceAttr
              ? std::min(resourceAttr.getInt(), recurrenceAttr.getInt())
              : score.cost;
      if (auto ubAttr = attempt->getAttrOfType<IntegerAttr>(
              mlir::triton::kCVPipelineCostUBPeak)) {
        score.ubBytes = ubAttr.getInt();
      }
      if (auto stageAttr = attempt->getAttrOfType<IntegerAttr>(
              mlir::triton::kCVPipelineCostPipelineStages)) {
        score.stages = stageAttr.getInt();
      }
      if (auto capAttr = attempt->getAttrOfType<IntegerAttr>(
              mlir::triton::kCVPipelineCostUBCapacity)) {
        score.ubCapacity = capAttr.getInt();
      }
      if (auto depthAttr = attempt->getAttrOfType<IntegerAttr>(
              mlir::triton::kCVPipelineCostPipelineDepth)) {
        score.depth = depthAttr.getInt();
      }
    }
  }
  // The clone is the trial's, and its one reference is here: dropping it
  // releases the whole tree the pipeline built inside it.
  attempt->destroy();
  return score;
}

} // namespace

// must collect all sub-passes since they now are not added to pipeline
void AddDynamicCVPipelinePass::getDependentDialects(
    DialectRegistry &registry) const {
  Base::getDependentDialects(registry);
  OpPassManager tempPM(ModuleOp::getOperationName());
  addPasses(tempPM, this->mainLoopUnrollFactor);
  tempPM.getDependentDialects(registry);
}

AddDynamicCVPipelinePass::AddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options)
    : AddDynamicCVPipelineBase(options) {}

void AddDynamicCVPipelinePass::runOnOperation() {
  auto moduleOp = getOperation();
  OpBuilder builder(moduleOp.getContext());
  compileOn91095Flag = this->compileOn91095;

  LDBG("Enter pass");
  moduleOp->removeAttr(CVPipeline::ERRCODE_ATTR);

  if (!compileOn91095Flag) {
    llvm::errs() << "Add-dynamic-cv-pipeline is only supported on 91095 now.\n";
    return;
  }
  if (this->demoteF32Reduction) {
    moduleOp->setAttr("triton_ascend.demote_f32_reduction", builder.getUnitAttr());
  }

  ModuleOp moduleBackup(moduleOp->clone());

  auto buildPipeline = [&](PassManager &pm) {
    addPasses(pm, this->mainLoopUnrollFactor);
  };

  // Try several orderings and keep the cheapest, when asked to. The winning
  // seed is compiled once more at the end with the estimate allowed to speak,
  // so the module that survives and the report that describes it are the same
  // compilation rather than two that happen to agree.
  //
  // The candidates are independent: each trial starts from the untouched
  // input and publishes only its score, so several of them may run at once
  // and the wall clock is the slowest candidate instead of the sum of them
  // all. The only state a trial shares with the others is the context -- the
  // IR it works on never leaves the thread that runs it -- and the dialect
  // pre-load below takes care of the one place a trial would still write to
  // it.
  const int variantCount = getVariantCount();
  if (variantCount > 1) {
    prepareVariantSearchDialects(&getContext(), this->mainLoopUnrollFactor);

    // How many trials to score at once, capped at the number of candidates.
    const int threadCount = std::min(getVariantThreadCount(), variantCount);
    llvm::errs() << "[" << DEBUG_TYPE << "] variant search: trying "
                 << variantCount << " operation orderings";
    if (threadCount > 1) {
      llvm::errs() << " on " << threadCount << " thread(s)";
    }
    llvm::errs() << "\n";

    const std::optional<int64_t> ubSlack = getUBSlackBytes();
    const std::optional<int64_t> stageSlack = getStageSlack();
    SmallVector<VariantScore> scores(variantCount);

    {
      // Nothing a trial prints is worth reading: it describes IR that is
      // about to be discarded, and with several trials at once the
      // interleaving is noise even where a single line would have made
      // sense. The estimate is asked to stay quiet, and the rest of the
      // pipeline is silenced at the descriptor. The redirection is
      // process-wide, which is why there is one silencer for the whole
      // phase rather than one per thread: two silencers would undo each
      // other.
      OutputSilencer hush;
      ScopedContextThreading threading(&getContext(), threadCount > 1);
      if (threadCount > 1) {
        // The workers are plain std::thread rather than a borrow from the
        // context's pool: that pool is sized for the machine, and how many
        // candidates are scored at once is a property of the search. (The
        // standalone llvm::ThreadPool has been removed from recent LLVM, so
        // the search carries its own tiny pool: a counter, a vector of
        // threads, and a join.)
        std::atomic<int64_t> nextSeed(0);
        SmallVector<std::thread> workers;
        workers.reserve(threadCount);
        for (int worker = 0; worker < threadCount; ++worker) {
          workers.emplace_back([&] {
            for (;;) {
              const int64_t seed = nextSeed.fetch_add(1);
              if (seed >= variantCount) {
                break;
              }
              scores[seed] = runVariantTrial(&getContext(), moduleBackup,
                                             this->mainLoopUnrollFactor, seed);
            }
          });
        }
        for (auto &worker : workers) {
          worker.join();
        }
      } else {
        for (int64_t seed = 0; seed < variantCount; ++seed) {
          scores[seed] =
              runVariantTrial(&getContext(), moduleBackup,
                              this->mainLoopUnrollFactor, seed);
        }
      }
    }

    // The ranking is a function of the scores alone, so it runs here, in
    // seed order, no matter in which order the trials came back. The two
    // baseline guards are applied in this replay rather than inside the
    // trials: the baseline is seed 0's own numbers, which do not exist until
    // seed 0 is back, and in a parallel search the other trials are long
    // finished by then. Seed 0 always passes its own baseline, so the replay
    // makes exactly the comparisons the serial search made.
    int64_t bestSeed = -1;
    int64_t bestCost = 0;
    // The bound the total did not come from, for the winner. Used only to
    // separate candidates whose totals tie.
    int64_t bestSecond = 0;
    int64_t worstCost = 0;
    int64_t usable = 0;
    // How often the second bound was what decided. A search that never used it
    // was ordered by the total alone; one that used it often is running against
    // a wall the total cannot see past, which is worth knowing before trusting
    // the winner.
    int64_t tieBreaks = 0;
    // How many different scores the candidates got. One means every ordering
    // that compiled looked identical to the estimate, which says where to look
    // next: either the generator is producing one shape under many names, or
    // the estimate cannot see what separates them. Without this the two are
    // indistinguishable from outside.
    //
    // Keyed on the pair, not on the total: since the second bound decides
    // ties, two candidates with the same total are no longer the same
    // candidate, and counting totals alone under-reports how much the search
    // can actually tell apart.
    llvm::DenseSet<std::pair<int64_t, int64_t>> distinctCosts;
    // Candidates whose total was indistinguishable from the best so far. The
    // difference between "no ties happened" and "ties happened and none of
    // them was an improvement" is not visible from the winner alone, and the
    // two mean opposite things about whether the second bound is earning its
    // keep.
    int64_t tiedCandidates = 0;
    // Every accepted candidate, so that a winner rejected further down the
    // toolchain -- by the binary compiler running out of Unified Buffer, say,
    // which no pass here can foresee -- leaves the next choices on record
    // instead of sending the search back to the start.
    // Ordered by the first two, so the trailing fields are carried along rather
    // than compared: seed is unique and stages is a diagnostic.
    SmallVector<std::tuple<int64_t, int64_t, int64_t, int64_t, int64_t>> ranked;
    // Set by seed 0, the untouched pipeline. Everything after it is judged
    // against these.
    std::optional<int64_t> baselineUBPeak;
    std::optional<int64_t> baselineDepth;
    // Derived from the baseline when TRITON_ASCEND_CV_UB_SLACK_BYTES says
    // nothing: the room to spend is what the baseline leaves unused.
    int64_t ubAllowance = 0;
    if (scores[0].accepted) {
      baselineUBPeak = scores[0].ubBytes;
      baselineDepth = scores[0].depth;
      // What the baseline leaves unused, unless the caller named a figure.
      // A baseline that already fills the buffer yields zero and admits only
      // candidates that hold no more than it does -- the right answer, and
      // the one flash attention gives at 128x128.
      ubAllowance = ubSlack ? *ubSlack
                            : std::max<int64_t>(0, scores[0].ubCapacity -
                                                          scores[0].ubBytes);
    }

    // Why the rejected candidates were rejected, counted by error code. A
    // search that keeps nothing is otherwise indistinguishable from one that
    // was never asked to do anything.
    SmallVector<std::pair<int, int64_t>> failures;
    auto countFailure = [&](int code) {
      for (auto &entry : failures) {
        if (entry.first == code) {
          ++entry.second;
          return;
        }
      }
      failures.push_back({code, 1});
    };

    for (int64_t seed = 0; seed < variantCount; ++seed) {
      VariantScore &score = scores[seed];

      // -3: it compiled and scored, but holds more Unified Buffer live than
      // the baseline plus what the caller allowed. Counted as a rejection
      // rather than ranked, so the summary shows how much of the search this
      // threw away -- and if that is most of it, the answer for this kernel
      // is that the partition axis has no room, which is worth seeing rather
      // than inferring from a winner that will not build. Both guards below
      // run against the baseline rather than against an absolute, so nothing
      // here is fitted to a kernel; the baseline itself always passes, having
      // set the reference, so the worst a wrong estimate can do is return it.
      if (score.accepted && baselineUBPeak &&
          score.ubBytes > *baselineUBPeak + ubAllowance) {
        score.accepted = false;
        score.errCode = -3;
      }
      // -4: the software pipeline was cut into more stages than the
      // baseline's, which the binary compiler refuses as a prologue that
      // reads a buffer before its first write.
      if (score.accepted && baselineDepth &&
          score.depth > *baselineDepth + stageSlack.value_or(0)) {
        score.accepted = false;
        score.errCode = -4;
      }

      if (!score.accepted) {
        countFailure(score.errCode);
        continue;
      }

      ++usable;
      if (seed == 0) {
        llvm::errs() << "[" << DEBUG_TYPE << "]   baseline: " << score.ubBytes
                     << " bytes of unified buffer live against a capacity of "
                     << score.ubCapacity << ", buffer factor " << score.depth
                     << ", " << score.stages << " pipeline stage(s)"
                     << "; candidates may spend " << ubAllowance
                     << " more byte(s) and " << stageSlack.value_or(0)
                     << " more buffer factor(s). Stage count is not guarded:"
                     << " the fill and drain it costs are charged in the"
                     << " estimate, so a deeper candidate has to earn it\n";
      }
      distinctCosts.insert({score.cost, score.second});
      ranked.push_back({score.cost, score.second, score.ubBytes, seed,
                        score.stages});
      if (worstCost < score.cost) {
        worstCost = score.cost;
      }
      // Order by the total, and by the other bound only when the totals cannot
      // be told apart. The two bounds are not interchangeable -- the total is
      // still what the model claims the kernel costs -- so this never lets a
      // candidate with a worse total win.
      bool improves = bestSeed < 0;
      bool byTieBreak = false;
      if (!improves) {
        if (totalsAreTied(score.cost, bestCost)) {
          ++tiedCandidates;
          improves = score.second < bestSecond;
          byTieBreak = improves;
        } else {
          improves = score.cost < bestCost;
        }
      }
      if (improves) {
        bestSeed = seed;
        bestCost = score.cost;
        bestSecond = score.second;
        if (byTieBreak) {
          ++tieBreaks;
        }
        llvm::errs() << "[" << DEBUG_TYPE << "]   seed " << seed << ": "
                     << bestCost << " cycles";
        if (byTieBreak) {
          llvm::errs() << " (tied on the total; won on the other bound at "
                       << bestSecond << ")";
        }
        llvm::errs() << " (best so far)\n";
      }
    }

    llvm::errs() << "[" << DEBUG_TYPE << "] variant search: " << usable << " of "
                 << variantCount << " ordering(s) compiled";
    if (bestSeed >= 0) {
      llvm::errs() << ", keeping seed " << bestSeed << " at " << bestCost
                   << " cycles\n";
      llvm::errs() << "[" << DEBUG_TYPE << "]   " << distinctCosts.size()
                   << " distinct estimate(s), worst " << worstCost
                   << " cycles; seed 0 is the untouched pipeline\n";
      if (tiedCandidates > 0) {
        llvm::errs() << "[" << DEBUG_TYPE << "]   " << tiedCandidates
                     << " candidate(s) tied with the best on the total, "
                     << tieBreaks << " of which won on the second bound";
        if (tieBreaks == 0) {
          llvm::errs() << " -- so the ties were real and none of them had more"
                          " room than the incumbent, which is what to expect"
                          " when the best candidate already has the shortest"
                          " dependency chain";
        }
        llvm::errs() << "\n";
      }
      // Ordered the same way the search ordered them, so the runner-up is a
      // seed and not a re-run. Worth having because a winner can still be
      // refused by the binary compiler for something no pass here models --
      // Unified Buffer capacity being the case that has actually bitten.
      // One seed per distinct score. Many seeds reach the same pair -- 118 of
      // 200 tied on one run -- and listing five of those is five names for one
      // candidate, which is useless precisely when the list is needed: the
      // winner having been refused downstream, what is wanted is a *different*
      // candidate to try, not another spelling of the same one.
      llvm::sort(ranked);
      llvm::errs() << "[" << DEBUG_TYPE
                   << "]   best distinct score(s), in case the winner is"
                      " refused downstream:";
      int64_t shown = 0;
      for (size_t i = 0; i < ranked.size() && shown < 5; ++i) {
        if (i > 0 && std::get<0>(ranked[i]) == std::get<0>(ranked[i - 1]) &&
            std::get<1>(ranked[i]) == std::get<1>(ranked[i - 1])) {
          continue;
        }
        llvm::errs() << " " << std::get<3>(ranked[i]) << "("
                     << std::get<0>(ranked[i]) << "/" << std::get<1>(ranked[i])
                     << "/" << std::get<2>(ranked[i]) << "B/"
                     << std::get<4>(ranked[i]) << "st)";
        ++shown;
      }
      llvm::errs() << "  [seed(total/second bound/unified buffer/pipeline"
                      " stages)]\n";
    } else {
      llvm::errs() << ", none usable; compiling without a variant\n";
    }
    for (const auto &entry : failures) {
      llvm::errs() << "[" << DEBUG_TYPE << "]   rejected " << entry.second
                   << " with ";
      if (entry.first == -1) {
        llvm::errs() << "a pipeline failure and no error code\n";
      } else if (entry.first == -2) {
        llvm::errs() << "no estimate produced\n";
      } else if (entry.first == -3) {
        llvm::errs() << "more than the baseline's "
                     << baselineUBPeak.value_or(0) << " bytes of unified"
                        " buffer plus its "
                     << ubAllowance << "-byte allowance\n";
      } else if (entry.first == -4) {
        llvm::errs() << "a larger software-pipeline buffer factor than the"
                        " baseline's "
                     << baselineDepth.value_or(0)
                     << ", which the binary compiler refuses as a prologue that"
                        " reads a buffer before its first write\n";
      } else {
        llvm::errs() << "error code " << entry.first << "\n";
      }
    }
    if (bestSeed < 0) {
      llvm::errs() << "[" << DEBUG_TYPE
                   << "]   reproduce one of them with"
                      " TRITON_ASCEND_REORDER_SEED=0 and no"
                      " TRITON_ASCEND_CV_VARIANTS to see the failure itself\n";
    }

    // Reset to the input and compile the winner for real, with the estimate
    // allowed to report. A search that found nothing leaves the seed unset,
    // which is the ordinary path.
    ModuleOp fresh(moduleBackup->clone());
    restoreModuleFromBackup(moduleOp, fresh);
    fresh->destroy();
    if (bestSeed >= 0) {
      moduleOp->setAttr(CVPipeline::kReorderSeed,
                        builder.getI64IntegerAttr(bestSeed));
    }
  }

  ScopedDiagnosticHandler handler(&getContext(), [&](Diagnostic &diag) {
    LLVM_DEBUG({
      // In debug mode, continue to other handlers, i.e. print to stderr
      return llvm::failure();
    });

    // otherwise prohibit any other handler
    return llvm::success();
  });

  std::optional<int64_t> lastErrCode = std::nullopt;
  bool tuplePreloadFailed = false;
  for (unsigned attempt = 0; attempt < MAX_RETRY_TIMES; ++attempt) {
    // restore() consumes the saved region bodies. Each attempt needs its own
    // snapshot, taken before changing buffer counts for the retry.
    CVPipeline::FallbackHelper fallback(moduleOp);
    if (attempt > 0) {
      BufferCountManager bufferCountManager(moduleOp);
      bufferCountManager.setBufferCount(BufferCountManager::DepType::IntraCore,
                                        2);
      bufferCountManager.setBufferCount(BufferCountManager::DepType::InterCore,
                                        1);
    }

    // Do not reuse pass instances or partially transformed IR on retry.
    PassManager pm(&getContext(), moduleOp.getOperationName());
    buildPipeline(pm);

    // run passes in separate pm, instead of the pipeline to suppress reproducer
    auto result = pm.run(moduleOp);
    auto errCode = getErrorCode(moduleOp);
    if (succeeded(result) && !errCode.has_value() &&
        !CVPipeline::hasFallbackAttr(moduleOp)) {
      moduleBackup->destroy();
    }
    if (succeeded(result) && !errCode.has_value()) {
      if (tuplePreloadFailed) {
        CVPipeline::setFallbackAttr(moduleOp,
                                    CVPipeline::ERRCODE_TUPLE_PRELOAD_FAILED);
      }
      LDBG("Process successfully");
      return;
    }

    if (errCode == CVPipeline::ERRCODE_TUPLE_PRELOAD_FAILED) {
      if (attempt + 1 < MAX_RETRY_TIMES) {
        LDBG("Tuple-buffer failed; Retrying with tuple preload disabled.");
        tuplePreloadFailed = true;
        fallback.restore();
        moduleOp->removeAttr(CVPipeline::ERRCODE_ATTR);
        continue;
      }
      // Consume repeated retry requests instead of leaking code 3 to callers.
      errCode = CVPipeline::ERRCODE_FAILED;
    }

    lastErrCode = errCode;
    break;
  }

  int finalErrCode =
      lastErrCode ? static_cast<int>(*lastErrCode) : CVPipeline::ERRCODE_FAILED;
  std::cout << "[VDV DEBUG] DynamicCVPipeline failed errCode=" << finalErrCode
            << std::endl;

  if (!lastErrCode.has_value()) {
    moduleOp->emitWarning() << "[" << DEBUG_TYPE << "] "
                            << "Unexpected pass failure (no fallback attr "
                               "set); fallback to compilation without "
                               "dynamic CV pipeline.";
  } else if (*lastErrCode == CVPipeline::ERRCODE_IGNORED) {
    // This is an expected fallback: do not attach the full module IR.
    mlir::emitWarning(moduleOp->getLoc())
        << "[" << DEBUG_TYPE << "] "
        << "Kernel not applicable for dynamic CV pipeline "
           "(no matmul / already scope-optimized / unsupported "
           "pattern); falling back to standard compilation.";
  } else {
    moduleOp->emitWarning()
        << "[" << DEBUG_TYPE << "] " << "Pass failed (errcode=" << *lastErrCode
        << "); "
           "falling back to compilation without "
           "dynamic CV pipeline.";
  }

  restoreModuleFromBackup(moduleOp, moduleBackup);
  moduleBackup->destroy();
  moduleOp->setAttr(CVPipeline::ERRCODE_ATTR,
                    builder.getI32IntegerAttr(finalErrCode));
  LDBG("Process successfully");
}

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createAddDynamicCVPipelinePass(
    const AddDynamicCVPipelineOptions &options) {
  return std::make_unique<AddDynamicCVPipelinePass>(options);
}
