//===- PipelineAnalysisPass.cpp - Pipeline scheduling analysis -----------===//
//
// Uses HardwareConfig for configurable hardware parameters.
// Handles dynamic loop bounds via arg-bindings option (supports program_id).
// Generates Perfetto trace with loop unrolling visualization.
// Uses Roofline model for cycle estimation with HW unit overlap.
//
//===----------------------------------------------------------------------===//

#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"
#include "AscendModel/Analysis/PipelineAnalysis.h"
#include "AscendModel/HardwareConfig.h"
#include "AscendModel/Utils.h"

#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/raw_ostream.h"

#include <algorithm>

namespace mlir {
namespace ascend {

#define GEN_PASS_DEF_PIPELINEANALYSISPASS
#include "AscendModel/Transforms/Passes.h.inc"

namespace {

using utils::getScfForTripCount;
using utils::getLoopMultiplier;
using utils::getScfForTripCountWithBindings;
using utils::parseBindings;
using utils::parseLoopTripCounts;

int getTrackId(HWUnit unit) {
  switch (unit) {
    case HWUnit::Cube:     return 1;
    case HWUnit::CubeMTE2: return 2;
    case HWUnit::FixPipe:  return 3;
    case HWUnit::Vector:   return 4;
    case HWUnit::VecMTE2:  return 5;
    case HWUnit::MTE3:     return 6;
    case HWUnit::Scalar:   return 7;
    default:               return 0;
  }
}

const char* getColorName(HWUnit unit) {
  switch (unit) {
    case HWUnit::Cube:     return "rail_response";
    case HWUnit::CubeMTE2: return "rail_load";
    case HWUnit::FixPipe:  return "cq_build_passed";
    case HWUnit::Vector:   return "rail_animation";
    case HWUnit::VecMTE2:  return "good";
    case HWUnit::MTE3:     return "bad";
    case HWUnit::Scalar:   return "grey";
    default:               return "generic_work";
  }
}

HWUnit getOpHWUnit(Operation *op) {
  if (isa<MatmulOp>(op)) return HWUnit::Cube;
  if (isa<CubeLoadOp>(op)) return HWUnit::CubeMTE2;
  if (isa<CubeStoreOp>(op)) return HWUnit::FixPipe;
  if (isa<VectorLoadOp>(op)) return HWUnit::VecMTE2;
  if (isa<VectorStoreOp>(op)) return HWUnit::MTE3;
  if (isa<AddOp, SubOp, MulOp, DivOp, MaxOp, MinOp,
          ExpOp, LogOp, SqrtOp, RsqrtOp, TanhOp, SigmoidOp,
          NegOp, AbsOp, ReluOp, CastOp,
          ReduceSumOp, ReduceMaxOp, ReduceMinOp, ReduceProdOp,
          BroadcastOp, SelectOp>(op))
    return HWUnit::Vector;
  return HWUnit::Scalar;
}

/// Generate Perfetto trace with loop unrolling.
/// If maxIterations > 0, limits the number of iterations shown in trace.
void generatePerfettoTrace(const PipelineScheduler &scheduler,
                           StringRef filename,
                           int64_t oneIterCycles,
                           int64_t totalCycles,
                           int64_t maxIterations = 100) {
  std::error_code EC;
  llvm::raw_fd_ostream file(filename, EC, llvm::sys::fs::OF_Text);
  if (EC) {
    llvm::errs() << "Error opening file " << filename << ": " << EC.message() << "\n";
    return;
  }
  
  const auto &config = scheduler.getConfig();
  const auto &allOps = scheduler.getAllOps();
  
  // Calculate the maximum loop multiplier to determine iteration count
  int64_t maxLoopMultiplier = 1;
  for (const auto &op : allOps) {
    maxLoopMultiplier = std::max(maxLoopMultiplier, op.loopMultiplier);
  }
  
  // Limit iterations for visualization (avoid huge traces)
  int64_t numIterations = std::min(maxLoopMultiplier, maxIterations);
  bool truncated = (maxLoopMultiplier > maxIterations);
  
  double cycleToUs = 1.0;  // 1 cycle = 1 unit for visualization
  
  file << "{\n  \"traceEvents\": [\n";
  bool first = true;
  
  // Track metadata
  struct TrackInfo { int tid; const char* name; };
  TrackInfo tracks[] = {
    {1, "Cube Core"}, {2, "Cube MTE2 (HBM->L1)"}, {3, "FixPipe (L0C->HBM)"},
    {4, "Vector Core"}, {5, "Vec MTE2 (HBM->UB)"}, {6, "MTE3 (UB->HBM)"}, {7, "Scalar"}
  };
  
  // Write track metadata
  for (const auto &track : tracks) {
    if (!first) file << ",\n";
    first = false;
    file << "    {\"name\": \"thread_name\", \"ph\": \"M\", \"pid\": 1, \"tid\": " 
         << track.tid << ", \"args\": {\"name\": \"" << track.name << "\"}}";
  }
  
  for (const auto &track : tracks) {
    file << ",\n    {\"name\": \"thread_sort_index\", \"ph\": \"M\", \"pid\": 1, \"tid\": " 
         << track.tid << ", \"args\": {\"sort_index\": " << track.tid << "}}";
  }
  
  file << ",\n    {\"name\": \"process_name\", \"ph\": \"M\", \"pid\": 1, "
       << "\"args\": {\"name\": \"" << config.getName().str() << " Pipeline";
  if (truncated) {
    file << " (showing " << numIterations << "/" << maxLoopMultiplier << " iterations)";
  }
  file << "\"}}";
  
  // Calculate actual total cycles shown in trace
  int64_t traceTotalCycles = 0;
  
  // Generate events for each iteration
  // Key insight: operations with different loopMultipliers execute different numbers of times
  // We need to track per-HW-unit time to model pipeline parallelism across iterations
  
  // Track end time for each hardware unit
  llvm::DenseMap<HWUnit, int64_t> hwUnitEndTime;
  for (int i = 0; i <= static_cast<int>(HWUnit::Scalar); ++i) {
    hwUnitEndTime[static_cast<HWUnit>(i)] = 0;
  }
  
  // For each iteration
  for (int64_t iter = 0; iter < numIterations; ++iter) {
    // Track dependencies within this iteration
    llvm::DenseMap<int64_t, int64_t> opEndTimes;  // opId -> endTime in this iter
    
    for (const auto &op : allOps) {
      // Check if this op executes in this iteration
      // An op with loopMultiplier=N executes N times
      if (iter >= op.loopMultiplier)
        continue;
      
      // Calculate start time considering:
      // 1. Dependencies from previous ops in this iteration
      // 2. Hardware unit availability
      int64_t startTime = hwUnitEndTime[op.hwUnit];
      
      // Check dependencies
      for (int64_t depId : op.dependsOn) {
        auto it = opEndTimes.find(depId);
        if (it != opEndTimes.end()) {
          startTime = std::max(startTime, it->second);
        }
      }
      
      int64_t endTime = startTime + op.duration;
      
      // Update tracking
      hwUnitEndTime[op.hwUnit] = endTime;
      opEndTimes[op.opId] = endTime;
      traceTotalCycles = std::max(traceTotalCycles, endTime);
      
      // Write event
      int tid = getTrackId(op.hwUnit);
      file << ",\n    {\"name\": \"" << op.opName;
      if (op.loopMultiplier > 1) {
        file << "[" << iter << "]";  // Show iteration number
      }
      file << "\", "
           << "\"cat\": \"" << stringifyHWUnit(op.hwUnit).str() << "\", \"ph\": \"X\", "
           << "\"ts\": " << llvm::format("%.3f", startTime * cycleToUs) << ", "
           << "\"dur\": " << llvm::format("%.3f", op.duration * cycleToUs) << ", "
           << "\"pid\": 1, \"tid\": " << tid << ", "
           << "\"cname\": \"" << getColorName(op.hwUnit) << "\", "
           << "\"args\": {"
           << "\"op_id\": " << op.opId << ", "
           << "\"iteration\": " << iter << ", "
           << "\"cycles\": " << op.duration << ", "
           << "\"loop_multiplier\": " << op.loopMultiplier
           << "}}";
    }
  }
  
  // Add markers for total timeline
  for (const auto &track : tracks) {
    file << ",\n    {\"name\": \"\", \"cat\": \"marker\", \"ph\": \"i\", \"s\": \"t\", "
         << "\"ts\": 0, \"pid\": 1, \"tid\": " << track.tid << "}";
    file << ",\n    {\"name\": \"\", \"cat\": \"marker\", \"ph\": \"i\", \"s\": \"t\", "
         << "\"ts\": " << llvm::format("%.3f", traceTotalCycles * cycleToUs) 
         << ", \"pid\": 1, \"tid\": " << track.tid << "}";
  }
  
  // Add iteration markers
  if (numIterations > 1) {
    // Add counter track for iteration progress
    file << ",\n    {\"name\": \"Iterations\", \"ph\": \"C\", \"ts\": 0, \"pid\": 1, "
         << "\"args\": {\"shown\": " << numIterations << ", \"total\": " << maxLoopMultiplier << "}}";
  }
  
  file << "\n  ],\n";
  
  // Metadata
  file << "  \"metadata\": {\n";
  file << "    \"hardware\": \"" << config.getName().str() << "\",\n";
  file << "    \"one_iter_cycles\": " << oneIterCycles << ",\n";
  file << "    \"total_cycles\": " << totalCycles << ",\n";
  file << "    \"trace_cycles\": " << traceTotalCycles << ",\n";
  file << "    \"iterations_shown\": " << numIterations << ",\n";
  file << "    \"iterations_total\": " << maxLoopMultiplier << ",\n";
  file << "    \"clock_freq_ghz\": " << config.getClockFrequencyGHz() << ",\n";
  file << "    \"estimated_time_us\": " << llvm::format("%.3f", config.cyclesToMicroseconds(totalCycles)) << "\n";
  file << "  },\n";
  
  file << "  \"displayTimeUnit\": \"ns\"\n";
  file << "}\n";
  
  file.close();
}

struct PipelineAnalysisPass
    : public impl::PipelineAnalysisPassBase<PipelineAnalysisPass> {
  using PipelineAnalysisPassBase::PipelineAnalysisPassBase;
  
  void runOnOperation() override {
    ModuleOp module = getOperation();
    
    // Load hardware config from file if specified
    if (!hardwareConfigPath.empty()) {
      std::string error;
      if (!loadHardwareConfigFromFile(hardwareConfigPath, error)) {
        emitError(module.getLoc(), error);
        return signalPassFailure();
      }
    }
    
    const HardwareConfig &config = getHardwareConfig();
    
    // Parse bindings
    llvm::DenseMap<unsigned, int64_t> argBindings;
    llvm::StringMap<int64_t> programIdBindings;
    SmallVector<int64_t> loopTripCountOverrides;
    
    if (!argBindingsStr.empty()) {
      std::string parseError;
      if (!parseBindings(argBindingsStr, argBindings, programIdBindings, parseError)) {
        emitError(module.getLoc(), parseError);
        return signalPassFailure();
      }
    }
    
    if (!loopTripCountsStr.empty()) {
      std::string parseError;
      if (!parseLoopTripCounts(loopTripCountsStr, loopTripCountOverrides, parseError)) {
        emitError(module.getLoc(), parseError);
        return signalPassFailure();
      }
    }
    
    // Collect loops and ensure trip counts are set
    SmallVector<scf::ForOp> allLoops;
    module.walk([&](scf::ForOp forOp) { allLoops.push_back(forOp); });
    
    bool hasError = false;
    for (size_t loopIdx = 0; loopIdx < allLoops.size(); ++loopIdx) {
      scf::ForOp forOp = allLoops[loopIdx];
      
      if (forOp->hasAttr("ascend.trip_count"))
        continue;
      
      int64_t tripCount = 1;
      if (loopIdx < loopTripCountOverrides.size()) {
        tripCount = loopTripCountOverrides[loopIdx];
      } else {
        auto result = getScfForTripCountWithBindings(forOp, argBindings, programIdBindings);
        if (result.isStatic) {
          tripCount = result.staticTripCount;
        } else {
          emitError(forOp.getLoc(), "Loop " + std::to_string(loopIdx) + 
                    " trip count unknown. " + result.errorMsg);
          hasError = true;
          continue;
        }
      }
      
      forOp->setAttr("ascend.trip_count",
                     IntegerAttr::get(IntegerType::get(forOp.getContext(), 64), tripCount));
    }
    
    if (hasError) return signalPassFailure();
    
    // Build scheduler
    PipelineScheduler scheduler(&config);
    llvm::DenseMap<Value, int64_t> valueProducers;
    
    module.walk([&](Operation *op) {
      if (isa<scf::ForOp, scf::YieldOp, scf::IfOp>(op)) return;
      
      auto opIdAttr = op->getAttrOfType<IntegerAttr>("op_id");
      if (!opIdAttr) return;
      
      int64_t opId = opIdAttr.getInt();
      auto cyclesAttr = op->getAttrOfType<IntegerAttr>("estimated_cycles");
      int64_t cycles = cyclesAttr ? cyclesAttr.getInt() : 1;
      
      PipelineOp pipelineOp;
      pipelineOp.opId = opId;
      pipelineOp.hwUnit = getOpHWUnit(op);
      pipelineOp.duration = cycles;
      pipelineOp.mlirOp = op;
      pipelineOp.opName = op->getName().getStringRef().str();
      pipelineOp.loopMultiplier = getLoopMultiplier(op);
      
      for (Value operand : op->getOperands()) {
        auto it = valueProducers.find(operand);
        if (it != valueProducers.end()) {
          pipelineOp.dependsOn.push_back(it->second);
          scheduler.addDependency(it->second, opId);
        }
      }
      
      for (Value result : op->getResults())
        valueProducers[result] = opId;
      
      scheduler.addOperation(pipelineOp);
    });
    
    if (!scheduler.schedule()) {
      emitError(module.getLoc(), "Failed to schedule pipeline");
      return signalPassFailure();
    }
    
    // Calculate cycles using roofline model
    // oneIterCycles from scheduler already considers HW unit parallelism for one iteration
    int64_t oneIterCycles = scheduler.getTotalCycles();
    
    // For total cycles with loops, we need to consider:
    // 1. Each HW unit's total work across all iterations
    // 2. Take max (not sum) since they can overlap
    
    // Collect per-HW-unit cycles
    llvm::DenseMap<HWUnit, int64_t> hwUnitCycles;
    for (const auto &pipelineOp : scheduler.getAllOps()) {
      hwUnitCycles[pipelineOp.hwUnit] += pipelineOp.duration * pipelineOp.loopMultiplier;
    }
    
    // Group by path and apply roofline model
    // Cube path: max(Cube, CubeMTE2, FixPipe)
    int64_t cubePathCycles = std::max({
      hwUnitCycles[HWUnit::Cube],
      hwUnitCycles[HWUnit::CubeMTE2],
      hwUnitCycles[HWUnit::FixPipe]
    });
    
    // Vector path: max(Vector, VecMTE2, MTE3)
    int64_t vectorPathCycles = std::max({
      hwUnitCycles[HWUnit::Vector],
      hwUnitCycles[HWUnit::VecMTE2],
      hwUnitCycles[HWUnit::MTE3]
    });
    
    // Total: max of paths (assuming Cube and Vector can overlap)
    int64_t rooflineTotalCycles = std::max(cubePathCycles, vectorPathCycles);
    
    // Also calculate simple sum for comparison
    int64_t simpleSumCycles = 0;
    for (const auto &pipelineOp : scheduler.getAllOps())
      simpleSumCycles += pipelineOp.duration * pipelineOp.loopMultiplier;
    
    module->setAttr("ascend.scheduled_cycles_one_iter",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), oneIterCycles));
    module->setAttr("ascend.roofline_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), rooflineTotalCycles));
    module->setAttr("ascend.simple_sum_cycles",
                    IntegerAttr::get(IntegerType::get(module.getContext(), 64), simpleSumCycles));
    
    // Generate trace with loop unrolling (limit to 100 iterations for visualization)
    generatePerfettoTrace(scheduler, "pipeline_trace.json", oneIterCycles, rooflineTotalCycles, 100);
  }
};

} // namespace
} // namespace ascend
} // namespace mlir
