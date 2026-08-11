//===- Passes.td - AscendModel transformation passes -------*- tablegen -*-===//
//
// This file declares the passes for the AscendModel dialect.
//
//===----------------------------------------------------------------------===//

#ifndef ASCEND_MODEL_TRANSFORMS_PASSES_TD
#define ASCEND_MODEL_TRANSFORMS_PASSES_TD

include "mlir/Pass/PassBase.td"

//===----------------------------------------------------------------------===//
// ConvertTritonToAscend
//===----------------------------------------------------------------------===//

def ConvertTritonToAscendPass : Pass<"convert-triton-to-ascend", "ModuleOp"> {
  let summary = "Convert Triton IR to AscendModel IR";
  let description = [{
    This pass converts Triton dialect operations to AscendModel dialect
    operations for performance modeling on Ascend 910B hardware.
  }];
  let dependentDialects = [
    "mlir::ascend::AscendModelDialect",
    "mlir::arith::ArithDialect",
    "mlir::math::MathDialect",
    "mlir::scf::SCFDialect",
    "mlir::tensor::TensorDialect"
  ];
}

//===----------------------------------------------------------------------===//
// InsertDataTransfers
//===----------------------------------------------------------------------===//

def InsertDataTransfersPass : Pass<"insert-data-transfers", "ModuleOp"> {
  let summary = "Insert memory transfer operations between Cube and Vector";
  let description = [{
    This pass analyzes data dependencies between Cube (matrix) and Vector
    (SIMD) operations and inserts the necessary memory transfer operations.

    On Ascend 910B, Cube and Vector have separate memory hierarchies:
    - Cube: HBM → L1 → L0A/L0B → Cube → L0C → HBM
    - Vector: HBM → UB → Vector → UB → HBM

    When data flows between them, explicit transfers are needed:
    - Vector → Cube: insert vector_store (UB→HBM) + cube_load (HBM→L1)
    - Cube → Vector: insert cube_store (L0C→HBM) + vector_load (HBM→UB)
  }];
}

//===----------------------------------------------------------------------===//
// AssignOpIDs
//===----------------------------------------------------------------------===//

def AssignOpIDsPass : Pass<"assign-op-ids", "ModuleOp"> {
  let summary = "Assign unique IDs to operations";
  let description = [{
    This pass assigns unique operation IDs to all AscendModel operations
    for tracking and scheduling purposes.
  }];
}

//===----------------------------------------------------------------------===//
// EstimateCycles
//===----------------------------------------------------------------------===//

def EstimateCyclesPass : Pass<"estimate-cycles", "ModuleOp"> {
  let summary = "Estimate execution cycles for operations";
  let description = [{
    This pass estimates the execution cycles for each AscendModel operation
    based on hardware parameters from JSON config file.

    For loops with dynamic bounds, provide bindings via arg-bindings option:

    Example for Triton kernels (options are space-separated):
      inproc-costmodel --estimate-cycles="arg-bindings=arg2=100,pid_x=0" input.mlir

    Supported binding formats:
      - argN=value or N=value  : function argument %argN
      - pid_x=value            : tt.get_program_id x
      - pid_y=value            : tt.get_program_id y
      - pid_z=value            : tt.get_program_id z

    Alternatively, use loop-trip-counts to directly override trip counts:
      inproc-costmodel --estimate-cycles="loop-trip-counts=4,6588" input.mlir

    To use custom hardware config:
      inproc-costmodel --estimate-cycles="hardware-config=path/to/config.json" input.mlir

    Multiple options (use space, not comma, to separate):
      inproc-costmodel --estimate-cycles="hardware-config=config.json arg-bindings=pid_x=0" input.mlir
  }];

  let options = [
    Option<"argBindingsStr", "arg-bindings", "std::string",
           /*default=*/"\"\"",
           "Bindings for args and program_ids (e.g., 'arg2=100,pid_x=0')">,
    Option<"loopTripCountsStr", "loop-trip-counts", "std::string",
           /*default=*/"\"\"",
           "Direct loop trip count overrides in order (e.g., '4,6588')">,
    Option<"hardwareConfigPath", "hardware-config", "std::string",
           /*default=*/"\"\"",
           "Path to hardware configuration JSON file">
  ];
}

//===----------------------------------------------------------------------===//
// PipelineAnalysis
//===----------------------------------------------------------------------===//

def PipelineAnalysisPass : Pass<"analyze-pipeline", "ModuleOp"> {
  let summary = "Analyze pipeline scheduling";
  let description = [{
    This pass performs pipeline scheduling analysis for AscendModel operations,
    considering hardware unit parallelism and dependencies.

    For loops with dynamic bounds (options are space-separated):
      inproc-costmodel --analyze-pipeline="arg-bindings=arg2=100,pid_x=0" input.mlir

    To use custom hardware config:
      inproc-costmodel --analyze-pipeline="hardware-config=path/to/config.json" input.mlir

    Multiple options (use space, not comma, to separate):
      inproc-costmodel --analyze-pipeline="hardware-config=config.json arg-bindings=pid_x=0" input.mlir
  }];

  let options = [
    Option<"argBindingsStr", "arg-bindings", "std::string",
           /*default=*/"\"\"",
           "Bindings for args and program_ids (e.g., 'arg2=100,pid_x=0')">,
    Option<"loopTripCountsStr", "loop-trip-counts", "std::string",
           /*default=*/"\"\"",
           "Direct loop trip count overrides in order (e.g., '4,6588')">,
    Option<"hardwareConfigPath", "hardware-config", "std::string",
           /*default=*/"\"\"",
           "Path to hardware configuration JSON file">
  ];
}

//===----------------------------------------------------------------------===//
// HIVMAnalysis
//===----------------------------------------------------------------------===//

def HIVMAnalysisPass : Pass<"analyze-hivm", "ModuleOp"> {
  let summary = "Analyze native HIVM IR scheduling and synchronization";
  let description = [{
    This pass runs the HIVM-native scheduler directly on parsed HIVM IR.
    It expects `hivm.hir.*` ops to be available as native MLIR ops, and
    reports per-pipe utilization, sync costs, and critical-path timing.

    Example:
      inproc-costmodel --analyze-hivm input.npuir.mlir
      inproc-costmodel --analyze-hivm="hardware-config=configs/ascend_910b.json arg-bindings=arg10=128" input.npuir.mlir
  }];

  let options = [
    Option<"schedulerMode", "scheduler", "std::string",
           /*default=*/"\"static\"",
           "Scheduler backend: static or des">,
    Option<"argBindingsStr", "arg-bindings", "std::string",
           /*default=*/"\"\"",
           "Bindings for dynamic HIVM values (e.g., 'arg10=128,arg11=64')">,
    Option<"hardwareConfigPath", "hardware-config", "std::string",
           /*default=*/"\"\"",
           "Path to hardware configuration JSON file">,
    Option<"reportFile", "report-file", "std::string",
           /*default=*/"\"\"",
           "Optional file to write the textual HIVM report to">,
    Option<"perfettoTraceFile", "perfetto-trace-file", "std::string",
           /*default=*/"\"\"",
           "Optional Perfetto-compatible JSON trace output file">
  ];
}

//===----------------------------------------------------------------------===//
// PerfReport
//===----------------------------------------------------------------------===//
def PerfReportPass : Pass<"perf-report", "ModuleOp"> {
  let summary = "Generate performance report";
  let description = [{
    This pass generates a comprehensive performance report including
    roofline analysis, utilization metrics, and bottleneck identification.
  }];
}

//===----------------------------------------------------------------------===//
// SelectSimdSimtCostModel
//===----------------------------------------------------------------------===//

def SelectSimdSimtCostModelPass
    : Pass<"select-simd-simt-costmodel", "ModuleOp"> {
  let summary = "Select SIMD, SIMT-only, or mixed execution with the C++ cost model";
  let description = [{
    This pass runs the AscendModel C++ SIMD/SIMT candidate model directly on
    Triton IR.  Report mode always records the complete three-candidate
    diagnostic report.  Auto mode first checks calibration coverage and skips
    candidate scoring for out-of-domain kernels; an admitted in-domain decision
    becomes the authoritative execution intent.  A mixed decision passes the
    same immutable anchor plan used for scoring directly to the C++
    materializer, which creates local scope.scope regions without persisting
    per-operation selection attributes.

    Mandatory mixed is intentionally not part of this pass.  Singleton
    legality fast paths can be added in front of the model after the calibrated
    three-candidate path is complete.
  }];

  let options = [
    Option<"mode", "mode", "std::string",
           /*default=*/"\"report\"",
           "Application mode: report or auto">,
    Option<"profilePath", "profile-path", "std::string",
           /*default=*/"\"\"",
           "Versioned SIMD/SIMT calibration profile JSON">,
    Option<"actualTarget", "actual-target", "std::string",
           /*default=*/"\"\"",
           "Backend target used for profile compatibility checks">,
    Option<"numWarps", "num-warps", "int64_t",
           /*default=*/"32",
           "Number of logical warps in the candidate execution plan">,
    Option<"marginRatio", "margin-ratio", "double",
           /*default=*/"0.10",
           "Minimum relative score advantage required for an online action">,
    Option<"compileOn91095", "compile-on-910-95", "bool",
           /*default=*/"false",
           "Whether the target exposes the 910/95 SIMT lowering templates">,
    Option<"reportFile", "report-file", "std::string",
           /*default=*/"\"\"",
           "Optional JSONL path for the C++ selection report">
  ];
}

//===----------------------------------------------------------------------===//
// MaterializeSimtScopes
//===----------------------------------------------------------------------===//

def MaterializeSimtScopesPass
    : Pass<"materialize-simt-scopes", "ModuleOp"> {
  let summary = "Validate the scope-only contract for an admitted mixed route";
  let description = [{
    Production materialization is performed from the immutable anchor plan in
    the selector pass.  This compatibility pass verifies that:

      ascend.simt_costmodel.effective = "mixed_simd_simt"
      scope.scope {vec_mode = "simt"}

    coexist in the IR.  It does not rediscover anchors or read/write a
    per-operation selected marker.
  }];
}


#endif // ASCEND_MODEL_TRANSFORMS_PASSES_TD
