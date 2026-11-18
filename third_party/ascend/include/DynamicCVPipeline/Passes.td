/*
 * Copyright (c) Huawei Technologies Co.
 * Licensed under the MIT license.
 */

#ifndef TRITON_DYNAMIC_CV_PIPELINE_PASSES
#define TRITON_DYNAMIC_CV_PIPELINE_PASSES

include "mlir/Pass/PassBase.td"

def AddDynamicCVPipeline : Pass<"add_dynamic_cv_pipeline", "mlir::ModuleOp"> {
    let summary = "Add dynamic cv pipeline based on ssbufer";
    let constructor = "triton::createAddDynamicCVPipelinePass()";
    let options = [
        Option<"compileOn91095", "compile-on-910-95",
        "bool", /*default*/"false",
        "compile on 910_95">,
        Option<"mainLoopUnrollFactor", "main-loop-unroll-factor",
        "int", /*default*/"1",
        "Unroll factor of the main loop; <= 1 disables the unroll">,
        Option<"demoteF32Reduction", "demote-f32-reduction",
        "bool", /*default*/"false",
        "Demote f32 reduction between extf and truncf to f16">,
        Option<"quietVariantTrials", "quiet-variant-trials",
        "bool", /*default*/"true",
        "Discard what the variant search's trials print; turn off to see why a trial failed">,
        Option<"reorderSeed", "reorder-seed",
        "int64_t", /*default*/"-1",
        "Compile this operation ordering instead of searching for one; -1 searches as usual">
    ];
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect", "LLVM::LLVMDialect"];
}

def PreCheckAvailablePass : Pass<"pre-check-dynamic-cv-pipeline-available", "mlir::ModuleOp"> {
  let summary = "Check whether dynamic cv pipeline can process the input module";
  let constructor = "triton::createPreCheckAvailablePass()";
  let dependentDialects = ["scope::ScopeDialect"];
}

def OpClassifierPass : Pass<"op-classifier", "mlir::ModuleOp"> {
  let summary = "Classify operations as CUBE or VECTOR";
  let constructor = "triton::createOpClassifierPass()";
}

def PlanComputeBlockPass : Pass<"plan-compute-block", "mlir::ModuleOp"> {
  let summary = "Plan compute block";
  let constructor = "triton::createPlanComputeBlockPass()";
}

def AddBlockIdForControlOpsPass : Pass<"add-block-id-for-control-ops", "mlir::ModuleOp"> {
    let summary = "Add block_id attribute to control flow operations";
    let constructor = "triton::createAddBlockIdForControlOpsPass()";
}

def DataDependencyAnalysisPass : Pass<"data-dependency-analysis", "mlir::ModuleOp"> {
    let summary = "Analyze data dependencies between Vector and Cube blocks";
    let constructor = "triton::createDataDependencyAnalysisPass()";
}

def InterCoreTransferAndSyncPass : Pass<"inter-core-transfer-and-sync", "mlir::ModuleOp"> {
    let summary = "Insert inter-core transfer operations and PIPE_S synchronization between Vector and Cube cores";
    let constructor = "triton::createInterCoreTransferAndSyncPass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def MainLoopUnroll : Pass<"main-loop-unroll", "mlir::ModuleOp"> {
    let summary = "Unroll the main computation loop by a given factor";
    let description = [{
        Unrolls the main loop, i.e. the loop `mark-main-loop` marks with
        `ssbuffer.main_loop` because it carries the cube <-> vector transfers.
        Those transfers only exist further down the pipeline, so the loop is
        located by running `add-block-id-for-control-ops`,
        `data-dependency-analysis`, `inter-core-transfer-and-sync` and
        `mark-main-loop` on a throw-away clone of the module.

        Expects compute blocks to be planned (runs after `plan-compute-block`)
        and must run before `split-dataflow`: the transfers, sync flags and
        multi buffers are then planned for each unrolled copy separately. The
        compute block ids of the copies are shifted so that every copy forms
        blocks of its own. A factor <= 1 leaves the module untouched.
    }];
    let constructor = "triton::createMainLoopUnrollPass()";
    let options = [
        Option<"unrollFactor", "unroll-factor",
        "int", /*default*/"1",
        "Unroll factor of the main loop; <= 1 disables the pass">
    ];
    // The probe pipeline runs inter-core-transfer-and-sync, so every dialect it
    // may create ops from has to be loaded before this pass starts.
    let dependentDialects = ["hivm::HIVMDialect", "arith::ArithDialect", "scf::SCFDialect",
        "func::FuncDialect", "linalg::LinalgDialect",
        "tensor::TensorDialect", "memref::MemRefDialect",
        "bufferization::BufferizationDialect", "hivm::HIVMDialect",
        "scope::ScopeDialect", "annotation::AnnotationDialect",
        "LLVM::LLVMDialect"];
}

def MarkMainLoopPass : Pass<"mark-main-loop", "mlir::ModuleOp"> {
    let summary = "Mark main computation loops containing FixpipeOp or CopyOp with main_loop attribute";
    let constructor = "triton::createMarkMainLoopPass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def SeparateCVScopePass : Pass<"separate-cv-scope", "mlir::ModuleOp"> {
    let summary = "separate ops into cv scope";
    let constructor = "triton::createSeparateCVScopePass()";
    let dependentDialects = ["hivm::HIVMDialect", "bufferization::BufferizationDialect", "scope::ScopeDialect", "annotation::AnnotationDialect"];
}

def PreserveControlAttrsCanonicalizePass : Pass<"preserve-control-attrs-canonicalize", "mlir::ModuleOp"> {
    let summary = "Canonicalize while preserving selected control-flow attrs";
    let constructor = "triton::createPreserveControlAttrsCanonicalizePass()";
}

def ComputeBlockOptPass : Pass<"compute-block-id-opt", "mlir::ModuleOp"> {
  let summary = "Block ID Optimization Pass";
  let constructor = "triton::createComputeBlockOptPass()";
}

def EstimateCVPipelineCostPass : Pass<"estimate-cv-pipeline-cost", "mlir::ModuleOp"> {
  let summary = "Estimate the execution cost of the dynamic CV pipeline IR";
  let description = [{
    Annotates the module with an estimated cycle count for the CV-pipelined IR.
    Must run before remove-ssbuf-attr, since the estimate is driven by the
    ssbuffer.* attributes (core_type in particular) that pass strips.

    The estimate is advisory: it never mutates compute IR and never fails the
    pipeline. It exists so that alternative CV-pipeline IR variants can be
    ranked against each other.
  }];
  let constructor = "triton::createEstimateCVPipelineCostPass()";
  let dependentDialects = ["scope::ScopeDialect", "hivm::HIVMDialect"];
}

def RemoveSsbufAttrPass : Pass<"remove-ssbuf-attr", "mlir::ModuleOp"> {
  let summary = "Remove ssbuf attributes";
  let constructor = "triton::createRemoveSsbufAttrPass()";
}

def CheckUnsupportedScenarioPass : Pass<"ssbuf-check-unsupported-scenario", "mlir::ModuleOp"> {
  let summary = "Check unsupported scenario pass for ssbuf";
  let constructor = "triton::createCheckUnsupportedScenarioPass()";
  let dependentDialects = ["memref::MemRefDialect"];
}

#endif // TRITON_DYNAMIC_CV_PIPELINE_PASSES
