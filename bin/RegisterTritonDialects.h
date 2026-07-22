#pragma once
#include "ascend/include/AutoBlockify/Passes.h"
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.h"
#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/ComputeBlockOptPass.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"
#include "ascend/include/DynamicCVPipeline/RemoveAttributes.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AddMultiBufferToGMLoadPass.h"
#include "ascend/include/DynamicCVPipeline/SeparateMemoryFromCompute/AsyncLoadHoistingPass.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/RefineArgsBlockId.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"
#include "ascend/include/TritonControlFlowOpt/Passes.h"
#include "ascend/include/TritonToAnnotation/Passes.h"
#include "ascend/include/TritonToGraph/Passes.h"
#include "ascend/include/TritonToHFusion/Passes.h"
#include "ascend/include/TritonToHIVM/Passes.h"
#include "ascend/include/TritonToLLVM/Passes.h"
#include "ascend/include/TritonToLinalg/Passes.h"
#include "ascend/include/TritonToStructured/Passes.h"
#include "ascend/include/TritonToUnstructure/Passes.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "proton/Dialect/include/Conversion/ProtonGPUToLLVM/Passes.h"
#include "proton/Dialect/include/Conversion/ProtonToProtonGPU/Passes.h"
#include "proton/Dialect/include/Dialect/Proton/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/IR/Dialect.h"
#include "proton/Dialect/include/Dialect/ProtonGPU/Transforms/Passes.h"
#include "triton/Dialect/Gluon/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonInstrument/IR/Dialect.h"

// Below headers will allow registration to ROCm passes

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonInstrument/Transforms/Passes.h"

#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/Dialect/LLVMIR/Transforms/InlinerInterfaceImpl.h"
#include "mlir/InitAllPasses.h"

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::triton::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  mlir::triton::instrument::registerTritonInstrumentPasses();
  mlir::triton::gluon::registerGluonPasses();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerRelayoutTritonGPUPass();
  mlir::triton::gpu::registerAllocateSharedMemoryPass();
  mlir::triton::gpu::registerTritonGPUAllocateWarpGroups();
  mlir::triton::gpu::registerTritonGPUGlobalScratchAllocationPass();
  mlir::triton::registerTritonToLinalgPasses();
  mlir::triton::registerTritonControlFlowOptPasses();
  mlir::triton::registerDiscreteMaskAccessConversion();
  mlir::triton::registerTritonToStructuredPasses();
  mlir::triton::registerTritonToAnnotationPasses();
  mlir::triton::registerTritonToUnstructurePasses();
  mlir::triton::registerTritonToHIVMPasses();
  mlir::triton::registerTritonToHFusionPasses();
  mlir::triton::registerTritonToLLVMPasses();
  mlir::triton::registerAutoBlockifyPasses();
  mlir::triton::cfg::registerTritonToGraphPasses();
  mlir::triton::registerBubbleUpOperationPass();

  mlir::registerLLVMDIScope();
  mlir::LLVM::registerInlinerInterface(registry);
  mlir::NVVM::registerInlinerInterface(registry);
  mlir::registerLLVMDILocalVariable();

  // Proton passes
  mlir::triton::proton::registerConvertProtonToProtonGPU();
  mlir::triton::proton::gpu::registerAllocateProtonSharedMemoryPass();
  mlir::triton::proton::gpu::registerAllocateProtonGlobalScratchBufferPass();
  mlir::triton::proton::gpu::registerScheduleBufferStorePass();

  // DynamicCVPipeline passes
  mlir::triton::registerAddDynamicCVPipelinePasses();
  mlir::triton::registerPreCheckAvailablePasses();
  mlir::triton::registerStandardizeOpPasses();
  mlir::triton::registerAddControlFlowConditionPasses();
  mlir::triton::registerAddMultiBufferOuterScopePasses();
  mlir::triton::registerAddMultiBufferInnerScopePasses();
  mlir::triton::registerRemoveSsbufAttrPasses();
  mlir::triton::registerAnalyzeDataFlowPasses();
  mlir::triton::registerComputeBlockOptPasses();
  mlir::triton::registerPlanComputeBlockPasses();
  mlir::triton::registerOpClassifierPass();
  mlir::triton::registerRefineArgsBlockIdPasses();
  mlir::triton::registerAsyncLoadHoistingPasses();
  mlir::triton::registerAddMultiBufferToGMLoadPasses();

  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
                  mlir::triton::gpu::TritonGPUDialect,
                  mlir::triton::instrument::TritonInstrumentDialect,
                  mlir::math::MathDialect, mlir::arith::ArithDialect,
                  mlir::scf::SCFDialect, mlir::tensor::TensorDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect, mlir::triton::proton::ProtonDialect,
                  mlir::triton::proton::gpu::ProtonGPUDialect,
                  mlir::ROCDL::ROCDLDialect, mlir::triton::gluon::GluonDialect,
                  mlir::triton::ascend::TritonAscendDialect,
                  mlir::hivm::HIVMDialect, mlir::scope::ScopeDialect,
                  mlir::hacc::HACCDialect, mlir::annotation::AnnotationDialect,
                  mlir::hfusion::HFusionDialect, mlir::tensor::TensorDialect,
                  mlir::linalg::LinalgDialect, mlir::memref::MemRefDialect,
                  mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect>();
}
