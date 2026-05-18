#pragma once
#include "ascend/include/DynamicCVPipeline/ComputeBlockOptPass.h"
#include "ascend/include/DynamicCVPipeline/SplitDataflow/RefineArgsBlockId.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"
#include "ascend/include/DynamicCVPipeline/StandardizeOp.h"
#include "ascend/include/TritonToLinalg/Passes.h"
#include "ascend/include/TritonControlFlowOpt/Passes.h"
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h"
#include "ascend/include/DynamicCVPipeline/AddControlFlowCondition.h"
#include "ascend/include/DynamicCVPipeline/AnalyzeDataFlow.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.h"
#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferInnerScope.h"
#include "ascend/include/DynamicCVPipeline/Passes.h"
#include "ascend/include/DynamicCVPipeline/RemoveAttributes.h"
#include "ascend/include/TritonToStructured/Passes.h"
#include "ascend/include/TritonToAnnotation/Passes.h"
#include "ascend/include/TritonToUnstructure/Passes.h"
#include "ascend/include/TritonToHIVM/Passes.h"
#include "ascend/include/TritonToHFusion/Passes.h"
#include "ascend/include/TritonToLLVM/Passes.h"
#include "ascend/include/AutoBlockify/Passes.h"
// #include "amd/include/Dialect/TritonAMDGPU/IR/Dialect.h"
// #include "amd/include/TritonAMDGPUTransforms/Passes.h"
// #include "third_party/nvidia/include/Dialect/NVGPU/IR/Dialect.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/TritonGPU/IR/Dialect.h"
#include "triton/Dialect/TritonNvidiaGPU/IR/Dialect.h"

// Below headers will allow registration to ROCm passes
// #include "TritonAMDGPUToLLVM/Passes.h"
// #include "TritonAMDGPUTransforms/Passes.h"
// #include "TritonAMDGPUTransforms/TritonGPUConversion.h"

#include "triton/Dialect/Triton/Transforms/Passes.h"
#include "triton/Dialect/TritonGPU/Transforms/Passes.h"
#include "triton/Dialect/TritonNvidiaGPU/Transforms/Passes.h"

// #include "nvidia/include/NVGPUToLLVM/Passes.h"
// #include "nvidia/include/TritonNVIDIAGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonGPUToLLVM/Passes.h"
#include "triton/Conversion/TritonToTritonGPU/Passes.h"
#include "triton/Target/LLVMIR/Passes.h"

#include "mlir/Dialect/LLVMIR/NVVMDialect.h"
#include "mlir/Dialect/LLVMIR/ROCDLDialect.h"
#include "mlir/InitAllPasses.h"

#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"

namespace mlir {
namespace test {
// void registerTestAliasPass();
// void registerTestAlignmentPass();
// void registerTestAllocationPass();
// void registerTestMembarPass();
} // namespace test
} // namespace mlir

inline void registerTritonDialects(mlir::DialectRegistry &registry) {
  mlir::registerAllPasses();
  mlir::registerTritonPasses();
  mlir::triton::gpu::registerTritonGPUPasses();
  // mlir::registerTritonNvidiaGPUPasses();
  // mlir::test::registerTestAliasPass();
  // mlir::test::registerTestAlignmentPass();
  // mlir::test::registerTestAllocationPass();
  // mlir::test::registerTestMembarPass();
  mlir::triton::registerConvertTritonToTritonGPUPass();
  mlir::triton::registerAllocateSharedMemoryPass();
  // mlir::triton::registerConvertTritonGPUToLLVMPass();
  // mlir::triton::registerConvertNVGPUToLLVMPass();
  // mlir::triton::registerDecomposeUnsupportedNVIDIAConversions();
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
  mlir::registerLLVMDIScope();

  // TritonAMDGPUToLLVM passes
  // mlir::triton::registerConvertTritonAMDGPUToLLVM();
  // mlir::triton::registerConvertBuiltinFuncToLLVM();
  // mlir::triton::registerDecomposeUnsupportedAMDConversions();
  // mlir::triton::registerOptimizeAMDLDSUsage();

  // TritonAMDGPUTransforms passes
  // mlir::registerTritonAMDGPUAccelerateMatmul();
  // mlir::registerTritonAMDGPUOptimizeEpilogue();
  // mlir::registerTritonAMDGPUReorderInstructions();
  // mlir::registerTritonAMDGPUStreamPipelineV2();
  // mlir::registerTritonAMDGPUCanonicalizePointers();
  // mlir::registerTritonAMDGPUConvertToBufferOps();

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


  // TODO: register Triton & TritonGPU passes
  registry.insert<mlir::triton::TritonDialect, mlir::cf::ControlFlowDialect,
                  /*mlir::triton::nvidia_gpu::TritonNvidiaGPUDialect,*/
                  mlir::triton::gpu::TritonGPUDialect, mlir::math::MathDialect,
                  mlir::arith::ArithDialect, mlir::scf::SCFDialect,
                  mlir::gpu::GPUDialect, mlir::LLVM::LLVMDialect,
                  mlir::NVVM::NVVMDialect, /*mlir::triton::nvgpu::NVGPUDialect,*/
                  /*mlir::triton::amdgpu::TritonAMDGPUDialect,*/
                  mlir::ROCDL::ROCDLDialect,
                  mlir::triton::ascend::TritonAscendDialect,
                  mlir::hivm::HIVMDialect, mlir::scope::ScopeDialect, mlir::hacc::HACCDialect,
                  mlir::annotation::AnnotationDialect,
                  mlir::tensor::TensorDialect, mlir::linalg::LinalgDialect,
                  mlir::memref::MemRefDialect, mlir::bufferization::BufferizationDialect,
                  mlir::func::FuncDialect>();
}
