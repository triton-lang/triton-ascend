# Share the pointer/offset analysis without pulling the conversion pass
# (which depends on backend-wide route globals) into CostModel clients.
add_triton_library(TritonOffsetAnalysis
  OffsetAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRTensorDialect
  MLIRSupport
  TritonIR
  TritonGPUIR
  TritonAscendIR
  TritonGPUTransforms
  TritonNvidiaGPUTransforms
  MLIRTensorUtils
  MLIRTritonNPUUtils
  BiShengIRHIVMDialect
  BiShengIRAnnotationDialect
)

add_triton_library(TritonToUnstructure
        IndirectAtomicUtils.cpp
        UnstructureConversionPass.cpp
        BubbleUpOperation.cpp
        ReplaceArguments.cpp

  DEPENDS
  TritonToUnstructureConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonAnalysis
  MLIRSCFTransforms
  TritonOffsetAnalysis
)
