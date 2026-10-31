add_triton_library(TritonMemoryAccessCore
  OpFoldResultUtils.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRSupport
)

add_triton_library(TritonMemoryAccessAnalysis
  MaskAnalysis.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMemRefDialect
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  MLIRTransforms
  TritonIR
  TritonMemoryAccessCore
)

# Consumer-independent numerical analysis for all Triton pointer forms.
add_triton_library(TritonPtrAnalysis
  IntegerUtils.cpp
  PtrAnalysis.cpp
  PtrAnalysisRules/CommonPtrRules.cpp
  PtrAnalysisRules/StructuredPtrRules.cpp
  PtrInfoUtils.cpp
  PtrAnalysisRules/ControlFlowPtrRules.cpp
  PtrAnalysisInfo.cpp

  LINK_LIBS PUBLIC
  LLVMCore
  MLIRArithDialect
  MLIRIR
  MLIRSideEffectInterfaces
  MLIRTensorDialect
  MLIRSupport
  TritonIR
  TritonMemoryAccessCore
)
