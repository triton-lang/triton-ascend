add_triton_library(TritonToUnstructure
        IndirectAtomicUtils.cpp
        UnstructureConversionPass.cpp
        OffsetAnalysis.cpp
        BubbleUpOperation.cpp
        ReplaceArguments.cpp

  DEPENDS
  TritonToUnstructureConversionPassIncGen
  # Headers pull in HIVM.h; wait for tablegen on clean parallel builds.
  BiShengIRHIVMEnumsIncGen

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
)
