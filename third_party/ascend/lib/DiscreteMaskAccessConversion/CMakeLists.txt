add_triton_library(DiscreteMaskAccessConversion
  DiscreteMaskAccessConversionPass.cpp
  StoreOwnershipAnalysis.cpp

  DEPENDS
  DiscreteMaskAccessConversionPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  MLIRAnalysis
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  MLIRSupport
  TritonIR
  TritonToLinalg
  TritonMemoryAccessAnalysis
  TritonToStructured
  TritonToLinalg
)
