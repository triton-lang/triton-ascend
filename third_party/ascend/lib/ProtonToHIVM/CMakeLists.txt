add_triton_library(AscendProtonToHIVM
  ProtonToHIVM.cpp

  LINK_LIBS PUBLIC
  BiShengIRHIVMDialect
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRMemRefDialect
  MLIRPass
  MLIRSCFDialect
  MLIRTransforms
  ProtonAnalysis
  ProtonIR
  TritonIR
)

target_include_directories(AscendProtonToHIVM PRIVATE
  ${PROJECT_SOURCE_DIR}/third_party/proton/Dialect/include
  ${PROJECT_BINARY_DIR}/third_party/proton/Dialect/include
)
