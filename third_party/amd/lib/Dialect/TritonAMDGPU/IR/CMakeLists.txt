add_triton_library(TritonAMDGPUIR
  Dialect.cpp

  DEPENDS
  TritonAMDGPUTableGen
  TritonAMDGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
  TritonIR
  TritonGPUIR
  TritonAMDUtils
)

target_include_directories(TritonAMDGPUIR PUBLIC
  ${PROJECT_SOURCE_DIR}/third_party/amd/include
  ${CMAKE_BINARY_DIR}/third_party/amd/include
)
