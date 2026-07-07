add_triton_library(NVGPUIR
  Dialect.cpp

  DEPENDS
  NVGPUTableGen
  NVGPUAttrDefsIncGen

  LINK_LIBS PUBLIC
  MLIRLLVMDialect
)

target_include_directories(NVGPUIR PUBLIC
  ${PROJECT_SOURCE_DIR}/third_party/nvidia/include
  ${CMAKE_BINARY_DIR}/third_party/nvidia/include
)
