add_triton_library(NVWSIR
  Dialect.cpp
  Ops.cpp

  DEPENDS
  NVWSTableGen
  NVWSAttrDefsIncGen

  LINK_LIBS PUBLIC
  TritonIR
  TritonGPUIR
)

target_include_directories(NVWSIR PUBLIC
  ${PROJECT_SOURCE_DIR}/third_party/nvidia/include
  ${CMAKE_BINARY_DIR}/third_party/nvidia/include
)
