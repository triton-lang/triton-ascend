if(NOT TARGET NVGPUTableGen)
  add_subdirectory(
    ${PROJECT_SOURCE_DIR}/third_party/nvidia/include/Dialect/NVGPU/IR
    ${CMAKE_BINARY_DIR}/third_party/nvidia/include/Dialect/NVGPU/IR)
endif()
if(NOT TARGET NVGPUIR)
  add_subdirectory(
    ${PROJECT_SOURCE_DIR}/third_party/nvidia/lib/Dialect/NVGPU/IR
    ${CMAKE_BINARY_DIR}/third_party/nvidia/lib/Dialect/NVGPU/IR)
endif()

add_triton_library(TritonInstrumentToLLVM
    InstrumentationToLLVM.cpp

    DEPENDS
    NVGPUTableGen
    NVGPUAttrDefsIncGen

    LINK_LIBS PUBLIC
    MLIRIR
    MLIRPass
    TritonIR
    TritonGPUIR
    TritonInstrumentIR
    TritonNvidiaGPUIR
    NVGPUIR
)
