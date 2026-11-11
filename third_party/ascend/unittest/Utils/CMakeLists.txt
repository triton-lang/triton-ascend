include(${CMAKE_SOURCE_DIR}/cmake/AddTritonUnitTest.cmake)

add_triton_ut(
  NAME TestTritonAscendUtils
  SRCS ForwardTraversalTest.cpp
  LIBS
    MLIRTritonNPUUtils
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    TritonAscendIR
    BiShengIRHIVMDialect
    MLIRParser
    MLIRArithDialect
    MLIRFuncDialect
    MLIRTensorUtils
)
