include(${CMAKE_SOURCE_DIR}/cmake/AddTritonUnitTest.cmake)

add_triton_ut(
  NAME ExecutionPolicy
  SRCS
    ExecutionPolicyTest.cpp
    ../../lib/TritonToLinalg/ExecutionPolicy.cpp
  LIBS
    MLIRArithDialect
    MLIRFuncDialect
    MLIRLinalgDialect
    MLIRMathDialect
    MLIRMemRefDialect
    MLIRParser
    MLIRSCFDialect
    MLIRTensorDialect
)
