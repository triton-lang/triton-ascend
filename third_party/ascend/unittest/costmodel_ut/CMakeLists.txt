include(${CMAKE_SOURCE_DIR}/cmake/AddTritonUnitTest.cmake)

add_triton_ut(
  NAME CostModelHardwareConfig
  SRCS HardwareConfigTest.cpp
  LIBS
    AscendModelAnalysis
)

add_triton_ut(
  NAME CostModelPipelineScheduler
  SRCS PipelineSchedulerTest.cpp
  LIBS
    AscendModelAnalysis
    AscendModelIR
)

add_triton_ut(
  NAME CostModelPasses
  SRCS PassesTest.cpp
  DEFS
    TRITON_ASCEND_DAVID_TEST_CONFIG_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../costmodel/configs/ascend_davidv100.json"
    TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json"
  LIBS
    AscendModelTransforms
    AscendModelAnalysis
    AscendModelIR
    BiShengIRScopeDialect
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSCFDialect
    MLIRSupport
)

add_triton_ut(
  NAME SimdSimtCostModel
  SRCS SimdSimtCostModelTest.cpp
  DEFS
    TRITON_ASCEND_SIMD_SIMT_TEST_PROFILE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../costmodel/profiles/simd_simt/david_v100_simd_simt_v1.json"
  LIBS
    AscendModelRouteModel
)
