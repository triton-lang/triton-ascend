include(${CMAKE_SOURCE_DIR}/cmake/AddTritonUnitTest.cmake)

add_executable(GraphOptimizationResourceCostModelTest
  GraphOptimizationResourceCostModelTest.cpp
)

target_link_libraries(GraphOptimizationResourceCostModelTest
  PRIVATE
  GTest::gtest_main
  gmock
  TritonToGraph
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  MLIRTensorDialect
  TritonGPUTransforms
  TritonNvidiaGPUTransforms
)

if(NOT MSVC)
  target_compile_options(GraphOptimizationResourceCostModelTest PRIVATE -fno-rtti)
endif()

gtest_discover_tests(GraphOptimizationResourceCostModelTest DISCOVERY_TIMEOUT 60)
add_dependencies(TritonUnitTests GraphOptimizationResourceCostModelTest)
