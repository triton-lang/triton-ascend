include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)
include_directories(${CMAKE_CURRENT_BINARY_DIR}/include)
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/lib)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/cmake")

# Triton Ascend is dependent on AscendNPU IR
set(ASCENDNPU_IR_SRC_DIR "${CMAKE_CURRENT_SOURCE_DIR}/AscendNPU-IR" CACHE PATH
    "Path to AscendNPU-IR source root")
set(ASCENDNPU_IR_BINARY_DIR "${CMAKE_CURRENT_BINARY_DIR}/AscendNPU-IR")

set(BISHENGIR_ENABLE_A5_UNPUBLISHED_FEATURES ON)
set(BISHENGIR_BUILD_STANDALONE_IR_ONLY ON)

# Temporarily save and clear the RULE_LAUNCH_* attributes in the current directory to ensure AscendNPU-IR is not affected.
get_property(_saved_launch_compile DIRECTORY PROPERTY RULE_LAUNCH_COMPILE)
get_property(_saved_launch_link DIRECTORY PROPERTY RULE_LAUNCH_LINK)
set_property(DIRECTORY PROPERTY RULE_LAUNCH_COMPILE "")
set_property(DIRECTORY PROPERTY RULE_LAUNCH_LINK "")

add_subdirectory(${ASCENDNPU_IR_SRC_DIR} ${ASCENDNPU_IR_BINARY_DIR})

# restore properties
set_property(DIRECTORY PROPERTY RULE_LAUNCH_COMPILE ${_saved_launch_compile})
set_property(DIRECTORY PROPERTY RULE_LAUNCH_LINK ${_saved_launch_link})

include_directories(${ASCENDNPU_IR_SRC_DIR}/bishengir/include)
include_directories(${ASCENDNPU_IR_BINARY_DIR}/bishengir/include) # Tablegen'd files

add_subdirectory(include)
add_subdirectory(lib)


option(TRITON_ASCEND_ENABLE_COSTMODEL_INPROC "Enable in-process costmodel C++ bridge" ON)
if(TRITON_ASCEND_ENABLE_COSTMODEL_INPROC)
  set(TRITON_ASCEND_COSTMODEL_SOURCE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/costmodel" CACHE PATH "Path to vendored vTriton source root")
  set(_costmodel_include_dir "${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include/AscendModel")
  set(_costmodel_lib_dir "${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/lib/AscendModel")

  if(EXISTS "${_costmodel_include_dir}/CMakeLists.txt" AND EXISTS "${_costmodel_lib_dir}/CMakeLists.txt")
    # Generate HardwareParams.h required by AscendModel analysis/transforms.
    set(ASCEND_CUBE_TFLOPS 320 CACHE STRING "Cube Core FP16 TFLOPS")
    set(ASCEND_CUBE_M 16 CACHE STRING "Cube Core M dimension")
    set(ASCEND_CUBE_N 16 CACHE STRING "Cube Core N dimension")
    set(ASCEND_CUBE_K 16 CACHE STRING "Cube Core K dimension")
    set(ASCEND_VECTOR_TFLOPS 10 CACHE STRING "Vector Core FP32 TFLOPS")
    set(ASCEND_VECTOR_WIDTH 256 CACHE STRING "Vector width in bits")
    set(ASCEND_HBM_BW 1555 CACHE STRING "HBM Bandwidth GB/s")
    set(ASCEND_L2_SIZE 192 CACHE STRING "L2 Cache Size MB")
    set(ASCEND_L1_SIZE 1024 CACHE STRING "L1 Buffer Size KB")
    set(_costmodel_gen_include "${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/include")
    file(MAKE_DIRECTORY "${_costmodel_gen_include}/AscendModel")
    configure_file(
      "${_costmodel_include_dir}/HardwareParams.h.in"
      "${_costmodel_gen_include}/AscendModel/HardwareParams.h"
      @ONLY
    )
    include_directories("${_costmodel_gen_include}")
    # TableGen outputs (*.inc) from AscendModel/include land in this binary dir.
    include_directories("${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/include_AscendModel")
    include_directories("${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include")

    add_subdirectory(${_costmodel_include_dir} ${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/include_AscendModel)
    add_subdirectory(${_costmodel_lib_dir} ${CMAKE_BINARY_DIR}/third_party/ascend/costmodel_build/lib_AscendModel)
    set(TRITON_ASCEND_HAS_INPROC_COSTMODEL 1)
  else()
    message(WARNING "Minimal costmodel sources not found under ${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}; in-process costmodel bridge disabled")
    set(TRITON_ASCEND_HAS_INPROC_COSTMODEL 0)
  endif()
else()
  set(TRITON_ASCEND_HAS_INPROC_COSTMODEL 0)
endif()

add_triton_plugin(TritonAscend
  ${CMAKE_CURRENT_SOURCE_DIR}/triton_ascend.cc
  ${CMAKE_CURRENT_SOURCE_DIR}/ascend_ir.cc
  LINK_LIBS
  TritonToLinalg
  BiShengIRScopeDialect
  BiShengIRHIVMDialect
)

option(TRITON_ENABLE_COVERAGE_LLVM_COV "Enable code llvm-cov coverage tool for Ascend plugin " OFF)
if(TRITON_ENABLE_COVERAGE_LLVM_COV)
  message(STATUS "Enabling llvm-cov coverage tool flags for TritonAscend")
  target_compile_options(TritonAscend PRIVATE
    -fprofile-arcs
    -ftest-coverage
    -O0
    -fprofile-update=atomic
    --coverage
  )
  target_link_options(TritonAscend PRIVATE
    --coverage
    -lgcov
  )
  # branch coverage
  target_compile_definitions(TritonAscend PRIVATE
    COVERAGE_ENABLED=1
  )
endif()


# To enable hitest coverage tool
if(TRITON_ENABLE_COVERAGE_HITEST)
  set_target_properties(TritonAscend PROPERTIES
    RULE_LAUNCH_COMPILE "hitestwrapper"
    RULE_LAUNCH_LINK "hitestwrapper"
  )
endif()

if(TRITON_BUILD_UT)
  add_subdirectory(unittest)
endif()


if(TRITON_ASCEND_HAS_INPROC_COSTMODEL)
  target_compile_definitions(TritonAscend PRIVATE TRITON_ASCEND_HAS_INPROC_COSTMODEL=1)
  target_link_libraries(TritonAscend PRIVATE
    AscendModelIR
    AscendModelTransforms
    AscendModelAnalysis
    MLIROptLib
    MLIRIR
    MLIRParser
    MLIRPass
    MLIRSupport
    MLIRTransforms
    MLIRFuncDialect
    MLIRSCFDialect
    MLIRArithDialect
    MLIRMathDialect
    MLIRTensorDialect
    LLVMSupport
  )
else()
  target_compile_definitions(TritonAscend PRIVATE TRITON_ASCEND_HAS_INPROC_COSTMODEL=0)
endif()
