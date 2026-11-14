llvm_canonicalize_cmake_booleans(
  MLIR_ENABLE_BINDINGS_PYTHON
  TRITON_BUILD_PROTON
)

configure_lit_site_cfg(
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.site.cfg.py.in
  ${CMAKE_CURRENT_BINARY_DIR}/lit.site.cfg.py
  MAIN_CONFIG
  ${CMAKE_CURRENT_SOURCE_DIR}/lit.cfg.py
)

set(TRITON_TEST_DEPENDS
  triton-opt
)

set(FILECHECK_PATH "${LLVM_LIBRARY_DIR}/../bin/FileCheck")
set(LIT_ARGS "-Dfilecheck=${FILECHECK_PATH}")

add_lit_testsuite(check-triton-ascend-lit-tests "Running the triton regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  ARGS ${LIT_ARGS}
  DEPENDS ${TRITON_TEST_DEPENDS}
  )

set_target_properties(check-triton-ascend-lit-tests PROPERTIES FOLDER "Tests")

add_lit_testsuites(TRITON-LIT-TESTS ${CMAKE_CURRENT_SOURCE_DIR} DEPENDS ${TRITON_TEST_DEPENDS})

include(${CMAKE_SOURCE_DIR}/cmake/AddTritonUnitTest.cmake)
add_triton_ut(
  NAME TestAscendBatchRewriteWorkspace
  SRCS BlockBatching/WorkspaceTest.cpp
  LIBS MLIRIR
)

# The owned-fact codec and the affine profile acceptance domain are pinned by
# two regressions that provide their own main() and deliberately link no MLIR,
# so they cannot go through add_triton_ut (which links GTest's main). They were
# previously built only by an opportunistic pytest that skips whenever no host
# compiler or LLVM development install is present, which left their invariants
# unguarded in an ordinary build.
set(_BLOCK_BATCHING_LIB ${CMAKE_CURRENT_SOURCE_DIR}/../lib/TritonToGraph/BlockBatching)

add_executable(TestAscendBatchEffectProjection
  BlockBatching/EffectProjectionTest.cpp
  ${_BLOCK_BATCHING_LIB}/BatchExpr.cpp
  ${_BLOCK_BATCHING_LIB}/BatchFactsJSON.cpp
)

add_executable(TestAscendBatchAffineProfiles
  BlockBatching/AffineProfilesTest.cpp
  ${_BLOCK_BATCHING_LIB}/BatchExpr.cpp
  ${_BLOCK_BATCHING_LIB}/BatchAffine.cpp
  ${_BLOCK_BATCHING_LIB}/BatchRangeAnalysis.cpp
  ${_BLOCK_BATCHING_LIB}/BatchLoopSchedule.cpp
  ${_BLOCK_BATCHING_LIB}/BatchLoopLegality.cpp
)

foreach(_batch_check TestAscendBatchEffectProjection TestAscendBatchAffineProfiles)
  target_link_libraries(${_batch_check} PRIVATE LLVMSupport)
  if(NOT MSVC)
    # AffineProfilesTest states its contract with assert(); a release build
    # would otherwise define NDEBUG and leave an empty executable passing.
    target_compile_options(${_batch_check} PRIVATE -fno-rtti -UNDEBUG)
  endif()
  add_test(NAME ${_batch_check} COMMAND ${_batch_check})
  add_dependencies(TritonUnitTests ${_batch_check})
endforeach()

if(TRITON_ASCEND_HAS_INPROC_COSTMODEL)
  add_subdirectory(costmodel_ut)
endif()

add_subdirectory(Utils)

# The deallocation observer uses ELF linker wrapping.
if(CMAKE_SYSTEM_NAME STREQUAL "Linux")
  add_subdirectory(Analysis)
endif()
