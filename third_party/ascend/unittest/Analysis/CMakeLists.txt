include(${CMAKE_SOURCE_DIR}/cmake/AddTritonUnitTest.cmake)

# Link the compiler objects into this executable so ELF wrapping observes
# their deallocations; calls inside a shared compiler library cannot be wrapped.
get_property(offset_cache_test_libs GLOBAL PROPERTY TRITON_LIBS)

add_triton_ut(
  NAME TestTritonOffsetCacheLifetime
  SRCS OffsetCacheLifetimeTest.cpp
  LIBS
    ${offset_cache_test_libs}
    TritonNVIDIAGPUToLLVM
    BiShengIRAnnotationDialect
    BiShengIRHACCDialect
    BiShengIRHFusionDialect
    BiShengIRHIVMDialect
    BiShengIRMathExtDialect
    BiShengIRMemRefExtDialect
    BiShengIRScopeDialect
    BiShengIRSymbolDialect
    BiShengIRTensorDialect
    MLIRParser
)

# Observe real BlockArgumentImpl deallocation without production hooks.
# Both forms of operator delete are used by supported Linux/ELF toolchains.
target_link_options(TestTritonOffsetCacheLifetime PRIVATE
  "LINKER:--wrap=_ZdlPv"
  "LINKER:--wrap=_ZdlPvm"
)

add_triton_ut(
  NAME TestPublicPtrAnalysis
  SRCS
    PtrAnalysisTest.cpp
    PtrAnalysisAddressTest.cpp
    PtrAnalysisBitcastTest.cpp
    PtrAnalysisTestUtils.cpp
    PtrAnalysisShapeTest.cpp
    PtrAnalysisIntegerTest.cpp
    BlockPtrAnalysisTest.cpp
    PtrAnalysisLifecycleTest.cpp
    PtrAnalysisExtensionTest.cpp
    PtrAnalysisControlFlowTest.cpp
  LIBS
    TritonPtrAnalysis
    TritonIR
    TritonGPUIR
    TritonGPUTransforms
    TritonNvidiaGPUTransforms
    MLIRArithDialect
    MLIRFuncDialect
    MLIRIR
    MLIRParser
    MLIRSCFDialect
    MLIRTensorDialect
    MLIRSupport
    MLIRTransforms
)
