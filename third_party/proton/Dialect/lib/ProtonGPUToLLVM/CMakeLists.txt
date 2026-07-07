add_triton_library(ProtonGPUToLLVM
    AllocateProtonGlobalScratchBuffer.cpp
    AllocateProtonSharedMemory.cpp
    PatternProtonGPUOpToLLVM.cpp
    Utility.cpp

    DEPENDS
    ProtonGPUConversionPassIncGen

    LINK_LIBS PUBLIC
    ProtonIR
    ProtonGPUIR
    ProtonAnalysis
)

if("nvidia" IN_LIST TRITON_CODEGEN_BACKENDS)
  add_subdirectory(ProtonNvidiaGPUToLLVM)
endif()
if("amd" IN_LIST TRITON_CODEGEN_BACKENDS)
  add_subdirectory(ProtonAMDGPUToLLVM)
endif()
