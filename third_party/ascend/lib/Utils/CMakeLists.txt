add_triton_library(MLIRTritonNPUUtils
        Utils.cpp
        InterleaveOptimization.cpp

  # Headers pull in HIVM.h; wait for tablegen on clean parallel builds.
  DEPENDS
  BiShengIRHIVMEnumsIncGen

  LINK_LIBS PUBLIC
  MLIRIR
  TritonIR
)
