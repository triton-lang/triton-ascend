add_triton_library(MLIRTritonNPUUtils
        Utils.cpp
        InterleaveOptimization.cpp
        DebugUtils.cpp

  LINK_LIBS PUBLIC
  MLIRIR
  MLIRLLVMDialect
  TritonIR
  TritonMemoryAccessAnalysis
  TritonMemoryAccessCore
  # Utils.cpp includes bishengir/Dialect/HIVM/IR/HIVM.h, which includes the
  # tablegen'd HIVMEnums.h.inc. Link the bishengir dialect so ninja orders
  # the generation before this compile (same pattern as TritonToLinalg and
  # DynamicCVPipeline); otherwise parallel builds race and fail with
  # "HIVMEnums.h.inc file not found".
  BiShengIRHIVMDialect
)
