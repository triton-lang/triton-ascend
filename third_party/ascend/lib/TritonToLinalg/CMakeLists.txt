add_triton_library(TritonToLinalg
        TritonToLinalgPass.cpp
        LoadStoreConverter.cpp
        FunctionConverter.cpp
        ArgMinMaxConverter.cpp
        TritonOpConverter.cpp
        HoistBroadcast.cpp
        BlockPtrAnalysis.cpp
        UseAnalysis.cpp
        ImplicitPermute.cpp
        DescriptorConverter.cpp
        MarkTensorKindPass.cpp
        DevicePrintOffsetRewrite.cpp

  DEPENDS
  TritonToLinalgConversionPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRPass
  MLIRTensorDialect
  MLIRTransforms
  MLIRSupport
  BiShengIRHIVMDialect
  TritonIR
  TritonTransforms
  TritonAnalysis
  TritonMemoryAccessAnalysis
  TritonToGraph
  MLIRTritonNPUUtils
  MLIRSCFTransforms
  MLIRLinalgTransforms
)
