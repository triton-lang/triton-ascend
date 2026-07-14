add_triton_library(GatherOptimization
  GatherOptimization.cpp

  DEPENDS
  GatherOptimizationConversionPassIncGen

  LINK_LIBS
  BiShengIRAnnotationDialect
  BiShengIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)
