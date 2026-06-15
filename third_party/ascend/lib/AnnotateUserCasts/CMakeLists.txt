add_triton_library(AnnotateUserCasts
  AnnotateUserCasts.cpp

  DEPENDS
  AnnotateUserCastsConversionPassIncGen

  LINK_LIBS
  BiShengIRAnnotationDialect
  BiShengIRDialectUtils
  MLIRIR
  MLIRPass
  MLIRTransforms
  MLIRSupport
  TritonIR
)