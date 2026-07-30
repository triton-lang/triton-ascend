add_triton_library(CVSplitScheduling
  CVSplitScheduling.cpp
  CrossScopeTransfers.cpp
  classifyAllOps.cpp
  DependencyScheduler.cpp
  PreCheck.cpp
  QStaging.cpp
  ScopeSeparation.cpp
  UnfusePVMatmuls.cpp

  DEPENDS
  CVSplitSchedulingPassIncGen

  LINK_LIBS
  BiShengIRHIVMDialect
  BiShengIRScopeDialect
  BiShengIRAnnotationDialect
  MLIRFuncDialect
  MLIRLinalgDialect
  MLIRSCFDialect
  MLIRSCFTransforms
  MLIRTransforms
  DynamicCVPipeline
)
