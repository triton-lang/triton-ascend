if(TRITONSIM_HAS_BISHENGIR_HIVM)
  add_compile_definitions(TRITONSIM_HAS_BISHENGIR_HIVM=1)
  include_directories(SYSTEM ${TRITONSIM_BISHENGIR_INCLUDE_DIRS})
endif()

set(ASCEND_MODEL_ANALYSIS_DEPENDS
  AscendModelOpsIncGen
)
if(TRITONSIM_HAS_BISHENGIR_HIVM)
  list(APPEND ASCEND_MODEL_ANALYSIS_DEPENDS
    ${TRITONSIM_BISHENGIR_HIVM_LIBS}
  )
endif()

add_mlir_library(AscendModelAnalysis
  PARTIAL_SOURCES_INTENDED
  HardwareConfig.cpp
  HIVMAnalysis.cpp
  PipelineAnalysis.cpp

  ADDITIONAL_HEADER_DIRS
  ${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include/AscendModel/Analysis

  DEPENDS
  ${ASCEND_MODEL_ANALYSIS_DEPENDS}

  LINK_LIBS PUBLIC
  AscendModelProfile
  AscendModelIR
  MLIRArithDialect
  MLIRFuncDialect
  MLIRIR
  MLIRParser
  MLIRSCFDialect
  MLIRSupport
  LLVMSupport
)

if(TRITONSIM_HAS_BISHENGIR_HIVM)
  target_include_directories(AscendModelAnalysis SYSTEM PUBLIC
    ${TRITONSIM_BISHENGIR_INCLUDE_DIRS}
  )
  target_link_libraries(AscendModelAnalysis PUBLIC
    ${TRITONSIM_BISHENGIR_HIVM_LIBS}
    MLIRBufferizationDialect
    MLIRDLTIDialect
    MLIRDestinationStyleOpInterface
    MLIRLinalgDialect
    MLIRLLVMCommonConversion
    MLIRLLVMDialect
    MLIRMemRefDialect
    MLIRSideEffectInterfaces
    MLIRTensorDialect
    MLIRTransformUtils
    MLIRViewLikeInterface
  )
endif()
