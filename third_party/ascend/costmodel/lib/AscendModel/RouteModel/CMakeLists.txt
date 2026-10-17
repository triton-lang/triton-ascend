# The source-tree profile is a fallback for native tests and standalone C++
# users. Python production wiring passes the packaged profile path explicitly.
add_compile_definitions(
  TRITON_ASCEND_SIMD_SIMT_PROFILE_PATH="${CMAKE_CURRENT_SOURCE_DIR}/../../../profiles/simd_simt/ascend_950pr_simd_simt_v1.json"
)

add_mlir_library(AscendModelRouteModel
  ../Analysis/SimtAnchorAnalysis.cpp
  ../Analysis/StagePartitioner.cpp
  ../Analysis/SimtRegisterPressureAnalysis.cpp
  StageCostModels.cpp
  CubeGridCostModel.cpp
  StageRouteCostModel.cpp
  SimdSimtCostModel.cpp
  ../Transforms/SelectSimdSimtCostModel.cpp
  ../Transforms/MaterializeSimtScopes.cpp

  ADDITIONAL_HEADER_DIRS
  ${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include/AscendModel/RouteModel

  DEPENDS
  AscendModelOpsIncGen
  AscendModelTransformsPassIncGen

  LINK_LIBS PUBLIC
  AscendModelProfile
  TritonIR
  MLIRIR
  MLIRAnalysis
  MLIRPass
  MLIRSupport
  MLIRTransformUtils
  LLVMSupport
)
