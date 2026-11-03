add_triton_library(TritonToTensorView
  TensorViewBuilder.cpp
  TritonToTensorView.cpp

  DEPENDS
  TritonToTensorViewPassIncGen

  LINK_LIBS PUBLIC
  BiShengIRTensorViewDialect
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRTransforms
  TritonIR
)

target_include_directories(TritonToTensorView PUBLIC
  ${ASCENDNPU_IR_SRC_DIR}/bishengir/include
  ${ASCENDNPU_IR_BINARY_DIR}/bishengir/include
  ${CMAKE_CURRENT_SOURCE_DIR}/../../include
  ${CMAKE_CURRENT_BINARY_DIR}/../../include
)
