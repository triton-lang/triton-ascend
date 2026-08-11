add_mlir_library(AscendModelProfile
  MicrobenchmarkProfile.cpp

  ADDITIONAL_HEADER_DIRS
  ${TRITON_ASCEND_COSTMODEL_SOURCE_DIR}/include/AscendModel/Profile

  LINK_LIBS PUBLIC
  MLIRSupport
  LLVMSupport
)
