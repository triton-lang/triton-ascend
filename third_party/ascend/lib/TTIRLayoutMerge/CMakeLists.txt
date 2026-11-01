add_triton_library(TritonPackInterleavedAccess
  PackInterleavedAccess.cpp

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRIR
  MLIRPass
  MLIRSideEffectInterfaces
  MLIRTransforms
  TritonIR
)
