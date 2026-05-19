add_triton_library(TritonControlFlowOpt
        TritonControlFlowOptPass.cpp

  DEPENDS
  TritonControlFlowOptPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRFuncDialect
  MLIRIR
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  TritonIR
)
