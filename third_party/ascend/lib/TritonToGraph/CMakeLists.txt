add_triton_library(TritonToGraph
  AliasAnalysis.cpp
  ControlFlowGraph.cpp
  ControlFlowGraphBuilder.cpp
  DataflowGraph.cpp
  GraphOptimizationContext.cpp
  GraphOptimizationPass.cpp
  MemorySsaBuilder.cpp
  PermutationAnalysis.cpp
  Passes.cpp
  Rules/LoadStoreTransposeRule.cpp
  Rules/TransposePointwiseReorderRule.cpp
  Rules/UBPreloadRule.cpp

  DEPENDS
  TritonToGraphPassIncGen

  LINK_LIBS PUBLIC
  MLIRArithDialect
  MLIRControlFlowDialect
  MLIRDialectUtils
  MLIRIR
  MLIRMathDialect
  MLIRFuncDialect
  MLIRPass
  MLIRSCFDialect
  MLIRSupport
  MLIRTransforms
  TritonIR
  TritonTransforms
)
