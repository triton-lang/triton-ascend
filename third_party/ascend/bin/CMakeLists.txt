get_property(triton_libs GLOBAL PROPERTY TRITON_LIBS)

# triton-mlir-opt: Triton-Ascend optimizer combining MLIR and BishengIR dialects
add_llvm_executable(triton-mlir-opt triton-mlir-opt.cpp PARTIAL_SOURCES_INTENDED)
llvm_update_compile_flags(triton-mlir-opt)

target_link_libraries(triton-mlir-opt PRIVATE
  ${triton_libs}
  # MLIR core
  MLIROptLib
  MLIRPass
  MLIRRegisterAllDialects
  MLIRRegisterAllPasses
  MLIRTransforms
  # Core BishengIR dialect libraries
  BiShengIRAnnotationDialect
  BiShengIRHACCDialect
  BiShengIRHFusionDialect
  BiShengIRHIVMDialect
  BiShengIRMathExtDialect
  BiShengIRMemRefExtDialect
  BiShengIRScopeDialect
  BiShengIRSymbolDialect
  BiShengIRTensorDialect
)

mlir_check_all_link_libraries(triton-mlir-opt)
