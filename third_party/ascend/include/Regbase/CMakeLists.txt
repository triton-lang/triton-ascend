# Generate the existing NPU-IR ops with Triton's MLIR version. The wrapper
# only supplies the renamed type predicate; all op definitions come from NPU-IR.
set(LLVM_TARGET_DEFINITIONS HIVMRegbaseIntrins.td)
set(regbase_tablegen_flags)
if(LLVM_VERSION_MAJOR GREATER_EQUAL 22)
  list(APPEND regbase_tablegen_flags -D__LLVM_MAJOR_VERSION_22_COMPATIBLE__)
endif()
mlir_tablegen(HIVMRegbaseIntrins.h.inc -gen-op-decls ${regbase_tablegen_flags})
mlir_tablegen(HIVMRegbaseIntrins.cpp.inc -gen-op-defs ${regbase_tablegen_flags})
mlir_tablegen(HIVMRegbaseIntrinsDialect.h.inc -gen-dialect-decls
  -dialect=hivm_regbaseintrins ${regbase_tablegen_flags})
mlir_tablegen(HIVMRegbaseIntrinsDialect.cpp.inc -gen-dialect-defs
  -dialect=hivm_regbaseintrins ${regbase_tablegen_flags})
mlir_tablegen(HIVMRegbaseAttrs.h.inc -gen-attrdef-decls
  -attrdefs-dialect=hivm_regbaseintrins ${regbase_tablegen_flags})
mlir_tablegen(HIVMRegbaseAttrs.cpp.inc -gen-attrdef-defs
  -attrdefs-dialect=hivm_regbaseintrins ${regbase_tablegen_flags})
add_public_tablegen_target(MLIRHIVMRegbaseIntrinOpsIncGen)
add_dependencies(mlir-headers MLIRHIVMRegbaseIntrinOpsIncGen)
