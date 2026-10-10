#ifndef ASCEND_COSTMODEL_UT_STAGEIRTESTUTILS_H
#define ASCEND_COSTMODEL_UT_STAGEIRTESTUTILS_H

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::ascend::test {

// Generic tt.* fixtures must not register Triton: they deliberately use
// placeholder operands instead of verified Triton operations.
class IRTestContext : public MLIRContext {
public:
  explicit IRTestContext(bool triton = false, bool unregistered = false) {
    getOrLoadDialect<arith::ArithDialect>();
    getOrLoadDialect<func::FuncDialect>();
    getOrLoadDialect<scf::SCFDialect>();
    if (triton)
      getOrLoadDialect<mlir::triton::TritonDialect>();
    allowUnregisteredDialects(unregistered);
  }

  OwningOpRef<ModuleOp> parse(llvm::StringRef source) {
    return parseSourceString<ModuleOp>(source, this);
  }
};

} // namespace mlir::ascend::test
#endif // ASCEND_COSTMODEL_UT_STAGEIRTESTUTILS_H
