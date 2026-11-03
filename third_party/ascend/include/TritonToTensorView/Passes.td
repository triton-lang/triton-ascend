include "mlir/Pass/PassBase.td"

def TritonToTensorView : Pass<"triton-to-tensor-view", "mlir::ModuleOp"> {
  let summary = "Legalize Triton pointer operands of TensorView accesses";
  let constructor = "triton::createTritonToTensorViewPass()";
  let dependentDialects = ["tv::TensorViewDialect", "triton::TritonDialect",
                           "arith::ArithDialect"];
}
