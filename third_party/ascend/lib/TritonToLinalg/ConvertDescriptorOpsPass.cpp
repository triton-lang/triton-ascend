/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "ascend/include/TritonToLinalg/ConvertDescriptorOpsPass.h"
#include "ascend/include/TritonToLinalg/DescriptorConverter.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Transforms/DialectConversion.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

using namespace mlir;
using namespace triton;

void ConvertDescriptorOpsPass::getDependentDialects(
    DialectRegistry &registry) const {
  // Matches the dialects that DescriptorConverter patterns actually read
  // from and create ops in (arith/scf/tensor/triton).
  registry.insert<arith::ArithDialect, scf::SCFDialect, tensor::TensorDialect,
                  triton::TritonDialect>();
}

void ConvertDescriptorOpsPass::runOnOperation() {
  ModuleOp moduleOp = getOperation();

  // --- ConversionTarget: dynamic legality checks ---
  mlir::ConversionTarget target(getContext());
  target.addLegalDialect<mlir::tensor::TensorDialect>();

  // Dialect-level dynamic legality: ops are legal if none of their
  // operands/results use TensorDescType.
  target.addDynamicallyLegalDialect<
      mlir::arith::ArithDialect, mlir::scf::SCFDialect, triton::TritonDialect>(
      [](mlir::Operation *op) {
        return !DescriptorConverter::hasATensorDescriptorType(
                   op->getOperandTypes()) &&
               !DescriptorConverter::hasATensorDescriptorType(
                   op->getResultTypes());
      });
  // Function signature legality: Triton FuncOp is legal if its inputs/outputs
  // contain no TensorDescType.
  target.addDynamicallyLegalOp<triton::FuncOp>([](triton::FuncOp funcOp) {
    return !DescriptorConverter::hasATensorDescriptorType(
               funcOp.getFunctionType().getInputs()) &&
           !DescriptorConverter::hasATensorDescriptorType(
               funcOp.getFunctionType().getResults());
  });
  target.addLegalOp<triton::MakeTensorDescOp>();
  target.addIllegalOp<triton::DescriptorLoadOp, triton::DescriptorStoreOp,
                      triton::DescriptorScatterOp, triton::DescriptorGatherOp,
                      triton::DescriptorReduceOp>();

  // --- Patterns ---
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<DescriptorConverter::DescriptorLoadConverter>(
      patterns.getContext());
  patterns.add<DescriptorConverter::DescriptorStoreConverter>(
      patterns.getContext());
  patterns.add<DescriptorConverter::DescriptorScatterConverter>(
      patterns.getContext());
  patterns.add<DescriptorConverter::DescriptorGatherConverter>(
      patterns.getContext());
  patterns.add<DescriptorConverter::DescriptorReduceConverter>(
      patterns.getContext());

  mlir::ConversionConfig config;
  config.buildMaterializations = true;
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns),
                                    config))) {
    moduleOp->emitError("failed to convert tensor descriptor operations");
    signalPassFailure();
  }
}

std::unique_ptr<OperationPass<ModuleOp>>
triton::createConvertDescriptorOpsPass() {
  return std::make_unique<ConvertDescriptorOpsPass>();
}
