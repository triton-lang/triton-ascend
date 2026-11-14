// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchRewriteWorkspace.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "gtest/gtest.h"

namespace mlir::triton::cfg {
namespace {
TEST(BatchRewriteWorkspaceTest, MissingAndExpiredBindings) {
  MLIRContext context;
  context.allowUnregisteredDialects();
  OpBuilder builder(&context);
  auto loc = builder.getUnknownLoc();
  OwningOpRef<ModuleOp> module = ModuleOp::create(loc);
  Block *block = module->getBody();
  Value source = block->addArgument(builder.getI32Type(), loc);
  Value replacement = block->addArgument(builder.getI32Type(), loc);
  builder.setInsertionPointToEnd(block);
  OperationState use(loc, "test.use");
  use.addOperands(source);
  Operation *op = builder.create(use);

  BatchRewriteWorkspace workspace;
  EXPECT_FALSE(workspace.loopID(op));
  EXPECT_FALSE(workspace.aggregation(source));
  EXPECT_FALSE(workspace.cloneOperands(op));
  workspace.recordLoop(op, 0);
  workspace.recordAccess(op);
  EXPECT_EQ(workspace.loopID(op), 0u);
  EXPECT_TRUE(workspace.validSourceMapping(1));
  workspace.recordAccess(op);
  EXPECT_FALSE(workspace.validSourceMapping(2));

  workspace.recordVariation(source, true);
  workspace.bindGenerated(source, replacement, false);
  workspace.mergeRows(source, llvm::SmallBitVector(1), true);
  workspace.beginVariationScope();
  EXPECT_FALSE(workspace.known(source));
  EXPECT_EQ(workspace.lookupGenerated(source), replacement);
  auto domain = workspace.aggregation(source);
  ASSERT_TRUE(domain);
  EXPECT_TRUE(domain->first.none());
  EXPECT_TRUE(domain->second);
  auto operands = workspace.cloneOperands(op);
  ASSERT_TRUE(operands);
  EXPECT_EQ(operands->lookupOrNull(source), replacement);
  EXPECT_FALSE(workspace.cloneOperands(op, {replacement}));

  workspace.invalidateGenerated();
  EXPECT_FALSE(workspace.lookupGenerated(source));
  EXPECT_FALSE(workspace.cloneOperands(op));
  EXPECT_TRUE(workspace.aggregation(source));
  EXPECT_EQ(workspace.loopID(op), 0u);

  workspace.bindGenerated(source, replacement, false);
  workspace.invalidateSource();
  EXPECT_FALSE(workspace.loopID(op));
  EXPECT_FALSE(workspace.aggregation(source));
  EXPECT_FALSE(workspace.lookupGenerated(source));
  EXPECT_FALSE(workspace.cloneOperands(op));
}
} // namespace
} // namespace mlir::triton::cfg
