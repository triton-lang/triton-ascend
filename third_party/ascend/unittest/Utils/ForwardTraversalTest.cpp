/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
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

#include "Utils/Utils.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/DenseMap.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
class ForwardTraversalTest : public testing::Test {
protected:
  void SetUp() override {
    context.loadDialect<arith::ArithDialect, func::FuncDialect>();
    module = parseSourceString<ModuleOp>(R"mlir(
      module {
        func.func @shared(%x: i32, %y: i32) -> i32 {
          %root = arith.addi %x, %y : i32
          %left = arith.muli %root, %x : i32
          %right = arith.subi %root, %y : i32
          %join = arith.addi %left, %right : i32
          %repeated = arith.xori %join, %join : i32
          %stop = arith.addi %repeated, %root : i32
          %tail = arith.muli %stop, %x : i32
          return %tail : i32
        }
      }
    )mlir",
                                         &context);
    ASSERT_TRUE(module);
    auto function = *module->getOps<func::FuncOp>().begin();
    for (Operation &op : function.getBody().front())
      nodes.push_back(&op);
    ASSERT_EQ(nodes.size(), 8u);
  }

  enum Node { Root, Left, Right, Join, Repeated, Stop, Tail, Return };
  MLIRContext context;
  OwningOpRef<ModuleOp> module;
  SmallVector<Operation *> nodes;
};

TEST_F(ForwardTraversalTest, VisitsSharedUsersAndRepeatedOperandsOnce) {
  llvm::DenseMap<Operation *, unsigned> visits;
  llvm::SmallPtrSet<Operation *, 16> stopOps;
  OpBuilder builder(&context);
  triton::traverseForwardUpdateUserChainIf(
      nodes[Root], [](Operation *) { return true; },
      [](Operation *) { return false; },
      [&](OpBuilder &, Operation *op) { ++visits[op]; }, builder, stopOps);

  // The diamond and the repeated operand must not repeat the join or its users.
  for (Operation *op : nodes)
    EXPECT_EQ(visits[op], 1u);
  EXPECT_TRUE(stopOps.empty());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST_F(ForwardTraversalTest, StopsBeforeActionAndDoesNotVisitUsers) {
  llvm::DenseMap<Operation *, unsigned> stopChecks;
  llvm::DenseMap<Operation *, unsigned> conditionChecks;
  llvm::SmallPtrSet<Operation *, 16> stopOps;
  triton::traverseForwardUpdateUserChainIf(
      nodes[Root],
      [&](Operation *op) {
        ++conditionChecks[op];
        return false;
      },
      [&](Operation *op) {
        ++stopChecks[op];
        return op == nodes[Stop];
      },
      nullptr, stopOps);

  for (unsigned i = Root; i <= Stop; ++i) {
    EXPECT_EQ(stopChecks[nodes[i]], 1u);
    EXPECT_EQ(conditionChecks[nodes[i]], i == Stop ? 0u : 1u);
  }
  for (unsigned i : {Tail, Return}) {
    EXPECT_EQ(stopChecks[nodes[i]], 0u);
    EXPECT_EQ(conditionChecks[nodes[i]], 0u);
  }
  ASSERT_EQ(stopOps.size(), 1u);
  EXPECT_TRUE(stopOps.contains(nodes[Stop]));
}

TEST_F(ForwardTraversalTest, StartsEachTraversalWithFreshVisitedState) {
  llvm::DenseMap<Operation *, unsigned> firstVisits;
  llvm::DenseMap<Operation *, unsigned> secondVisits;
  llvm::SmallPtrSet<Operation *, 16> stopOps;
  triton::traverseForwardUpdateUserChainIf(
      nodes[Root], [](Operation *) { return true; },
      [](Operation *) { return false; },
      [&](OpBuilder &, Operation *op) { ++firstVisits[op]; }, stopOps);
  triton::traverseForwardUpdateUserChainIf(
      nodes[Left], [](Operation *) { return true; },
      [](Operation *) { return false; },
      [&](OpBuilder &, Operation *op) { ++secondVisits[op]; }, stopOps);

  for (Operation *op : nodes) {
    EXPECT_EQ(firstVisits[op], 1u);
    EXPECT_EQ(secondVisits[op],
              op == nodes[Root] || op == nodes[Right] ? 0u : 1u);
  }
  EXPECT_TRUE(stopOps.empty());
}
} // namespace
