/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
 * SPDX-License-Identifier: MIT
 */

#include "TritonToLinalg/ImplicitPermute.h"
#include "TritonToLinalg/TritonToLinalgPass.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ScopeExit.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
std::string printModule(ModuleOp module) {
  std::string result;
  llvm::raw_string_ostream stream(result);
  module.print(stream);
  return result;
}

template <typename Op, typename Pattern>
void checkUnchanged(ModuleOp module, PatternRewriter &rewriter) {
  SmallVector<Op> ops;
  module.walk([&](Op op) { ops.push_back(op); });
  ASSERT_FALSE(ops.empty());
  Pattern pattern(module.getContext());
  for (Op op : ops) {
    auto before = printModule(module);
    rewriter.setInsertionPoint(op);
    EXPECT_TRUE(failed(pattern.matchAndRewrite(op, rewriter)));
    EXPECT_EQ(printModule(module), before);
    EXPECT_TRUE(succeeded(verify(module)));
  }
}
} // namespace

TEST(ImplicitPermute, UnchangedMemoryOpsLeaveNoAnalysisIR) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @unchanged(%base: !tt.ptr<i32>, %offset: i32) {
        %range = tt.make_range {start = 0 : i32, end = 16 : i32} : tensor<16xi32>
        %shift = tt.splat %offset : i32 -> tensor<16xi32>
        %indices = arith.addi %range, %shift : tensor<16xi32>
        %ptrs = tt.splat %base : !tt.ptr<i32> -> tensor<16x!tt.ptr<i32>>
        %ptr = tt.addptr %ptrs, %indices : tensor<16x!tt.ptr<i32>>, tensor<16xi32>
        %value = tt.load %ptr : tensor<16x!tt.ptr<i32>>
        tt.store %ptr, %value : tensor<16x!tt.ptr<i32>>
        %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %value : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>) -> tensor<16xi32>
        %cas = tt.atomic_cas relaxed, gpu, %ptr, %value, %old : (tensor<16x!tt.ptr<i32>>, tensor<16xi32>, tensor<16xi32>) -> tensor<16xi32>
        %scalar = tt.load %base : !tt.ptr<i32>
        tt.store %base, %scalar : !tt.ptr<i32>
        %s_old = tt.atomic_rmw add, relaxed, gpu, %base, %scalar : (!tt.ptr<i32>, i32) -> i32
        %s_cas = tt.atomic_cas relaxed, gpu, %base, %scalar, %s_old : (!tt.ptr<i32>, i32, i32) -> i32
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  bool savedTarget = compileOn91095Flag, savedDot = existDotFlag;
  auto restore = llvm::make_scope_exit([&] {
    compileOn91095Flag = savedTarget;
    existDotFlag = savedDot;
  });
  PatternRewriter rewriter(&context);
  for (bool isA5 : {false, true}) {
    compileOn91095Flag = isA5;
    existDotFlag = false;
    checkUnchanged<triton::LoadOp, ImplicitPermute::LoadConverter>(*module,
                                                                   rewriter);
    checkUnchanged<triton::StoreOp, ImplicitPermute::StoreConverter>(*module,
                                                                     rewriter);
    checkUnchanged<triton::AtomicRMWOp, ImplicitPermute::AtomicRMWConverter>(
        *module, rewriter);
    checkUnchanged<triton::AtomicCASOp, ImplicitPermute::AtomicCASConverter>(
        *module, rewriter);
  }
}

TEST(ImplicitPermute, UnsupportedMaskRollsBackPermutedPointer) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @unsupported_mask(%base: !tt.ptr<i32>, %mask: tensor<4x8xi1>, %offset: i32) {
        %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
        %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
        %stride = arith.constant dense<4> : tensor<8xi32>
        %scaled = arith.muli %c, %stride : tensor<8xi32>
        %rs = tt.expand_dims %r {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
        %cs = tt.expand_dims %scaled {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
        %rb = tt.broadcast %rs : tensor<4x1xi32> -> tensor<4x8xi32>
        %cb = tt.broadcast %cs : tensor<1x8xi32> -> tensor<4x8xi32>
        %indices = arith.addi %rb, %cb : tensor<4x8xi32>
        %shift = tt.splat %offset : i32 -> tensor<4x8xi32>
        %shifted = arith.addi %indices, %shift : tensor<4x8xi32>
        %prefix = arith.cmpi slt, %rb, %shift : tensor<4x8xi32>
        %combined = arith.andi %prefix, %mask : tensor<4x8xi1>
        %ptrs = tt.splat %base : !tt.ptr<i32> -> tensor<4x8x!tt.ptr<i32>>
        %ptr = tt.addptr %ptrs, %shifted : tensor<4x8x!tt.ptr<i32>>, tensor<4x8xi32>
        %zero = arith.constant dense<0> : tensor<4x8xi32>
        %value = tt.load %ptr, %combined, %zero : tensor<4x8x!tt.ptr<i32>>
        tt.store %ptr, %value, %combined : tensor<4x8x!tt.ptr<i32>>
        %old = tt.atomic_rmw add, relaxed, gpu, %ptr, %value, %combined : (tensor<4x8x!tt.ptr<i32>>, tensor<4x8xi32>, tensor<4x8xi1>) -> tensor<4x8xi32>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(verify(*module)));
  bool savedTarget = compileOn91095Flag;
  auto restore =
      llvm::make_scope_exit([&] { compileOn91095Flag = savedTarget; });
  compileOn91095Flag = false;
  PatternRewriter rewriter(&context);
  checkUnchanged<triton::LoadOp, ImplicitPermute::LoadConverter>(*module,
                                                                 rewriter);
  checkUnchanged<triton::StoreOp, ImplicitPermute::StoreConverter>(*module,
                                                                   rewriter);
  checkUnchanged<triton::AtomicRMWOp, ImplicitPermute::AtomicRMWConverter>(
      *module, rewriter);
}
