// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
// SPDX-License-Identifier: MIT

#include "ascend/include/TritonToLinalg/ExecutionPolicy.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Parser/Parser.h"
#include "gtest/gtest.h"

using namespace mlir;
using namespace mlir::triton;

class ExecutionPolicyTest : public ::testing::Test {
protected:
  MLIRContext context;
  OwningOpRef<ModuleOp> module;

  void SetUp() override {
    context.loadDialect<arith::ArithDialect, func::FuncDialect,
                        linalg::LinalgDialect, math::MathDialect,
                        memref::MemRefDialect, scf::SCFDialect,
                        tensor::TensorDialect>();
  }

  Operation *parse(StringRef text, StringRef name) {
    module = parseSourceString<ModuleOp>(text, &context);
    if (!module)
      return nullptr;
    Operation *found = nullptr;
    module->walk([&](Operation *op) {
      if (op->getName().getStringRef() == name)
        found = op;
    });
    return found;
  }

  ExecutionKind classify(Operation *op) {
    return classifyExecution(op, ExecutionMode::SIMD);
  }
};

TEST_F(ExecutionPolicyTest, TensorMathUsesVectorContract) {
  auto *op = parse(R"mlir(
    func.func @test(%a: tensor<1xf32>) -> tensor<1xf32> {
      %r = math.exp2 %a : tensor<1xf32>
      return %r : tensor<1xf32>
    }
  )mlir",
                   "math.exp2");
  ASSERT_NE(op, nullptr);
  EXPECT_TRUE(canLookAhead(op));
  EXPECT_EQ(classify(op), ExecutionKind::Vector);
  EXPECT_EQ(classifyExecution(op, ExecutionMode::SIMT), ExecutionKind::Unknown);
  EXPECT_EQ(classifyExecution(op, ExecutionMode::Unknown),
            ExecutionKind::Unknown);
}

TEST_F(ExecutionPolicyTest, IntegerOperationIsScalarAndMovable) {
  auto *op = parse(R"mlir(
    func.func @test(%a: i32, %b: i32) -> i32 {
      %r = arith.xori %a, %b : i32
      return %r : i32
    }
  )mlir",
                   "arith.xori");
  ASSERT_NE(op, nullptr);
  EXPECT_TRUE(canLookAhead(op));
  EXPECT_EQ(classify(op), ExecutionKind::Scalar);
}

TEST_F(ExecutionPolicyTest, ExpAndLogUseKnownVectorLowering) {
  for (StringRef name : {"math.exp", "math.log"}) {
    std::string text = "func.func @test(%a: tensor<1xf32>) -> tensor<1xf32> { "
                       "%r = " +
                       name.str() +
                       " %a : tensor<1xf32> "
                       "return %r : tensor<1xf32> }";
    Operation *op = parse(text, name);
    ASSERT_NE(op, nullptr);
    EXPECT_TRUE(canLookAhead(op));
    EXPECT_EQ(classify(op), ExecutionKind::Vector);
  }
}

TEST_F(ExecutionPolicyTest, UnclassifiedScalarMathCanStillBeMovable) {
  auto *op = parse(R"mlir(
    func.func @test(%a: f32) -> f32 {
      %r = math.exp %a : f32
      return %r : f32
    }
  )mlir",
                   "math.exp");
  ASSERT_NE(op, nullptr);
  EXPECT_TRUE(canLookAhead(op));
  EXPECT_EQ(classify(op), ExecutionKind::Unknown);
}

TEST_F(ExecutionPolicyTest, ReadsAndUnsafeArithmeticCannotMove) {
  auto *load = parse(R"mlir(
    func.func @test(%a: memref<1xi32>, %i: index) -> i32 {
      %r = memref.load %a[%i] : memref<1xi32>
      return %r : i32
    }
  )mlir",
                     "memref.load");
  ASSERT_NE(load, nullptr);
  EXPECT_FALSE(canLookAhead(load));

  auto *div = parse(R"mlir(
    func.func @test(%a: i32, %b: i32) -> i32 {
      %r = arith.divsi %a, %b : i32
      return %r : i32
    }
  )mlir",
                    "arith.divsi");
  ASSERT_NE(div, nullptr);
  EXPECT_FALSE(canLookAhead(div));
}

TEST_F(ExecutionPolicyTest, LargeTensorCanBeAnIntermediate) {
  auto *op = parse(R"mlir(
    func.func @test(%a: tensor<8xf32>) -> tensor<8xf32> {
      %r = math.absf %a : tensor<8xf32>
      return %r : tensor<8xf32>
    }
  )mlir",
                   "math.absf");
  ASSERT_NE(op, nullptr);
  EXPECT_TRUE(canLookAhead(op));
  EXPECT_EQ(classify(op), ExecutionKind::Vector);
}

TEST_F(ExecutionPolicyTest, MatmulIsCubeButItsBodyIsNotScalarCode) {
  auto *op = parse(R"mlir(
    func.func @test(%a: tensor<16x16xf16>, %b: tensor<16x16xf16>,
                    %c: tensor<16x16xf32>) -> tensor<16x16xf32> {
      %r = linalg.matmul ins(%a, %b : tensor<16x16xf16>, tensor<16x16xf16>)
                         outs(%c : tensor<16x16xf32>) -> tensor<16x16xf32>
      return %r : tensor<16x16xf32>
    }
  )mlir",
                   "linalg.matmul");
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(classify(op), ExecutionKind::Cube);
  EXPECT_FALSE(canLookAhead(op));
}

TEST_F(ExecutionPolicyTest, ScalarTypedElementBodyDoesNotImplyScalarPipeline) {
  auto *op = parse(R"mlir(
    func.func @test(%a: tensor<4xi32>, %b: tensor<4xi32>,
                    %out: tensor<4xi32>) -> tensor<4xi32> {
      %r = linalg.generic {
        indexing_maps = [affine_map<(i)->(i)>, affine_map<(i)->(i)>, affine_map<(i)->(i)>],
        iterator_types = ["parallel"]
      } ins(%a, %b : tensor<4xi32>, tensor<4xi32>) outs(%out : tensor<4xi32>) {
      ^bb0(%x: i32, %y: i32, %z: i32):
        %sum = arith.addi %x, %y : i32
        linalg.yield %sum : i32
      } -> tensor<4xi32>
      return %r : tensor<4xi32>
    }
  )mlir",
                   "arith.addi");
  ASSERT_NE(op, nullptr);
  EXPECT_EQ(classify(op), ExecutionKind::Unknown);
}
