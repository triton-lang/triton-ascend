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

#include "TritonToLinalg/TLAToLinalgPass.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "gtest/gtest.h"

using namespace mlir;

namespace {
class LegacyPointerConverters : public ::testing::Test {
protected:
  MLIRContext context;

  LegacyPointerConverters() {
    context.loadDialect<arith::ArithDialect, func::FuncDialect,
                        memref::MemRefDialect, scf::SCFDialect,
                        triton::TritonDialect, hivm::HIVMDialect>();
  }

  LogicalResult lower(ModuleOp module) {
    PassManager manager(&context);
    // Exercise the unchanged caller of the context-only legacy constructors.
    manager.addPass(triton::createTLAToLinalgPass());
    manager.addPass(createCanonicalizerPass());
    return manager.run(module);
  }
};
} // namespace

TEST_F(LegacyPointerConverters, ScalarLoadAndAddressKeepComposedOffset) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func public @scalar(%base: !tt.ptr<f32>, %out: memref<1xf32>,
                             %addresses: memref<1xi64>, %mask: i1) {
        %seven = arith.constant 7 : i32
        %minus_three = arith.constant -3 : i32
        %zero = arith.constant 0 : index
        %other = arith.constant 0.0 : f32
        %first = tt.addptr %base, %seven : !tt.ptr<f32>, i32
        %p = tt.addptr %first, %minus_three : !tt.ptr<f32>, i32
        %value = tt.load %p, %mask, %other : !tt.ptr<f32>
        %address = tt.ptr_to_int %p : !tt.ptr<f32> -> i64
        memref.store %value, %out[%zero] : memref<1xf32>
        memref.store %address, %addresses[%zero] : memref<1xi64>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(lower(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));

  unsigned loads = 0;
  module->walk([&](memref::LoadOp load) {
    ++loads;
    EXPECT_TRUE(load->getParentOfType<scf::IfOp>());
    auto view = load.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
    ASSERT_TRUE(view);
    // Offsets are in elements: 7 + (-3) = 4.
    EXPECT_EQ(getConstantIntValue(view.getMixedOffsets().front()), 4);
    EXPECT_EQ(view.getSource(),
              load->getParentOfType<func::FuncOp>().getArgument(2));
  });
  EXPECT_EQ(loads, 1u);

  unsigned addresses = 0;
  module->walk([&](memref::StoreOp store) {
    if (!store.getValue().getType().isInteger(64))
      return;
    ++addresses;
    auto cast = store.getValue().getDefiningOp<arith::IndexCastOp>();
    ASSERT_TRUE(cast);
    auto add = cast.getIn().getDefiningOp<arith::AddIOp>();
    ASSERT_TRUE(add);
    // ptr_to_int adds the same offset once, in bytes: 4 * sizeof(f32).
    EXPECT_EQ(getConstantIntValue(add.getRhs()), 16);
    EXPECT_TRUE(
        add.getLhs().getDefiningOp<memref::ExtractAlignedPointerAsIndexOp>());
  });
  EXPECT_EQ(addresses, 1u);
  module->walk([](Operation *op) {
    EXPECT_FALSE(
        (isa<triton::AddPtrOp, triton::LoadOp, triton::PtrToIntOp>(op)));
  });
}

TEST_F(LegacyPointerConverters, GMViewPreservesExplicitUnitAxisStride) {
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func public @view(%base: !tt.ptr<f32>, %out: memref<1xf32>) {
        %one = arith.constant 1 : i64
        %width = arith.constant 32 : i64
        %zero_i32 = arith.constant 0 : i32
        %zero = arith.constant 0 : index
        %seven = arith.constant 7 : i32
        %minus_four = arith.constant -4 : i32
        %first = tt.addptr %base, %seven : !tt.ptr<f32>, i32
        %p = tt.addptr %first, %minus_four : !tt.ptr<f32>, i32
        %block = tt.make_tensor_ptr %p, [%one, %width], [%width, %one],
            [%zero_i32, %zero_i32] {order = array<i32: 1, 0>} : <tensor<1x8xf32>>
        %view = "builtin.unrealized_conversion_cast"(%block) {tla.gm_view}
            : (!tt.ptr<tensor<1x8xf32>>) -> memref<1x8xf32, strided<[32, 1], offset: ?>>
        %value = memref.load %view[%zero, %zero]
            : memref<1x8xf32, strided<[32, 1], offset: ?>>
        memref.store %value, %out[%zero] : memref<1xf32>
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  ASSERT_TRUE(succeeded(lower(*module)));
  ASSERT_TRUE(succeeded(verify(*module)));
  unsigned loads = 0;
  module->walk([&](memref::LoadOp load) {
    ++loads;
    auto [strides, offset] = load.getMemRefType().getStridesAndOffset();
    // The unit axis retains its declared stride of 32, despite the block
    // containing only eight elements. TLA's GM bridge requires that layout.
    EXPECT_EQ(strides, (SmallVector<int64_t>{32, 1}));
    Value source = load.getMemRef();
    if (auto cast = source.getDefiningOp<memref::CastOp>())
      source = cast.getSource();
    auto view = source.getDefiningOp<memref::ReinterpretCastOp>();
    ASSERT_TRUE(view);
    EXPECT_EQ(getConstantIntValue(view.getMixedOffsets().front()), 3);
    EXPECT_EQ(view.getSource(),
              load->getParentOfType<func::FuncOp>().getArgument(2));
  });
  EXPECT_EQ(loads, 1u);
  module->walk([](Operation *op) {
    EXPECT_FALSE((isa<triton::AddPtrOp, triton::MakeTensorPtrOp>(op)));
    EXPECT_FALSE(op->hasAttr("tla.gm_view"));
  });
}
