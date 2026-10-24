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

#include "PointerAnalysisTestUtils.h"
#include "TritonToUnstructure/UnstructureConversionPass.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Verifier.h"
#include "mlir/Parser/Parser.h"
#include "llvm/ADT/ScopeExit.h"
#include "gtest/gtest.h"
#include <cstddef>

using namespace mlir;
using namespace mlir::triton;

namespace {
struct ReleaseProbe {
  Value watched;
  llvm::DenseMap<Value, PtrOffsetInfo> *cache;
  unsigned releases = 0;
  bool staleKey = false;
  bool stalePayload = false;
};
thread_local ReleaseProbe *activeProbe = nullptr;

// Observe deallocation only in this test executable. Do not allocate, print,
// or dereference the watched Value here: its destructor has already run.
void observeRelease(void *pointer) {
  auto *probe = activeProbe;
  if (!probe || pointer != probe->watched.getAsOpaquePointer())
    return;
  ++probe->releases;
  probe->staleKey |= probe->cache->contains(probe->watched);
  for (auto &entry : *probe->cache) {
    auto &info = entry.second;
    probe->stalePayload |=
        info.getPtr() == probe->watched || info.getOffset() == probe->watched;
    for (Value offset : info.getOffsetsRef())
      probe->stalePayload |= offset == probe->watched;
  }
}
} // namespace

// ELF linker wrapping redirects delete calls from the linked MLIR objects;
// the actual allocator and production implementation remain unchanged.
extern "C" void __real__ZdlPv(void *) noexcept;
extern "C" void __real__ZdlPvm(void *, std::size_t) noexcept;
extern "C" void __wrap__ZdlPv(void *pointer) noexcept {
  observeRelease(pointer);
  __real__ZdlPv(pointer);
}
extern "C" void __wrap__ZdlPvm(void *pointer, std::size_t size) noexcept {
  observeRelease(pointer);
  __real__ZdlPvm(pointer, size);
}

TEST(OffsetCacheLifetime, ReleasesOldValuesAfterCacheInvalidation) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, scf::SCFDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @cache_lifetime(%base: !tt.ptr<i32>, %n: i32) {
        %zero = arith.constant 0 : i32
        %one = arith.constant 1 : i32
        %p0 = tt.addptr %base, %n : !tt.ptr<i32>, i32
        %result = scf.for %i = %zero to %n step %one
            iter_args(%p = %p0) -> (!tt.ptr<i32>) : i32 {
          %next = tt.addptr %p, %one : !tt.ptr<i32>, i32
          scf.yield %next : !tt.ptr<i32>
        }
        tt.return
      }
    }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto loop = *func.getBody().front().getOps<scf::ForOp>().begin();
  Value oldArg = loop.getRegionIterArgs().front();
  llvm::DenseMap<Value, PtrOffsetInfo> cache;
  IRRewriter rewriter(&context);
  {
    OffsetAnalysisContext analysis(rewriter, cache, *module);
    parse(oldArg, oldArg.getLoc(), analysis);
  }
  ASSERT_TRUE(cache.contains(oldArg));

  ReleaseProbe probe{oldArg, &cache};
  ASSERT_EQ(activeProbe, nullptr);
  {
    activeProbe = &probe;
    auto resetProbe = llvm::make_scope_exit([] { activeProbe = nullptr; });
    replacePtrArguments(func, cache);
  }

  // Do not silently pass if the linker or MLIR allocation path stops routing
  // the watched deallocation through the wrappers.
  EXPECT_EQ(probe.releases, 1u);
  EXPECT_FALSE(probe.staleKey);
  EXPECT_FALSE(probe.stalePayload);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OffsetAnalysisContext, ResetInvalidatesImportedFactsAfterOperandChange) {
  MLIRContext mlirContext;
  mlirContext.loadDialect<arith::ArithDialect, triton::TritonDialect,
                          tensor::TensorDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @reset(%base: !tt.ptr<i32>, %dynamic: tensor<8xi32>) {
        %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
        %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
        %a = tt.addptr %p, %range : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
        tt.return
      }
    }
  )mlir",
                                            &mlirContext);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto add = *func.getBody().front().getOps<triton::AddPtrOp>().begin();
  llvm::DenseMap<Value, PtrOffsetInfo> cache;
  IRRewriter rewriter(&mlirContext);
  OffsetAnalysisContext analysis(rewriter, cache, *module);
  parse(add.getResult(), add.getLoc(), analysis);
  ASSERT_TRUE(cache.at(add.getResult()).isStructured());
  EXPECT_FALSE(cache.at(add.getResult()).isScalarLike());

  analysis.resetPointerAnalysis();
  add.getOffsetMutable().assign(func.getArgument(1));
  EXPECT_FALSE(cache.contains(add.getResult()));
  parse(add.getResult(), add.getLoc(), analysis);
  EXPECT_TRUE(cache.at(add.getResult()).isUnstructured());
  EXPECT_FALSE(cache.at(add.getResult()).isScalarLike());
  EXPECT_EQ(cache.at(add.getResult()).getPtr(), func.getArgument(0));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OffsetAnalysisContext, PublicArithmeticAndLocalNarrowDispatch) {
  MLIRContext mlirContext;
  mlirContext.loadDialect<arith::ArithDialect, triton::TritonDialect,
                          tensor::TensorDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @dispatch(%base: !tt.ptr<i32>, %x: tensor<8xi64>, %y: tensor<8xi64>, %start: i32) {
        %three = arith.constant dense<3> : tensor<8xi64>
        %sum = arith.addi %x, %y : tensor<8xi64>
        %scaled = arith.muli %sum, %three : tensor<8xi64>
        %delta = arith.subi %scaled, %x : tensor<8xi64>
        %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
        %public = tt.addptr %p, %delta : tensor<8x!tt.ptr<i32>>, tensor<8xi64>
        %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
        %s = tt.splat %start : i32 -> tensor<8xi32>
        %narrow = arith.addi %range, %s : tensor<8xi32>
        %local = tt.addptr %p, %narrow : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
        tt.return
      }
    }
  )mlir",
                                            &mlirContext);
  ASSERT_TRUE(module);
  auto func = *module->getOps<triton::FuncOp>().begin();
  auto adds = func.getBody().front().getOps<triton::AddPtrOp>();
  auto first = *adds.begin();
  auto second = *std::next(adds.begin());
  llvm::DenseMap<Value, PtrOffsetInfo> cache;
  IRRewriter rewriter(&mlirContext);
  OffsetAnalysisContext analysis(rewriter, cache, *module);
  ASSERT_TRUE(analysis.parseCommonPointer(first.getResult()));
  const auto &info = cache.at(first.getResult());
  EXPECT_EQ(info.getPtr(), func.getArgument(0));
  EXPECT_EQ(info.getOffset().getType(), first.getOffset().getType());
  EXPECT_TRUE(info.isUnstructured());
  EXPECT_FALSE(info.isScalarLike());
  // The local source-domain classification is not a public address proof.
  EXPECT_FALSE(analysis.parseCommonPointer(second.getResult()));
  parse(second.getResult(), second.getLoc(), analysis);
  EXPECT_TRUE(cache.at(second.getResult()).isStructured());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OffsetAnalysisContext, AffineProofDoesNotAuthorizeUnsupportedHandoff) {
  MLIRContext mlirContext;
  mlirContext.loadDialect<arith::ArithDialect, triton::TritonDialect,
                          tensor::TensorDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  tt.func @affine_integer_select(%base: !tt.ptr<i32>, %cond: i1, %start: i64) -> tensor<8xi32> {
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %r64 = arith.extsi %range : tensor<8xi32> to tensor<8xi64>
    %s = tt.splat %start : i64 -> tensor<8xi64>
    %shifted = arith.addi %r64, %s : tensor<8xi64>
    %offset = arith.select %cond, %r64, %shifted : tensor<8xi64>
    %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
    %ptr = tt.addptr %p, %offset : tensor<8x!tt.ptr<i32>>, tensor<8xi64>
    %v = tt.load %ptr : tensor<8x!tt.ptr<i32>>
    tt.return %v : tensor<8xi32>
  }
}
)mlir",
                                            &mlirContext);
  ASSERT_TRUE(module);
  triton::AddPtrOp ptrOp;
  module->walk([&](triton::AddPtrOp op) { ptrOp = op; });
  IRRewriter rewriter(&mlirContext);
  pointer::AnalysisOptions options;
  options.addressBitWidth = 64;
  options.indexBitWidth = 64;
  pointer::PointerAnalysis publicAnalysis(rewriter, options);
  auto proven = publicAnalysis.analyzePointer(ptrOp.getResult());
  ASSERT_TRUE(succeeded(proven));
  ASSERT_EQ(proven->offsets.axes.size(), 1u);
  EXPECT_EQ(proven->offsets.axes[0], pointer::AxisKind::Structured);
  llvm::DenseMap<Value, PtrOffsetInfo> cache;
  OffsetAnalysisContext analysis(rewriter, cache, *module);
  ASSERT_TRUE(analysis.parseCommonPointer(ptrOp.getResult()));
  EXPECT_TRUE(cache.at(ptrOp.getResult()).isUnstructured());
  EXPECT_FALSE(cache.at(ptrOp.getResult()).isScalarLike());
  EXPECT_EQ(cache.at(ptrOp.getResult()).getPtr(), proven->base);
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OffsetAnalysisContext, LocalArithmeticUsesPreparedSourceWithoutReset) {
  MLIRContext mlirContext;
  mlirContext.loadDialect<arith::ArithDialect, triton::TritonDialect,
                          tensor::TensorDialect, scf::SCFDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
module {
  tt.func @reset_during_local_integer_parse(%base: !tt.ptr<i32>, %offsets: tensor<2x4xi64>, %lb: index, %ub: index, %step: index) -> tensor<8xi32> {
    %zero = arith.constant dense<0> : tensor<8xi32>
    %range = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
    %p = tt.splat %base : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
    %ptrs = tt.addptr %p, %offsets : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi64>
    %flat = tt.reshape %ptrs allow_reorder : tensor<2x4x!tt.ptr<i32>> -> tensor<8x!tt.ptr<i32>>
    %loaded = tt.load %flat : tensor<8x!tt.ptr<i32>>
    %result = scf.for %iv = %lb to %ub step %step iter_args(%state = %zero) -> (tensor<8xi32>) {
      %sum = arith.addi %range, %loaded : tensor<8xi32>
      %next = arith.addi %sum, %state : tensor<8xi32>
      scf.yield %next : tensor<8xi32>
    }
    tt.return %result : tensor<8xi32>
  }
}
)mlir",
                                            &mlirContext);
  ASSERT_TRUE(module);
  arith::AddIOp sum;
  module->walk([&](arith::AddIOp op) {
    if (!sum)
      sum = op;
  });
  ASSERT_TRUE(sum);
  llvm::DenseMap<Value, PtrOffsetInfo> cache;
  IRRewriter rewriter(&mlirContext);
  normalizePointerAnalysisInputs(*module, rewriter);
  OffsetAnalysisContext analysis(rewriter, cache, *module);
  parse(sum.getResult(), sum.getLoc(), analysis);
  EXPECT_EQ(analysis.getGeneration(), 0u);
  ASSERT_TRUE(cache.contains(sum.getLhs()));
  ASSERT_TRUE(cache.contains(sum.getRhs()));
  EXPECT_TRUE(cache.at(sum.getLhs()).isStructured());
  EXPECT_TRUE(cache.at(sum.getRhs()).isUnstructured());
  EXPECT_TRUE(cache.at(sum.getResult()).isUnstructured());
  EXPECT_FALSE(cache.at(sum.getResult()).isScalarLike());
  auto load = sum.getRhs().getDefiningOp<triton::LoadOp>();
  ASSERT_TRUE(load);
  ASSERT_TRUE(cache.contains(load.getPtr()));
  EXPECT_EQ(cache.at(load.getPtr()).getOffset().getType(),
            RankedTensorType::get({8}, rewriter.getI64Type()));
  auto reshape = load.getPtr().getDefiningOp<triton::ReshapeOp>();
  ASSERT_TRUE(reshape);
  EXPECT_FALSE(reshape->hasAttr("allow_reorder"));
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OffsetAnalysisContext, CustomSourceMappingUsesPreparedSourceWithoutReset) {
  MLIRContext mlirContext;
  mlirContext.loadDialect<arith::ArithDialect, triton::TritonDialect,
                          tensor::TensorDialect, hivm::HIVMDialect,
                          annotation::AnnotationDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module {
      tt.func @custom_reset(%base: !tt.ptr<i32>, %offsets: tensor<2x4xi64>) {
        %r = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
        %p = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>
        %source = tt.addptr %p, %r : tensor<8x!tt.ptr<i32>>, tensor<8xi32>
        %p2 = tt.splat %base : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
        %pointers = tt.addptr %p2, %offsets : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi64>
        %flat = tt.reshape %pointers allow_reorder : tensor<2x4x!tt.ptr<i32>> -> tensor<8x!tt.ptr<i32>>
        %loaded = tt.load %flat : tensor<8x!tt.ptr<i32>>
        %empty = tensor.empty() : tensor<8x!tt.ptr<i32>>
        %result = hivm.hir.custom
          {hivm.is_distributed, hivm.pipe = #hivm.pipe<PIPE_V>, hivm.tcore_type = #hivm.tcore_type<VECTOR>, symbol = "test_reset", SrcPtrIndex = array<i32: 0>}
          "test_reset" ins(%source, %loaded : tensor<8x!tt.ptr<i32>>, tensor<8xi32>)
          outs(%empty : tensor<8x!tt.ptr<i32>>) -> tensor<8x!tt.ptr<i32>>
        annotation.mark %result {ContinuousMemAccess} : tensor<8x!tt.ptr<i32>>
        tt.return
      }
    }
  )mlir",
                                            &mlirContext);
  ASSERT_TRUE(module);
  hivm::CustomOp custom;
  module->walk([&](hivm::CustomOp op) { custom = op; });
  ASSERT_TRUE(custom);
  IRRewriter rewriter(&mlirContext);
  llvm::DenseMap<Value, PtrOffsetInfo> cache;
  normalizePointerAnalysisInputs(*module, rewriter);
  OffsetAnalysisContext analysis(rewriter, cache, *module);
  parse(custom->getResult(0), custom.getLoc(), analysis);
  EXPECT_EQ(analysis.getGeneration(), 0u);
  ASSERT_TRUE(cache.contains(custom->getOperand(0)));
  const auto &result = cache.at(custom->getResult(0));
  EXPECT_EQ(result.getRank(), 1);
  EXPECT_TRUE(result.isStructured());
  EXPECT_FALSE(result.isScalarLike());
  EXPECT_EQ(result.getPtr(),
            custom->getParentOfType<triton::FuncOp>().getArgument(0));
  EXPECT_EQ(result.getOffset(), cache.at(custom->getOperand(0)).getOffset());
  EXPECT_TRUE(succeeded(verify(*module)));
}

TEST(OffsetAnalysisContext, NewBoundariesKeepLongPointerChainsLinear) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  // Distinct opaque inputs force boundary discovery at every addptr. Exercise
  // both one root and successive roots in the same stable consumer context.
  for (unsigned length : {8u, 32u}) {
    for (bool incremental : {false, true}) {
      std::string ir = "module { tt.func @chain(%base: !tt.ptr<i32>";
      for (unsigned i = 0; i < length; ++i)
        ir += ", %x" + std::to_string(i) + ": tensor<8xi64>";
      ir +=
          ") { %p0 = tt.splat %base : !tt.ptr<i32> -> tensor<8x!tt.ptr<i32>>\n";
      for (unsigned i = 0; i < length; ++i)
        ir += "%p" + std::to_string(i + 1) + " = tt.addptr %p" +
              std::to_string(i) + ", %x" + std::to_string(i) +
              " : tensor<8x!tt.ptr<i32>>, tensor<8xi64>\n";
      ir += "tt.return } }";
      auto module = parseSourceString<ModuleOp>(ir, &context);
      ASSERT_TRUE(module);
      auto f = *module->getOps<triton::FuncOp>().begin();
      SmallVector<triton::AddPtrOp> pointers;
      SmallVector<std::pair<Operation *, SmallVector<Value>>> original;
      module->walk([&](Operation *op) {
        original.emplace_back(op, llvm::to_vector(op->getOperands()));
        if (auto p = dyn_cast<triton::AddPtrOp>(op))
          pointers.push_back(p);
      });
      IRRewriter rewriter(&context);
      llvm::DenseMap<Value, PtrOffsetInfo> cache;
      OffsetAnalysisContext analysis(rewriter, cache, *module);
      if (incremental) {
        for (auto p : pointers)
          parse(p.getResult(), p.getLoc(), analysis);
      } else {
        parse(pointers.back(), f.getLoc(), analysis);
      }
      unsigned additions = 0;
      module->walk([&](arith::AddIOp op) { ++additions; });
      // The offset tensor needs one sum per pointer link, independent of query
      // order. Quadratic repeated prefixes must not be hidden by later DCE.
      EXPECT_LE(additions, length);
      auto &last = cache.at(pointers.back().getResult());
      EXPECT_EQ(last.getPtr(), f.getArgument(0));
      pointer::test::Environment inputs;
      for (unsigned i = 0; i < length; ++i)
        for (unsigned lane = 0; lane < 8; ++lane)
          inputs[f.getArgument(i + 1)].emplace_back(64, i + lane);
      auto values = pointer::test::evaluate(last.getOffset(), inputs);
      ASSERT_TRUE(values);
      ASSERT_EQ(values->size(), 8u);
      for (unsigned lane = 0; lane < 8; ++lane)
        EXPECT_EQ((*values)[lane].getSExtValue(),
                  length * (length - 1) / 2 + length * lane);
      for (const auto &[op, operands] : original)
        EXPECT_EQ(llvm::to_vector(op->getOperands()), operands);
      EXPECT_EQ(analysis.getGeneration(), 0u);
      EXPECT_TRUE(succeeded(verify(*module)));
    }
  }
}

TEST(OffsetAnalysisContext, SourceNormalizationIsAnExplicitPhase) {
  MLIRContext context;
  context.loadDialect<arith::ArithDialect, tensor::TensorDialect,
                      triton::TritonDialect>();
  auto module = parseSourceString<ModuleOp>(R"mlir(
    module { tt.func @prepare(%base: !tt.ptr<i32>, %x: tensor<2x4xi64>) {
      %p = tt.splat %base : !tt.ptr<i32> -> tensor<2x4x!tt.ptr<i32>>
      %ptr = tt.addptr %p, %x : tensor<2x4x!tt.ptr<i32>>, tensor<2x4xi64>
      %flat = tt.reshape %ptr allow_reorder : tensor<2x4x!tt.ptr<i32>> -> tensor<8x!tt.ptr<i32>>
      tt.return
    } }
  )mlir",
                                            &context);
  ASSERT_TRUE(module);
  auto f = *module->getOps<triton::FuncOp>().begin();
  auto reshape = *f.getBody().front().getOps<triton::ReshapeOp>().begin();
  IRRewriter rewriter(&context);
  {
    llvm::DenseMap<Value, PtrOffsetInfo> facts;
    OffsetAnalysisContext analysis(rewriter, facts, *module);
    parse(reshape.getResult(), reshape.getLoc(), analysis);
    EXPECT_TRUE(reshape.getAllowReorder());
    EXPECT_EQ(facts.at(reshape.getResult()).getPtr(), reshape.getResult());
    EXPECT_EQ(analysis.getGeneration(), 0u);
  }
  normalizePointerAnalysisInputs(*module, rewriter);
  EXPECT_FALSE(reshape.getAllowReorder());
  auto attributes = reshape->getAttrDictionary();
  auto operands = llvm::to_vector(reshape->getOperands());
  {
    llvm::DenseMap<Value, PtrOffsetInfo> facts;
    OffsetAnalysisContext analysis(rewriter, facts, *module);
    parse(reshape.getResult(), reshape.getLoc(), analysis);
    EXPECT_EQ(facts.at(reshape.getResult()).getPtr(), f.getArgument(0));
    EXPECT_EQ(reshape->getAttrDictionary(), attributes);
    EXPECT_EQ(llvm::to_vector(reshape->getOperands()), operands);
    EXPECT_EQ(analysis.getGeneration(), 0u);
  }
  unsigned before = 0, after = 0;
  module->walk([&](Operation *) { ++before; });
  normalizePointerAnalysisInputs(*module, rewriter);
  module->walk([&](Operation *) { ++after; });
  EXPECT_EQ(before, after);
  EXPECT_TRUE(succeeded(verify(*module)));
}
