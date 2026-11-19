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

#include "ascend/include/TritonToLinalg/TLAToLinalgPass.h"

#include "ascend/include/TritonToLinalg/LoadStoreConverter.h"
#include "ascend/include/TritonToLinalg/TritonOpConverter.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/DialectConversion.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "triton/Dialect/Triton/IR/Types.h"

#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"

using namespace mlir;

namespace {

class TLATypeConverter final : public TypeConverter {
public:
  TLATypeConverter() {
    addConversion([](Type type) { return type; });
    addConversion([](triton::PointerType pointer) -> Type {
      Type elementType = pointer.getPointeeType();
      if (isa<triton::PointerType, ShapedType>(elementType))
        return {};
      // Match convertPointerType in the generic adapter: boolean pointers
      // use a byte-sized backing memref in the kernel ABI.
      if (elementType.isInteger(1))
        elementType = IntegerType::get(pointer.getContext(), 8);
      return MemRefType::get({ShapedType::kDynamic}, elementType);
    });
  }
};

class TLAKernelEntryConversion final
    : public OpConversionPattern<triton::FuncOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::FuncOp source, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    const TypeConverter *typeConverter = getTypeConverter();
    TypeConverter::SignatureConversion signature(source.getNumArguments());
    auto runtimeArg =
        MemRefType::get({ShapedType::kDynamic}, rewriter.getI8Type());
    signature.addInputs({runtimeArg, runtimeArg});
    if (failed(typeConverter->convertSignatureArgs(
            source.getFunctionType().getInputs(), signature)))
      return rewriter.notifyMatchFailure(
          source, "TLA kernel arguments require scalar pointees");
    SmallVector<Type> programTypes(6, rewriter.getI32Type());
    signature.addInputs(programTypes);

    auto target = rewriter.create<func::FuncOp>(
        source.getLoc(), source.getName(),
        rewriter.getFunctionType(signature.getConvertedTypes(), {}));
    SmallVector<DictionaryAttr> argAttrs;
    source.getAllArgAttrs(argAttrs);
    for (auto [i, attrs] : llvm::enumerate(argAttrs))
      target.setArgAttrs(i + 2, attrs);
    target->setAttr("global_kernel", rewriter.getStringAttr("local"));
    target->setAttr("mix_mode", rewriter.getStringAttr("aiv"));
    target->setAttr("parallel_mode", rewriter.getStringAttr("simd"));
    target->setAttr("SyncBlockLockArgIdx", rewriter.getI64IntegerAttr(0));
    target->setAttr("WorkspaceArgIdx", rewriter.getI64IntegerAttr(1));

    rewriter.inlineRegionBefore(source.getBody(), target.getBody(),
                                target.getBody().end());
    Block *body = rewriter.applySignatureConversion(&target.getBody().front(),
                                                    signature, typeConverter);
    for (BlockArgument arg : body->getArguments())
      if (arg.getArgNumber() < 2 ||
          arg.getArgNumber() >= body->getNumArguments() - 6)
        arg.setLoc(source.getLoc());
    rewriter.eraseOp(source);
    return success();
  }
};

// The frontend's block-pointer-to-buffer bridge carries no address arithmetic.
// Resolve it only after standard make_tensor_ptr lowering supplies a
// descriptor.
class TLABufferBridgeConversion final
    : public OpConversionPattern<UnrealizedConversionCastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(UnrealizedConversionCastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasAttr("tla.gm_view"))
      return failure();
    if (adaptor.getOperands().size() != 1 || op->getNumResults() != 1)
      return rewriter.notifyMatchFailure(
          op, "expected a one-to-one GM view bridge");
    Value source = adaptor.getOperands().front();
    auto sourceType = dyn_cast<MemRefType>(source.getType());
    auto resultType = dyn_cast<MemRefType>(op->getResult(0).getType());
    if (!sourceType || !resultType ||
        !memref::CastOp::areCastCompatible(sourceType, resultType))
      return rewriter.notifyMatchFailure(
          op,
          "block pointer did not lower to a compatible GM buffer descriptor");
    if (sourceType == resultType)
      rewriter.replaceOp(op, source);
    else
      rewriter.replaceOpWithNewOp<memref::CastOp>(op, resultType, source);
    return success();
  }
};

class TLABitcastConversion final
    : public OpConversionPattern<triton::BitcastOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::BitcastOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    Type srcType = op.getSrc().getType();
    Type dstType = op.getType();
    if (isa<triton::PointerType>(srcType) &&
        isa<triton::PointerType>(dstType)) {
      auto descriptor = dyn_cast<MemRefType>(adaptor.getSrc().getType());
      Type element = cast<triton::PointerType>(dstType).getPointeeType();
      if (element.isInteger(1))
        element = rewriter.getI8Type();
      // Forward only storage-preserving casts (including ABI i1 -> i8).
      // Reusing a descriptor with a different element size would change addptr.
      if (!descriptor || descriptor.getElementType() != element)
        return rewriter.notifyMatchFailure(
            op, "TLA pointer bitcast must preserve the backing element type");
      rewriter.replaceOp(op, adaptor.getSrc());
      return success();
    }
    if (!srcType.isIntOrFloat() || !dstType.isIntOrFloat())
      return rewriter.notifyMatchFailure(
          op, "TLA bitcast requires scalar numeric values or scalar pointers");
    rewriter.replaceOpWithNewOp<arith::BitcastOp>(op, dstType,
                                                  adaptor.getSrc());
    return success();
  }
};

// Program information and the entry terminator use the TLA kernel ABI.
class TLAProgramIdConversion final
    : public OpConversionPattern<triton::GetProgramIdOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetProgramIdOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned axis = static_cast<unsigned>(op.getAxis());
    auto entry = op->getParentOfType<func::FuncOp>();
    if (!entry || axis >= 3)
      return rewriter.notifyMatchFailure(op, "invalid TLA program ID axis");
    rewriter.replaceOp(op,
                       entry.getArgument(entry.getNumArguments() - 3 + axis));
    return success();
  }
};

class TLANumProgramsConversion final
    : public OpConversionPattern<triton::GetNumProgramsOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::GetNumProgramsOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    unsigned axis = static_cast<unsigned>(op.getAxis());
    auto entry = op->getParentOfType<func::FuncOp>();
    if (!entry || axis >= 3)
      return rewriter.notifyMatchFailure(op, "invalid TLA program count axis");
    rewriter.replaceOp(op,
                       entry.getArgument(entry.getNumArguments() - 6 + axis));
    return success();
  }
};

class TLAReturnConversion final : public OpConversionPattern<triton::ReturnOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::ReturnOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    rewriter.replaceOpWithNewOp<func::ReturnOp>(op, adaptor.getOperands());
    return success();
  }
};

class TLAOverflowAssertConversion final
    : public OpConversionPattern<triton::AssertOp> {
public:
  using OpConversionPattern::OpConversionPattern;

  LogicalResult
  matchAndRewrite(triton::AssertOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    if (!op->hasAttr("tt.auto_overflow_assert"))
      return rewriter.notifyMatchFailure(op, "user assertions are unsupported");
    // Match the generic TritonToLinalg handling of frontend-generated
    // overflow checks, while leaving user assertions unsupported.
    rewriter.eraseOp(op);
    return success();
  }
};

class TLAToLinalgPass final
    : public PassWrapper<TLAToLinalgPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(TLAToLinalgPass)

  StringRef getArgument() const final { return "tla-to-linalg"; }
  StringRef getDescription() const final {
    return "Adapt explicit TLA IR to the affinity memref kernel ABI";
  }

  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<arith::ArithDialect, func::FuncDialect, hivm::HIVMDialect,
                    memref::MemRefDialect, scf::SCFDialect>();
  }

  void runOnOperation() override {
    ModuleOp module = getOperation();
    SmallVector<triton::FuncOp> functions;
    module.walk([&](triton::FuncOp func) { functions.push_back(func); });
    if (functions.size() != 1 || functions.front().isDeclaration() ||
        functions.front().getSymVisibility() != "public") {
      module.emitError("TLA adapter expects one public tt.func definition");
      signalPassFailure();
      return;
    }
    triton::FuncOp source = functions.front();
    if (source.getFunctionType().getNumResults() != 0 ||
        !source.getBody().hasOneBlock()) {
      source.emitError("TLA adapter requires a single-block void kernel");
      signalPassFailure();
      return;
    }

    TLATypeConverter typeConverter;
    RewritePatternSet patterns(&getContext());
    patterns.add<TLAKernelEntryConversion>(typeConverter, &getContext());
    patterns.add<TLABufferBridgeConversion, TLABitcastConversion,
                 TLAProgramIdConversion, TLANumProgramsConversion,
                 TLAReturnConversion, TLAOverflowAssertConversion>(
        &getContext());
    // Reuse descriptor-based pointer lowering without changing the generic
    // adapter's pipeline or registering TLA-specific patterns in it.
    patterns.add<
        LoadStoreConverter::AddPtrConverter, LoadStoreConverter::LoadConverter,
        TTOpConverters::MakeTensorPtrConverter,
        TTOpConverters::PtrToIntConverter, TTOpConverters::ClampFConverter>(
        &getContext());

    ConversionTarget target(getContext());
    // Convert only the operations owned by this pass, not the entire Triton
    // dialect. Other pointer-free operations may be handled by later stages;
    // this does not introduce SIMT scope support or a generic Tile pipeline.
    target.addIllegalOp<triton::FuncOp, triton::ReturnOp, triton::AddPtrOp,
                        triton::PtrToIntOp, triton::GetProgramIdOp,
                        triton::GetNumProgramsOp, triton::AssertOp,
                        triton::LoadOp, triton::MakeTensorPtrOp,
                        triton::BitcastOp, triton::ClampFOp>();
    target.addDynamicallyLegalOp<UnrealizedConversionCastOp>(
        [&](Operation *op) {
          return !op->hasAttr("tla.gm_view") && typeConverter.isLegal(op);
        });
    // Unsupported pointer uses still need a conversion. No pointer source
    // materialization is provided, so entry pointers cannot leak through.
    target.markUnknownOpDynamicallyLegal(
        [&](Operation *op) { return typeConverter.isLegal(op); });
    if (failed(applyPartialConversion(module, target, std::move(patterns)))) {
      module.emitError(
          "unsupported TTIR or pointer use in TLA affinity module");
      signalPassFailure();
      return;
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>> mlir::triton::createTLAToLinalgPass() {
  return std::make_unique<TLAToLinalgPass>();
}
