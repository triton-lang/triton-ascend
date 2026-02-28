/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 * Copyright (c) Microsoft Corporation.
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

#include "ascend/include/TritonToLinalg/TritonToLinalgPass.h"
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/TritonToLinalg/ArgMinMaxConverter.h"
#include "ascend/include/TritonToLinalg/DescriptorConverter.h"
#include "ascend/include/TritonToLinalg/FunctionConverter.h"
#include "ascend/include/TritonToLinalg/HoistBroadcast.h"
#include "ascend/include/TritonToLinalg/LoadStoreConverter.h"
#include "ascend/include/TritonToLinalg/TritonOpConverter.h"
#include "ascend/include/TritonToLinalg/UseAnalysis.h"
#include "ascend/include/TritonToLinalg/ImplicitPermute.h"
#include "ascend/include/TritonToLinalg/MarkTensorKindPass.h"
#include "ascend/include/TritonToStructured/CannonicalizerConverter.h"
#include "ascend/include/Utils/InterleaveOptimization.h"
#include "ascend/include/Utils/Utils.h"

#include "bishengir/Dialect/HFusion/IR/HFusion.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/Operation.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

#include "triton/Dialect/Triton/IR/Dialect.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Transforms/Transforms.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypeInterfaces.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "mlir/Transforms/Passes.h"

#include "llvm/ADT/BitVector.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/SmallVectorExtras.h"
#include "llvm/ADT/Twine.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/ErrorHandling.h"
#include "llvm/Support/LogicalResult.h"

#include <cassert>
#include <cstdint>
#include <optional>

#define DEBUG_TYPE "triton-to-linalg"

using namespace mlir;
using namespace triton;

int nd2nzFlag = 0;
bool compileOn91095Flag = false;
bool existDotFlag = false;

// Convert CustomOp after operand type converted,
// for example tt.ptr converted to memref.
class CustomOpConverter : public OpConversionPattern<hivm::CustomOp> {
public:
  using OpConversionPattern<hivm::CustomOp>::OpConversionPattern;

  LogicalResult
  matchAndRewrite(hivm::CustomOp op, OpAdaptor adaptor,
                  ConversionPatternRewriter &rewriter) const override {
    auto res_types = adaptor.getOutputs().getTypes();
    auto new_op = rewriter.create<hivm::CustomOp>(
        op->getLoc(), res_types, adaptor.getOperands(), op->getAttrs());
    rewriter.replaceOp(op, new_op);
    return success();
  }
};

static bool isSIMTOp(Operation *op) {
  if (auto custom_op = dyn_cast<hivm::CustomOp>(op)) {
    return custom_op.getCoreType() == hivm::TCoreType::VECTOR &&
           custom_op.getVFMode() == hivm::VFMode::SIMT;
  }
  return isa<
      triton::ascend::IndexPutOp,
      triton::ascend::GatherOutToUbOp,
      triton::ascend::ScatterUbToOutOp,
      triton::ascend::IndirectLoadOp,
      triton::ascend::IndirectStoreOp
      >(op);
}

TritonTypeConverter::TritonTypeConverter() {
  addConversion([](Type type) { return type; });

  addConversion([](triton::PointerType ptrType) {
    Type elem = ptrType.getPointeeType();
    // Handling special case: ptr<i1> -> memref<?x i8>
    if (auto it = dyn_cast<IntegerType>(elem); it && it.getWidth() == 1) {
      elem = IntegerType::get(ptrType.getContext(), 8);
      LLVM_DEBUG({
        llvm::dbgs() << "[TritonTypeConverter] Normalize i1 pointer to i8 "
                        "memref. elemType="
                     << elem << "\n";
      });
    }
    return MemRefType::get({ShapedType::kDynamic}, elem);
  });

  addConversion([](TensorType tensorType) -> Type {
    auto elemType = tensorType.getElementType();
    if (auto ptrType = dyn_cast<triton::PointerType>(elemType)) {
      elemType = ptrType.getPointeeType();
    }
    // Handling special case: tensor<i1> -> memref<?x i8>
    if (auto it = dyn_cast<IntegerType>(elemType); it && it.getWidth() == 1) {
      elemType = IntegerType::get(tensorType.getContext(), 8);
      LLVM_DEBUG({
        llvm::dbgs() << "[TritonTypeConverter] Normalize i1 tensor to i8 "
                        "memref. elemType="
                     << elemType << "\n";
      });
    }
    return MemRefType::get(tensorType.getShape(), elemType);
  });
}

void TritonToLinalgPass::addProgramInfo(triton::FuncOp func,
                                        bool globalKernel) {
  OpBuilder b(func);

  auto origFuncType = func.getFunctionType();
  auto origInputTypes = origFuncType.getInputs();
  SmallVector<Type> newInputTypes(origInputTypes);
  newInputTypes.append(TRITON_PROGRAM_INFO_ARG_COUNT, b.getI32Type());

  auto newFuncType =
      b.getFunctionType(newInputTypes, origFuncType.getResults());

  func.setFunctionType(newFuncType);

  // If argument attributes exist, extend attribute list.
  if (func.getAllArgAttrs()) {
    SmallVector<DictionaryAttr> newArgAttrs;
    func.getAllArgAttrs(newArgAttrs);
    newArgAttrs.append(TRITON_PROGRAM_INFO_ARG_COUNT, DictionaryAttr());
    func.setAllArgAttrs(newArgAttrs);
  }

  // Append the arguments to the entry block.
  for (unsigned i = 0; i < TRITON_PROGRAM_INFO_ARG_COUNT; i++) {
    func.getBody().front().addArgument(b.getI32Type(), func.getLoc());
  }

  if (globalKernel) {
    func->setAttr(globalKernelAttr, b.getStringAttr(""));
  } else {
    func->setAttr(globalKernelAttr, b.getStringAttr("local"));
  }
}

LogicalResult
TritonToLinalgPass::convertMultipleBlockControlFlow(Operation *funcOp,
                                                    OpBuilder &builder) {
  if (!isa<func::FuncOp>(funcOp)) {
    funcOp->emitError(
        "convertMultipleBlockControlFlow can only process func::FuncOp!");
    return failure();
  }

  SmallVector<Operation *> candidate;
  SmallVector<Block *> eraseBlocks;
  for (Block &block : dyn_cast<func::FuncOp>(funcOp).getBody()) {
    auto curTerminator = block.getTerminator();
    if (isa<cf::CondBranchOp>(curTerminator)) {
      candidate.push_back(curTerminator);
    } else if (isa<triton::ReturnOp>(curTerminator)) {
      if (candidate.empty()) {
        curTerminator->emitError(
            "funcOp has more than one Block but got an early 'tt.return' Op.");
        return failure();
      }
    } else if (!isa<cf::BranchOp>(curTerminator)) {
      funcOp->emitError(
          "funcOp has more than one Block but found unsupported Terminator: ")
          << *curTerminator;
      return failure();
    }

    if (!block.isEntryBlock())
      eraseBlocks.push_back(&block);
  }

  LLVM_DEBUG({
    llvm::dbgs() << "Found " << candidate.size()
                 << " candidate cond_branch operations to convert.\n";
  });

  if (candidate.empty()) {
    funcOp->emitError("funcOp has more than one Block but no candidate "
                      "Terminator was found!");
    return failure();
  }

  llvm::BitVector visitFlag(candidate.size(), false);

  // Recursive function to convert all cf::CondBranchOp to scf::IfOp
  std::function<void(Operation *, Operation *)> convertToSCF =
      [&](Operation *op, Operation *insertPosOp) -> void {
    auto condBranchOp = dyn_cast_if_present<cf::CondBranchOp>(op);
    auto iter = llvm::find(candidate, condBranchOp);
    if (!(condBranchOp && iter != candidate.end())) {
      op->emitError(
          "convertToSCF must process with condBranchOp in candidates!");
      return;
    }
    visitFlag.set(iter - candidate.begin());

    OpBuilder::InsertionGuard guard(builder);
    builder.setInsertionPointAfter(insertPosOp);

    // Well, here force to destory original control flow
    builder.create<scf::IfOp>(
        condBranchOp->getLoc(), condBranchOp.getCondition(),
        /*thenBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          SmallVector<Operation *> movedOps = llvm::map_to_vector(
              condBranchOp.getTrueDest()->without_terminator(),
              [](Operation &op) { return &op; });
          for (auto *innerOp : movedOps) {
            innerOp->moveBefore(builder.getInsertionBlock(),
                                builder.getInsertionPoint());
          }

          auto blockTerm = condBranchOp.getTrueDest()->getTerminator();
          if (auto nextCond = dyn_cast<cf::CondBranchOp>(blockTerm)) {
            if (movedOps.empty()) {
              blockTerm->emitError("movedOps can not be empty before entering "
                                   "convertToSCF (then)!");
              return;
            }
            convertToSCF(nextCond, movedOps.back());
          } else if (!isa<cf::BranchOp, triton::ReturnOp>(blockTerm)) {
            blockTerm->emitError(
                "Unsupported terminator in then branch after structuring");
          }

          builder.create<scf::YieldOp>(loc);
        },
        /*elseBuilder=*/
        [&](OpBuilder &builder, Location loc) {
          SmallVector<Operation *> movedOps = llvm::map_to_vector(
              condBranchOp.getFalseDest()->without_terminator(),
              [](Operation &op) { return &op; });
          for (auto *innerOp : movedOps) {
            innerOp->moveBefore(builder.getInsertionBlock(),
                                builder.getInsertionPoint());
          }

          auto blockTerm = condBranchOp.getFalseDest()->getTerminator();
          if (auto nextCond = dyn_cast<cf::CondBranchOp>(blockTerm)) {
            if (movedOps.empty()) {
              blockTerm->emitError("movedOps can not be empty before entering "
                                   "convertToSCF (else)!");
              return;
            }
            convertToSCF(nextCond, movedOps.back());
          } else if (!isa<cf::BranchOp, triton::ReturnOp>(blockTerm)) {
            blockTerm->emitError(
                "Unsupported terminator in else branch after structuring");
          }
          builder.create<scf::YieldOp>(loc);
        });
  };

  Block::iterator insertOp(candidate.front());
  if (insertOp == candidate.front()->getBlock()->begin()) {
    // if the first operation is a cond_branch, we need to insert before it
    convertToSCF(candidate.front(), candidate.front());
  } else {
    --insertOp;
    convertToSCF(candidate.front(), &(*insertOp));
  }

  if (!visitFlag.all()) {
    funcOp->emitError("Not all cf.cond_br converted!");
    return failure();
  }

  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(candidate.front());
  builder.create<triton::ReturnOp>(candidate.front()->getLoc());

  for (Operation *eachTerm : candidate)
    eachTerm->erase();
  for (Block *block : llvm::reverse(eraseBlocks))
    block->erase();

  return success();
}

void TritonToLinalgPass::convertTTFunc(triton::FuncOp func, const bool existDot,
                                       const bool existSIMTOp) {
  OpBuilder builder(func);

  auto name = func.getName();
  auto type = func.getFunctionType();

  SmallVector<DictionaryAttr> argAttrs, resAttrs;
  func.getAllArgAttrs(argAttrs);
  func.getAllResultAttrs(resAttrs);

  // Special handling for bit-casted tt.ptr arguments
  SmallVector<Type> inputTypes{type.getInputs()};
  SmallVector<Type> retTypes{type.getResults()};
  if (func.getSymVisibility() == "public" && !func.isDeclaration()) {
    for (size_t i = 0; i < func.getNumArguments(); ++i) {
      auto arg = func.getArgument(i);
      // Special method for i1 arg
      if (!isa<BaseMemRefType>(arg.getType()) ||
          dyn_cast<BaseMemRefType>(arg.getType()).getElementTypeBitWidth() !=
              1) {
        continue;
      }

      SmallVector<Operation *> argVaildUser{arg.getUsers()};
      llvm::erase_if(argVaildUser, [](Operation *op) -> bool {
        return isOpTriviallyDead(op);
      });

      if (!argVaildUser.empty()) {
        LLVM_DEBUG({
          auto &os = llvm::dbgs();
          os << arg << " has users:\n";
          int cnt = 0;
          for (auto it : argVaildUser) {
            os << "users[" << cnt++ << "] = " << *it;
          }
        });
        if (llvm::all_of(argVaildUser, [](Operation *userOp) {
              return isa<UnrealizedConversionCastOp>(userOp);
            })) {
          auto castOp = cast<UnrealizedConversionCastOp>(*argVaildUser.begin());
          if (castOp.getInputs().size() == 1 &&
              castOp.getOutputs().size() == 1) {
            arg.setType(castOp.getOutputs()[0].getType());
            inputTypes[i] = arg.getType();
          }
        } else {
          func->emitError(Twine("Unsupported use of func arg at index ") +
                          Twine(i));
        }
      } else {
        // Process unused bool ptr type specially, which guarantees bool pointer
        // argument's type is realistic and don't mislead backend compiler.
        // realistic memory layout of bool pointer is 8 bit width
        auto memType = dyn_cast<BaseMemRefType>(arg.getType())
                           .cloneWith(std::nullopt, builder.getI8Type());
        arg.setType(memType);
        inputTypes[i] = arg.getType();
      }
    }
  }
  auto castType = FunctionType::get(func.getContext(), inputTypes, retTypes);

  auto funcFunc = builder.create<func::FuncOp>(func.getLoc(), name, castType);
  funcFunc.setAllArgAttrs(argAttrs);
  funcFunc.setAllResultAttrs(resAttrs);
  auto kernelAttr = func->getAttr(globalKernelAttr);
  if (kernelAttr) {
    funcFunc->setAttr(globalKernelAttr, kernelAttr);
  }
  std::string kernelMixMode = "aiv";
  if (existDot) {
    // mix also works for pure cube kernel by using the same MAGIC_ELF keyword
    kernelMixMode = "mix";
  }
  // Set mix_mode in the func attrs so that the backend could know
  // the mix_mode by parse the func attrs.
  // The backend needs to know the mix_mode because the host wrapper
  // needs to set the devbin.magic. Check npu_utils.cpp.
  funcFunc->setAttr(kernelMixModeName, builder.getStringAttr(kernelMixMode));

  std::string parallelMode = "simd";
  if (existSIMTOp) {
    parallelMode = "mix_simd_simt";
  }
  funcFunc->setAttr(kernelParallelModeName,
                    builder.getStringAttr(parallelMode));

  auto autoBlockifyAttr = func->getAttr("auto_blockify_size");
  if (autoBlockifyAttr)
    funcFunc->setAttr("auto_blockify_size", autoBlockifyAttr);

  auto &funcFuncBody = funcFunc.getBody();
  auto &funcBody = func.getBody();

  IRMapping map;
  funcBody.cloneInto(&funcFuncBody, map);

  if (!funcFuncBody.hasOneBlock()) {
    if (failed(convertMultipleBlockControlFlow(funcFunc, builder))) {
      llvm_unreachable("Encounter unsupported control flow");
    }
  }

  for (Block &block : funcFuncBody.getBlocks()) {
    auto term = block.getTerminator();
    builder.setInsertionPoint(term);
    builder.create<func::ReturnOp>(func.getLoc(), term->getOperands());
    term->erase();
  }
  func.erase();
}

void TritonToLinalgPass::addDynamicLegal(
    ConversionTarget &target, TritonTypeConverter &tritonTypeConverter) {
  target.addLegalDialect<func::FuncDialect, arith::ArithDialect,
                         math::MathDialect, linalg::LinalgDialect,
                         affine::AffineDialect, scf::SCFDialect,
                         cf::ControlFlowDialect, tensor::TensorDialect,
                         LLVM::LLVMDialect, bufferization::BufferizationDialect,
                         memref::MemRefDialect, annotation::AnnotationDialect,
                         hivm::HIVMDialect, hfusion::HFusionDialect>();

  // add legal dialect on condition
  target.addLegalOp<ModuleOp>();

  // decide which ops need conversion based on uses
  target.addDynamicallyLegalOp<mlir::UnrealizedConversionCastOp>(
      [](mlir::Operation *op) {
        if (op->use_empty()) {
          return false;
        } else {
          return true;
        }
      });

  target.addDynamicallyLegalOp<triton::FuncOp>([&](triton::FuncOp op) {
    return tritonTypeConverter.isSignatureLegal(op.getFunctionType());
  });

  // For CustomOp, tt.ptr should be converted to memref.
  target.addDynamicallyLegalOp<hivm::CustomOp>([&](hivm::CustomOp op) {
    return all_of(op->getOperandTypes(), [](Type t) {
      if (isa<triton::PointerType>(t)) {
        return false;
      }
      if (auto shapedType = dyn_cast<ShapedType>(t)) {
        return !isa<triton::PointerType>(shapedType.getElementType());
      }
      return true;
    });
  });

  target.addDynamicallyLegalOp<arith::ConstantOp>([](arith::ConstantOp op) {
    auto res = op.getResult();
    if (!isa<RankedTensorType>(res.getType())) {
      return true;
    }

    if (auto denseAttr = dyn_cast<DenseElementsAttr>(op.getValue())) {
      if (!denseAttr.isSplat() ||
          !isa<FloatType, IntegerType>(denseAttr.getElementType())) {
        return true;
      }
      if (res.hasOneUse() && isa<tensor::ReshapeOp>(*res.user_begin())) {
        return true;
      }
      return false;
    }
    return true;
  });

  target.addDynamicallyLegalOp<scf::ForOp, scf::YieldOp>([](Operation *op) {
    return llvm::all_of(op->getOperandTypes(), [](Type t) {
      if (isa<triton::PointerType>(t)) {
        return false;
      }
      if (auto shapedType = dyn_cast<ShapedType>(t)) {
        return shapedType.getElementType().isIntOrFloat();
      }
      assert(t.isIntOrIndexOrFloat());
      return true;
    });
  });

  target.addDynamicallyLegalDialect<arith::ArithDialect, math::MathDialect>(
      [this](Operation *op) {
        if (op->hasAttr("MetaUse")) {
          return false;
        }

        if (isa<arith::ConstantOp>(op)) {
          return true;
        }

        bool operateOnTensors =
            llvm::all_of(op->getOperandTypes(),
                         [](Type type) { return isa<RankedTensorType>(type); });

        return this->namedOps || !operateOnTensors;
      });
}

void TritonToLinalgPass::populateTritonToLinalgCanonicalizationPatterns(RewritePatternSet &patterns)
{
    patterns.add<LoadStoreConverter::LoadStoreCanonicalizer<triton::LoadOp>,
                 LoadStoreConverter::LoadStoreCanonicalizer<triton::StoreOp>,
                 LoadStoreConverter::LoadStoreCanonicalizer<triton::AtomicRMWOp>,
                 LoadStoreConverter::LoadStoreCanonicalizer<triton::AtomicCASOp>>(patterns.getContext());
    patterns.add<TTOpConverters::BitcastCanonicalizer>(patterns.getContext());
    patterns.add<TTOpConverters::FpToFpCanonicalizer>(patterns.getContext());
    patterns.add<LoadStoreConverter::ScalarStoreCanonicalizer>(patterns.getContext());
    patterns.add<LoadStoreConverter::ScalarAtomicRMWCanonicalizer>(patterns.getContext());
    patterns.add<LoadStoreConverter::ScalarAtomicCASCanonicalizer>(patterns.getContext());
    patterns.add<LoadStoreConverter::AtomicMaxMinCanonicalizer>(patterns.getContext());
    patterns.add<
        TTOpConverters::ScalarMathCanonicalizer<math::AbsFOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AcosOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AcoshOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AsinOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AsinhOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AtanOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::Atan2Op>,
        // TTOpConverters::ScalarMathCanonicalizer<math::AtanhOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::CeilOp>, TTOpConverters::ScalarMathCanonicalizer<math::CosOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::CoshOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::ErfOp>, TTOpConverters::ScalarMathCanonicalizer<math::ExpOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::Exp2Op>,
        // TTOpConverters::ScalarMathCanonicalizer<math::ExpM1Op>,
        TTOpConverters::ScalarMathCanonicalizer<math::FloorOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::FmaOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::LogOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::Log10Op>,
        // TTOpConverters::ScalarMathCanonicalizer<math::Log1pOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::Log2Op>,
        // TTOpConverters::ScalarMathCanonicalizer<math::PowFOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::RoundOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::RsqrtOp>, TTOpConverters::ScalarMathCanonicalizer<math::SinOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::SinhOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::SqrtOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::TanOp>,
        TTOpConverters::ScalarMathCanonicalizer<math::TanhOp>,
        // TTOpConverters::ScalarMathCanonicalizer<math::TruncOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::AddFOp>, TTOpConverters::ScalarMathCanonicalizer<arith::SubFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::MulFOp>, // TTOpConverters::ScalarMathCanonicalizer<arith::DivFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::NegFOp>, TTOpConverters::ScalarMathCanonicalizer<arith::RemFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::MaxNumFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::MaximumFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::MinNumFOp>,
        TTOpConverters::ScalarMathCanonicalizer<arith::MinimumFOp>
        // By test, the following ops do not need canonicalization.
        // TTOpConverters::ScalarMathCanonicalizer<arith::CmpFOp>
        // TTOpConverters::ScalarMathCanonicalizer<arith::ExtFOp>
        // TTOpConverters::ScalarMathCanonicalizer<arith::TruncFOp>
        >(patterns.getContext());
    patterns.add<TTOpConverters::ReduceSingleCanonicalizer>(patterns.getContext());
    if (this->enableSelectAnalysis) {
      patterns.add<TTOpConverters::SelectCanonicalizer>(patterns.getContext());
    }
}

void TritonToLinalgPass::populateTritonToLinalgConversionPatterns(
    TypeConverter &typeConverter, RewritePatternSet &patterns,
    unsigned int launchGridRank) {
  nd2nzFlag = this->enableNd2nzOnVector;
  populateFunctionOpInterfaceTypeConversionPattern<triton::FuncOp>(
      patterns, typeConverter);

  patterns.add<triton::MetaUseEraser>(patterns.getContext());
  patterns.add<LoadStoreConverter::StoreConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AddPtrConverter>(patterns.getContext());
  patterns.add<FunctionConverter::GetProgramIDConverter>(patterns.getContext());
  patterns.add<FunctionConverter::GetNumProgramsConverter>(
      patterns.getContext());
  patterns.add<LoadStoreConverter::LoadConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AtomicRMWConverter>(patterns.getContext());
  patterns.add<LoadStoreConverter::AtomicCASConverter>(patterns.getContext());
  patterns.add<TTOpConverters::MakeRangeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SplatConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ClampFConverter>(patterns.getContext());
  patterns.add<TTOpConverters::PreciseDivConverter>(patterns.getContext());
  // reduce converters
  patterns.add<TTOpConverters::ArgMinConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ArgMaxConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ReduceConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ScanConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ReshapeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ExpandDimsConverter>(patterns.getContext());
  patterns.add<TTOpConverters::BroadcastConverter>(patterns.getContext());

  patterns.add<TTOpConverters::DenseConstantConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ExternElementwiseClOpConverter>(
      patterns.getContext());
  patterns.add<TTOpConverters::TritonMulhiuiConverter>(patterns.getContext());
  patterns.add<TTOpConverters::TritonPreciseSqrtConverter>(
      patterns.getContext());
  patterns.add<TTOpConverters::MakeTensorPtrConverter>(patterns.getContext());
  patterns.add<TTOpConverters::AdvanceConverter>(patterns.getContext());
  patterns.add<TTOpConverters::TransposeConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SplitConverter>(patterns.getContext());
  patterns.add<TTOpConverters::JoinConverter>(patterns.getContext());
  patterns.add<TTOpConverters::CatConverter>(patterns.getContext());
  patterns.add<TTOpConverters::BitcastConverter>(patterns.getContext());
  patterns.add<TTOpConverters::LoopConverter<scf::ForOp>>(
      patterns.getContext());
  patterns.add<TTOpConverters::LoopConverter<scf::WhileOp>>(
      patterns.getContext());
  patterns.add<TTOpConverters::YieldConverter>(patterns.getContext());

  patterns.add<TTOpConverters::DeviceAssertConverter>(patterns.getContext());
  patterns.add<TTOpConverters::DevicePrintConverter>(patterns.getContext());
  patterns.add<TTOpConverters::MatmulConverter>(patterns.getContext());
  patterns.add<TTOpConverters::DotScaledConverter>(patterns.getContext());
  patterns.add<TTOpConverters::PtrToIntConverter>(patterns.getContext());

  patterns.add<TTOpConverters::IndirectLoadConverter>(patterns.getContext());
  patterns.add<TTOpConverters::IndirectStoreConverter>(patterns.getContext());
  patterns.add<TTOpConverters::GatherOutToUbConverter>(patterns.getContext());
  patterns.add<TTOpConverters::ScatterUbToOutConverter>(patterns.getContext());
  patterns.add<TTOpConverters::IndexSelectSimdConverter>(patterns.getContext());
  patterns.add<TTOpConverters::IndexPutConverter>(patterns.getContext());
  patterns.add<TTOpConverters::SortOpConverter>(patterns.getContext());
  patterns.add<TTOpConverters::FlipOpConverter>(patterns.getContext());
  patterns.add<TTOpConverters::GatherConverter>(patterns.getContext());

  // Add convert pattern for CustomOp.
  patterns.add<CustomOpConverter>(patterns.getContext());

  if (!this->namedOps) {
    linalg::populateElementwiseToLinalgConversionPatterns(patterns);
  }
}

void TritonToLinalgPass::getDependentDialects(DialectRegistry &registry) const {
  registry.insert<func::FuncDialect, arith::ArithDialect, math::MathDialect,
                  linalg::LinalgDialect, affine::AffineDialect, scf::SCFDialect,
                  tensor::TensorDialect, bufferization::BufferizationDialect,
                  memref::MemRefDialect, hfusion::HFusionDialect,
                  hivm::HIVMDialect, annotation::AnnotationDialect>();
}

LogicalResult
TritonToLinalgPass::processDescriptorOperations(ModuleOp moduleOp) {
  // --- ConversionTarget: dynamic legality checks ---
  mlir::ConversionTarget target(getContext());

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
  target.addIllegalOp<triton::DescriptorLoadOp, triton::DescriptorStoreOp>();

  // --- Patterns ---
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<DescriptorConverter::DescriptorLoadConverter>(
      patterns.getContext());
  patterns.add<DescriptorConverter::DescriptorStoreConverter>(
      patterns.getContext());

  mlir::ConversionConfig config;
  config.buildMaterializations = true;
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns),
                                    config))) {
    moduleOp->emitError("failed to convert tensor descriptor operations");
    return failure();
  }

  return success();
}

LogicalResult
TritonToLinalgPass::processPtrBroadcastOperations(ModuleOp moduleOp) {
  // --- ConversionTarget: dynamic legality checks ---
  mlir::ConversionTarget target(getContext());
  target.addLegalOp<triton::SplatOp>();
  target.addLegalOp<triton::AddPtrOp>();
  target.addDynamicallyLegalOp<triton::BroadcastOp>([](triton::BroadcastOp op) {
    if (op->hasAttr("MetaUse")) {
      return true;
    }
    auto resultType = dyn_cast<RankedTensorType>(op.getType());
    HoistBroadcast::BroadcastHoister hoister(op);
    return !(isa<triton::PointerType>(resultType.getElementType()) &&
             hoister.canBroadcast());
  });

  // --- Patterns ---
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<HoistBroadcast::BroadcastConverter>(patterns.getContext());

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    moduleOp->emitError("failed to convert ptr broadcast operations");
    return failure();
  }

  return success();
}

LogicalResult TritonToLinalgPass::processImplicitPermuteOperations(ModuleOp moduleOp)
{
  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<ImplicitPermute::LoadConverter>(patterns.getContext());
  patterns.add<ImplicitPermute::StoreConverter>(patterns.getContext());
  patterns.add<ImplicitPermute::AtomicRMWConverter>(patterns.getContext());
  patterns.add<ImplicitPermute::AtomicCASConverter>(patterns.getContext());
  patterns.add<CannonicalizerConverter::SplatCmpConverter>(patterns.getContext());

  if (failed(applyPatternsAndFoldGreedily(moduleOp, std::move(patterns)))) {
    LLVM_DEBUG({
      llvm::dbgs() << "ImplicitPermute: rewrite MemOp failed\n";
    });
  }

  mlir::PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  return runPipeline(pm, getOperation());
}

LogicalResult TritonToLinalgPass::processLegalStrideOperations(ModuleOp moduleOp)
{
  mlir::ConversionTarget target(getContext());
  target.addLegalOp<arith::ConstantOp>();
  target.addDynamicallyLegalOp<memref::ReinterpretCastOp>(
      [](memref::ReinterpretCastOp op) {
        return !LoadStoreConverter::ReinterpretCastStrideCanonicalizer::hasFixableZeroStride(op);
      });

  mlir::RewritePatternSet patterns(&getContext());
  patterns.add<LoadStoreConverter::ReinterpretCastStrideCanonicalizer>(
      patterns.getContext());

  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    moduleOp->emitError(
        "failed to legalize reinterpret_cast dynamic stride(0) with size(1)");
    return failure();
  }

  return success();
}

void TritonToLinalgPass::runOnOperation() {
  compileOn91095Flag = this->compileOn91095;

  auto moduleOp = getOperation();

  // Check if the kernel contains tl.dot. Without tl.dot,
  // the kernel would be pure AIV kernel.
  bool existDot = false;
  moduleOp.walk([&](triton::DotOp dotOp) {
    existDot = true;
    return WalkResult::interrupt();
  });
  moduleOp.walk([&](triton::DotScaledOp dotScaledOp) {
    existDot = true;
    return WalkResult::interrupt();
  });
  existDotFlag = existDot;

  bool existSIMTOp = false;
  moduleOp.walk([&](Operation *op) {
    if (isSIMTOp(op)) {
      existSIMTOp = true;
      LLVM_DEBUG({
        auto &os = llvm::dbgs();
        os << "Found SIMT op in function: ";
        os << op->getName();
        os << "\n";
      });
      return WalkResult::interrupt();
    }
    return WalkResult::advance();
  });

  // Execute tensor descriptor operations conversion
  if (failed(processDescriptorOperations(moduleOp))) {
    signalPassFailure();
  }

  // Execute implicit permute
  if (failed(processImplicitPermuteOperations(moduleOp))) {
    LLVM_DEBUG({
      llvm::dbgs() << "Failed to process implicit permute operations\n";
    });
    signalPassFailure();
  }

  // 0. Annotate Memory-Related Triton FuncOps with tensor_kind (used by profiling).
  {
    PassManager pm(&getContext(), moduleOp.getOperationName());
    pm.addPass(triton::createMarkTensorKindPass());
 	     if (failed(runPipeline(pm, moduleOp))) {
 	       moduleOp->emitError("failed to run LoopCanonicalizerPass");
 	       signalPassFailure();
 	       return;
 	     }
  }

  RewritePatternSet canonicalizerPatterns(&getContext());
  // 1. Canonicalize load/store related patterns.
  this->populateTritonToLinalgCanonicalizationPatterns(canonicalizerPatterns);
  if (failed(applyPatternsAndFoldGreedily(moduleOp,
                                          std::move(canonicalizerPatterns)))) {
    moduleOp->emitError("failed to apply Canonicalizer Patterns");
    signalPassFailure();
  }

  // 2. Perform use analysis on FuncOp.
  moduleOp.walk([this](triton::FuncOp op) {
    if (failed(runUseAnalysis(op))) {
      signalPassFailure();
    }
  });

  RewritePatternSet patterns(&getContext());
  ConversionTarget target(getContext());
  TritonTypeConverter tritonTypeConverter{};

  // 3. Mark legal dialects and operations.
  this->addDynamicLegal(target, tritonTypeConverter);

  // 4. Mark ops that must be converted explicitly (e.g. tt.scan).
  auto loopOpLegalFn = [](LoopLikeOpInterface op) {
    return !op.getOperation()->hasAttr("UnhandledLoopOp");
  };

  target.addIllegalOp<triton::ScanOp>();
  target.addDynamicallyLegalOp<scf::ForOp>(loopOpLegalFn);
  target.addDynamicallyLegalOp<scf::WhileOp>(loopOpLegalFn);

  // 5. Register converters for all illegal Triton ops.
  // Execute ptr broadcast operations conversion
  if (failed(processPtrBroadcastOperations(moduleOp))) {
    signalPassFailure();
  }
  this->populateTritonToLinalgConversionPatterns(tritonTypeConverter, patterns,
                                                 LAUNCH_GRID_RANK);

  // 6. Inject program id / number of programs arguments into each Triton kernel
  // function.
  for (auto func : getOperation().getOps<triton::FuncOp>()) {
    addProgramInfo(func, globalKernel);
  }

  moduleOp.walk([this](LoopLikeOpInterface loopOp) {
    auto *op = loopOp.getOperation();
    if (!op->hasAttr("ExtractedLoadOrStore"))
      op->setAttr("UnhandledLoopOp", UnitAttr::get(op->getContext()));

    for (auto res : loopOp->getResults()) {
      if (auto tensorType = dyn_cast<RankedTensorType>(res.getType());
          tensorType &&
          !isa<triton::PointerType>(tensorType.getElementType())) {
        IRRewriter rewriter(op->getContext());
        rewriter.setInsertionPointAfter(op);
        auto newVal =
            rewriter.create<tensor::CastOp>(op->getLoc(), res.getType(), res);
        rewriter.replaceAllUsesExcept(res, newVal, newVal);
      }
    }
  });

  // 7. Convert ops.
  if (failed(applyPartialConversion(moduleOp, target, std::move(patterns)))) {
    moduleOp->emitError("failed to apply Conversion Patterns");
    signalPassFailure();
  }

  // Execute legal stride operations conversion
  if (failed(processLegalStrideOperations(moduleOp))) {
    signalPassFailure();
  }

  // 8. Convert function prologue/epilogue.
  moduleOp.walk([&](triton::FuncOp func) {
    this->convertTTFunc(func, existDot, existSIMTOp);
  });

  // 9. Clean up dead code and simplify IR.
  PassManager pm(&getContext(), moduleOp.getOperationName());
  pm.addPass(createCSEPass());
  pm.addPass(createCanonicalizerPass());
  if (failed(runPipeline(pm, getOperation()))) {
    signalPassFailure();
  }

  // Calculate size of PointerCastOp precisely
  SmallVector<hivm::PointerCastOp> castOps;

  moduleOp.walk([&](hivm::PointerCastOp op) { castOps.push_back(op); });

  for (auto op : castOps) {
    SmallVector<Operation *> userOps(op->getUsers().begin(),
                                     op->getUsers().end());
    IRRewriter rewriter(&getContext());
    rewriter.setInsertionPointAfter(op);
    Value addr = op.getAddrs()[0];
    auto elementType =
        cast<MemRefType>(op.getResult().getType()).getElementType();
    Value elementTypeSize;
    if (auto intType = dyn_cast<IntegerType>(elementType)) {
      elementTypeSize = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          rewriter.getIntegerAttr(addr.getType(), intType.getWidth() / 8));
    } else if (auto floatType = dyn_cast<FloatType>(elementType)) {
      elementTypeSize = rewriter.create<arith::ConstantOp>(
          op.getLoc(),
          rewriter.getIntegerAttr(addr.getType(), floatType.getWidth() / 8));
    } else {
      llvm_unreachable("Cannot get memory size");
    }

    for (auto userOp : userOps) {
      auto reinterpretCastOp = cast<memref::ReinterpretCastOp>(userOp);
      auto sizes = reinterpretCastOp.getStaticSizes();
      auto staticStrides = reinterpretCastOp.getStaticStrides();
      auto strides = reinterpretCastOp.getStrides();
      if (reinterpretCastOp.getStaticOffsets().size() != 1)
        userOp->emitError("IntToPtrOp must converted to PointerCastOp of "
                          "memref<?xdtype> type");
      int64_t castOpSize = 0;
      SmallVector<int64_t> dynamicSizes;
      for (const auto &[size, stride] : llvm::zip_equal(sizes, staticStrides)) {
        assert(!ShapedType::isDynamic(size));
        if (ShapedType::isDynamic(stride))
          dynamicSizes.push_back(size);
        else
          castOpSize = size * stride;
      }
      rewriter.setInsertionPoint(reinterpretCastOp);
      Value dynamicSize = rewriter.create<arith::ConstantOp>(
          op.getLoc(), rewriter.getIndexAttr(castOpSize));
      for (const auto &[size, stride] :
           llvm::zip_equal(dynamicSizes, strides)) {
        Value axisSize = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIndexAttr(size));
        axisSize =
            rewriter.create<arith::MulIOp>(op.getLoc(), stride, axisSize);
        dynamicSize =
            rewriter.create<arith::AddIOp>(op.getLoc(), dynamicSize, axisSize);
      }
      Value offsetValue;
      auto staticOffset = reinterpretCastOp.getStaticOffsets()[0];
      if (ShapedType::isDynamic(staticOffset)) {
        offsetValue = reinterpretCastOp.getOffsets()[0];
        if (offsetValue.getType() != addr.getType())
          offsetValue = rewriter.create<arith::IndexCastOp>(
              op.getLoc(), addr.getType(), offsetValue);
      } else {
        offsetValue = rewriter.create<arith::ConstantOp>(
            op.getLoc(), rewriter.getIntegerAttr(addr.getType(), staticOffset));
      }
      offsetValue = rewriter.create<arith::MulIOp>(op.getLoc(), offsetValue,
                                                   elementTypeSize);
      Value realAddr =
          rewriter.create<arith::AddIOp>(op.getLoc(), addr, offsetValue);
      auto memrefType = MemRefType::get({ShapedType::kDynamic}, elementType);
      auto newCastOp = rewriter.create<hivm::PointerCastOp>(
          op.getLoc(), memrefType, realAddr, dynamicSize);
      auto markOp = rewriter.create<annotation::MarkOp>(op.getLoc(),
                                                        newCastOp.getResult());
      markOp->setAttr(hivm::AddressSpaceAttr::getMnemonic(),
                      {hivm::AddressSpaceAttr::get(rewriter.getContext(),
                                                   hivm::AddressSpace::GM)});
      rewriter.replaceOpWithNewOp<memref::ReinterpretCastOp>(
          reinterpretCastOp,
          cast<MemRefType>(reinterpretCastOp.getResult().getType()), newCastOp,
          ValueRange({}), reinterpretCastOp.getSizes(),
          reinterpretCastOp.getStrides(), SmallVector<int64_t>({0}),
          reinterpretCastOp.getStaticSizes(),
          reinterpretCastOp.getStaticStrides());
    }
    rewriter.eraseOp(op);
  }

  // Try interleave optimization
  llvm::DenseMap<BlockArgument, SmallVector<Operation *>> interleaveCandidate;
  llvm::DenseMap<BlockArgument, SmallVector<Operation *>>
      interleaveCandidateWithMask;
  moduleOp.walk([&](bufferization::MaterializeInDestinationOp materializeOp) {
    if (auto reinterpretCastOp =
            materializeOp.getDest()
                .getDefiningOp<memref::ReinterpretCastOp>()) {
      if (llvm::isa<BlockArgument>(reinterpretCastOp.getSource()) &&
          reinterpretCastOp.getStaticStrides().back() == 2) {
        interleaveCandidate[llvm::cast<BlockArgument>(
                                reinterpretCastOp.getSource())]
            .push_back(materializeOp);
      }
    }

    // Difference is that converted op chain of store with mask has
    // `memref::SubViewOp`
    if (auto subviewOp =
            materializeOp.getDest().getDefiningOp<memref::SubViewOp>()) {
      if (!llvm::isa<tensor::ExtractSliceOp>(
              materializeOp.getSource().getDefiningOp()))
        return WalkResult::advance();

      if (auto reinterpretCastOp =
              subviewOp.getSource()
                  .getDefiningOp<memref::ReinterpretCastOp>()) {
        if (llvm::isa<BlockArgument>(reinterpretCastOp.getSource()) &&
            reinterpretCastOp.getStaticStrides().back() == 2) {
          interleaveCandidateWithMask[llvm::cast<BlockArgument>(
                                          reinterpretCastOp.getSource())]
              .push_back(materializeOp);
        }
      }
    }

    return WalkResult::advance();
  });

  for (auto [blockArg, materializeVec] : interleaveCandidate) {
    // Just enable optimization where exists double materializeOp with same
    // block argument destination.
    if (materializeVec.size() != 2)
      continue;
    auto result = InterleaveStatusOptimization(materializeVec);
  }

  for (auto [blockArg, materializeVec] : interleaveCandidateWithMask) {
    if (materializeVec.size() != 2)
      continue;
    auto result = InterleaveStatusWithMaskOptimization(materializeVec);
  }

  // Force to add an argument at the beginning of function arguments, which
  // represents stub arg for workspace. Default type is memref<?xi8>
  for (auto func : getOperation().getOps<func::FuncOp>()) {
    if (!func->hasAttr("global_kernel"))
      continue;

    auto context = func.getContext();
    constexpr int64_t syncBlockLockArgIdx = 0;
    NamedAttribute syncBlockLockArgAttr(
        StringAttr::get(context, "syncBlockLock"), UnitAttr::get(context));
    MemRefType syncBlockLockArgType =
        MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                        IntegerType::get(context, 8));
    func.insertArgument(syncBlockLockArgIdx,      // argIndex
                        syncBlockLockArgType,     // argType
                        nullptr, func->getLoc()); // dicAttr
    func->setAttr("SyncBlockLockArgIdx",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                   0)); // 64: 64位整型

    constexpr int64_t workspaceArgIdx = 1;
    MemRefType workspaceArgType =
        MemRefType::get(SmallVector<int64_t>(1, ShapedType::kDynamic),
                        IntegerType::get(context, 8));
    NamedAttribute workspaceArgAttr(StringAttr::get(context, "workspace"),
                                    UnitAttr::get(context));

    func.insertArgument(/*argIndex*/ workspaceArgIdx,
                        /*argType*/ workspaceArgType,
                        /*dicAttr*/ nullptr, func->getLoc());
    func->setAttr("WorkspaceArgIdx",
                  IntegerAttr::get(IntegerType::get(&getContext(), 64),
                                   1)); // 64: 64位整型
  }

  // Fix the Location info
  moduleOp.walk([&](Operation *op) {
    auto loc = op->getLoc();
    if (isa<UnknownLoc>(loc)) {
      llvm::SmallPtrSet<Operation *, 16> stopOps;
      traverseForwardUpdateUserChainIf(
          op,
          /*conditionFn*/
          [](Operation *curOp) { return false; },
          /*stopFn*/
          [](Operation *curOp) { return !isa<UnknownLoc>(curOp->getLoc()); },
          /*actionFn*/
          nullptr, stopOps);
      if (stopOps.empty()) {
        op->emitWarning() << *op << " and its users all have no location!";
      } else {
        Operation *goodOp = *stopOps.begin();
        op->setLoc(goodOp->getLoc());
      }
    }
    return WalkResult::advance();
  });
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToLinalgPass(
    bool globalKernel, bool namedOps, bool enableNd2nzOnVector,
    bool enableSelectAnalysis, bool compileOn91095) {
  return std::make_unique<TritonToLinalgPass>(
      globalKernel, namedOps, enableNd2nzOnVector, enableSelectAnalysis,
      compileOn91095);
}

std::unique_ptr<OperationPass<ModuleOp>> triton::createTritonToLinalgPass() {
  return std::make_unique<TritonToLinalgPass>();
}
