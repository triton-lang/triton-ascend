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

#include "DynamicCVPipeline/Common/SSBufferManager.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "DynamicCVPipeline/ComputeBlockOpt/CubePageLoaders.h"
#include "DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Analysis/AliasAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/StaticValueUtils.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/MapVector.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>

using namespace mlir;

namespace {

bool isZero(Value value) { return matchPattern(value, m_PosZeroFloat()); }

bool isGMSource(Value value) {
  while (true) {
    auto space = hivm::getOptionalHIVMAddressSpace(value.getType());
    if (space && *space != hivm::AddressSpace::GM)
      return false;
    auto view = value.getDefiningOp<ViewLikeOpInterface>();
    if (!view)
      break;
    value = view.getViewSource();
  }
  if (auto arg = dyn_cast<BlockArgument>(value))
    return isa<func::FuncOp>(arg.getOwner()->getParentOp());
  auto space = hivm::getOptionalHIVMAddressSpace(value.getType());
  return space && *space == hivm::AddressSpace::GM;
}

std::optional<CVPipeline::CubePageLoader>
getPageLoader(tensor::InsertSliceOp insert) {
  auto tensor = insert.getSource().getDefiningOp<bufferization::ToTensorOp>();
  if (!tensor || !tensor->hasOneUse() ||
      utils::getAnnotateOpWithAttr(tensor.getResult(),
                                   hivm::kMayImplicitTransposeWithLastAxis))
    return std::nullopt;
  auto alloc = tensor.getBuffer().getDefiningOp<memref::AllocOp>();
  if (!alloc ||
      alloc.getType().getShape() != insert.getSourceType().getShape() ||
      alloc->getBlock() != insert->getBlock() ||
      tensor->getBlock() != insert->getBlock())
    return std::nullopt;

  CVPipeline::CubePageLoader page{insert, tensor, {}, {}};
  SmallVector<Value> worklist{alloc.getResult()};
  llvm::SetVector<Operation *> bufferOps;
  bufferOps.insert(alloc);
  bool padded = false;
  while (!worklist.empty()) {
    Value buffer = worklist.pop_back_val();
    for (Operation *user : buffer.getUsers()) {
      if (user == tensor)
        continue;
      if (auto view = dyn_cast<memref::SubViewOp>(user)) {
        // Only the leading rectangular mask used by a contiguous page load.
        if (view.getSource() != alloc ||
            llvm::any_of(view.getStaticOffsets(),
                         [](int64_t x) { return x != 0; }) ||
            llvm::any_of(view.getStaticStrides(),
                         [](int64_t x) { return x != 1; }) ||
            view.getStaticSizes()[1] <= 0 ||
            view.getStaticSizes()[1] > alloc.getType().getDimSize(1))
          return std::nullopt;
        if (bufferOps.insert(view))
          worklist.push_back(view.getResult());
        continue;
      }
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() != buffer || page.copy ||
            copy->getBlock() != insert->getBlock() ||
            !isGMSource(copy.getSource()))
          return std::nullopt;
        page.copy = copy;
        continue;
      }
      if (auto fill = dyn_cast<linalg::FillOp>(user)) {
        if (fill.getOutputs()[0] != alloc || !isZero(fill.getInputs()[0]))
          return std::nullopt;
        padded = true;
        bufferOps.insert(fill);
        continue;
      }
      if (isa<annotation::MarkOp>(user) &&
          !user->hasAttr(hivm::kMayImplicitTransposeWithLastAxis)) {
        bufferOps.insert(user);
        continue;
      }
      return std::nullopt;
    }
  }
  if (!page.copy || !page.copy->isBeforeInBlock(tensor))
    return std::nullopt;
  // Moving initialization past a writer would change the loaded values.
  for (Operation *op : bufferOps) {
    if (!isa<linalg::FillOp>(op))
      continue;
    while (op && op->getBlock() != insert->getBlock())
      op = op->getParentOp();
    if (!op || !op->isBeforeInBlock(page.copy))
      return std::nullopt;
  }
  auto srcType = dyn_cast<MemRefType>(page.copy.getSource().getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!srcType || srcType.getRank() != 2 ||
#if defined(__LLVM_MAJOR_VERSION_22_COMPATIBLE__)
      failed(srcType.getStridesAndOffset(strides, offset)) ||
#else
      failed(getStridesAndOffset(srcType, strides, offset)) ||
#endif
      strides[1] != 1 || srcType.getDimSize(1) <= 0 ||
      srcType.getDimSize(1) > alloc.getType().getDimSize(1))
    return std::nullopt;
  if (!padded && srcType.getShape() != alloc.getType().getShape())
    return std::nullopt;
  page.bufferOps.assign(bufferOps.begin(), bufferOps.end());
  return page;
}

} // namespace

std::optional<SmallVector<CVPipeline::CubePageLoader>>
CVPipeline::getCubePageLoaders(tensor::InsertSliceOp root) {
  if (!root)
    return std::nullopt;
  auto type = root.getType();
  if (type.getRank() != 2 || !type.hasStaticShape() ||
      (!type.getElementType().isF16() && !type.getElementType().isBF16()) ||
      type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0 ||
      type.getDimSize(0) % 16 || type.getDimSize(1) % 16)
    return std::nullopt;

  int64_t end = type.getDimSize(0);
  SmallVector<CubePageLoader> pages;
  Value current = root.getResult();
  while (auto insert = current.getDefiningOp<tensor::InsertSliceOp>()) {
    auto sourceType = insert.getSourceType();
    if (insert != root && !insert->hasOneUse())
      return std::nullopt;
    if (insert->getBlock() != root->getBlock() || sourceType.getRank() != 2 ||
        !sourceType.hasStaticShape() || insert.getType() != type ||
        insert.getStaticOffsets()[1] != 0 ||
        insert.getStaticSizes()[1] != type.getDimSize(1) ||
        insert.getStaticSizes()[0] != sourceType.getDimSize(0) ||
        llvm::any_of(insert.getStaticStrides(),
                     [](int64_t x) { return x != 1; }))
      return std::nullopt;
    int64_t row = insert.getStaticOffsets()[0];
    int64_t rows = sourceType.getDimSize(0);
    // Each page occupies whole NZ rows, possibly part of one 16-row block.
    if (row < 0 || rows <= 0 || row + rows != end ||
        (rows < 16 ? row % 16 + rows > 16 : row % 16 || rows % 16))
      return std::nullopt;
    auto page = getPageLoader(insert);
    if (!page)
      return std::nullopt;
    pages.push_back(std::move(*page));
    end = row;
    current = insert.getDest();
  }
  if (end != 0 || !isa_and_nonnull<tensor::EmptyOp, linalg::FillOp>(
                      current.getDefiningOp()))
    return std::nullopt;
  std::reverse(pages.begin(), pages.end());
  return pages;
}

namespace {

bool mayModifyOrFree(Operation *op, Value buffer, AliasAnalysis &aliases) {
  auto effects = getEffectsRecursively(op);
  if (!effects)
    return true;
  for (const auto &effect : *effects) {
    if (!isa<MemoryEffects::Write, MemoryEffects::Free>(effect.getEffect()))
      continue;
    if (!effect.getValue() || !aliases.alias(effect.getValue(), buffer).isNo())
      return true;
  }
  return false;
}

bool sameScalar(Value lhs, Value rhs, AliasAnalysis &aliases,
                unsigned depth = 0) {
  if (lhs == rhs)
    return true;
  Attribute lhsAttr, rhsAttr;
  if (matchPattern(lhs, m_Constant(&lhsAttr)) &&
      matchPattern(rhs, m_Constant(&rhsAttr)))
    return lhsAttr == rhsAttr;
  Operation *a = lhs.getDefiningOp(), *b = rhs.getDefiningOp();
  if (depth == 12 || !a || !b || a->getName() != b->getName() ||
      lhs.getType() != rhs.getType() || a->getNumResults() != 1 ||
      b->getNumResults() != 1 || a->getNumRegions() || b->getNumRegions() ||
      a->getNumOperands() != b->getNumOperands() ||
      a->getPropertiesAsAttribute() != b->getPropertiesAsAttribute())
    return false;
  NamedAttrList aAttrs(a->getAttrs()), bAttrs(b->getAttrs());
  for (StringRef name : {CVPipeline::kBlockId, CVPipeline::kCoreType}) {
    aAttrs.erase(name);
    bAttrs.erase(name);
  }
  if (aAttrs != bAttrs)
    return false;
  if (auto load = dyn_cast<memref::LoadOp>(a)) {
    // Compute-block planning clones counts and their address arithmetic.
    // Equal addresses are insufficient if memory changed between the reads.
    if (a->getBlock() != b->getBlock() || a->hasAttr("volatile") ||
        llvm::any_of(a->getUsers(), [](Operation *user) {
          return isa<annotation::MarkOp>(user);
        }) ||
        llvm::any_of(b->getUsers(), [](Operation *user) {
          return isa<annotation::MarkOp>(user);
        }))
      return false;
    Operation *first = a->isBeforeInBlock(b) ? a : b;
    Operation *last = first == a ? b : a;
    for (Operation *op = first->getNextNode(); op != last;
         op = op->getNextNode())
      if (mayModifyOrFree(op, load.getMemRef(), aliases))
        return false;
  } else if (!isMemoryEffectFree(a) || !isSpeculatable(a) ||
             !isMemoryEffectFree(b) || !isSpeculatable(b)) {
    return false;
  }
  for (auto [aOperand, bOperand] :
       llvm::zip(a->getOperands(), b->getOperands()))
    if (!sameScalar(aOperand, bOperand, aliases, depth + 1))
      return false;
  return true;
}

// An executing signed scf.for iteration satisfies iv < upper, including
// every operand of a signed minimum used as the upper bound.
bool upperImpliesLessThan(Value upper, Value bound, AliasAnalysis &aliases) {
  if (sameScalar(upper, bound, aliases))
    return true;
  auto min = upper.getDefiningOp<arith::MinSIOp>();
  return min && (upperImpliesLessThan(min.getLhs(), bound, aliases) ||
                 upperImpliesLessThan(min.getRhs(), bound, aliases));
}

bool isTrueInLoop(Value condition, scf::ForOp loop, AliasAnalysis &aliases) {
  if (matchPattern(condition, m_One()))
    return true;
  if (auto andOp = condition.getDefiningOp<arith::AndIOp>())
    return isTrueInLoop(andOp.getLhs(), loop, aliases) &&
           isTrueInLoop(andOp.getRhs(), loop, aliases);
  auto cmp = condition.getDefiningOp<arith::CmpIOp>();
  return cmp && cmp.getPredicate() == arith::CmpIPredicate::slt &&
         cmp.getLhs() == loop.getInductionVar() &&
         upperImpliesLessThan(loop.getUpperBound(), cmp.getRhs(), aliases);
}

struct MaskedMetadataLoad {
  scf::IfOp ifOp;
  memref::LoadOp load;
  memref::ReinterpretCastOp view;
  Value other;
};

std::optional<MaskedMetadataLoad> getMaskedMetadataLoad(scf::IfOp ifOp) {
  if (ifOp.getNumResults() != 1 || !ifOp.getResult(0).getType().isInteger(32) ||
      ifOp.getElseRegion().empty())
    return std::nullopt;
  auto &thenBlock = ifOp.getThenRegion().front();
  auto &elseBlock = ifOp.getElseRegion().front();
  if (!llvm::hasSingleElement(elseBlock))
    return std::nullopt;
  auto load = cast<scf::YieldOp>(thenBlock.getTerminator())
                  .getOperand(0)
                  .getDefiningOp<memref::LoadOp>();
  if (!load || load->getBlock() != &thenBlock || !load->hasOneUse() ||
      load->hasAttr("volatile") || load.getIndices().size() != 1 ||
      !matchPattern(load.getIndices()[0], m_Zero()))
    return std::nullopt;
  auto view = load.getMemRef().getDefiningOp<memref::ReinterpretCastOp>();
  if (!view || view->getBlock() != &thenBlock || !view->hasOneUse() ||
      !isGMSource(view.getSource()) ||
      view.getSource().getParentBlock() == &thenBlock ||
      view.getType().getShape() != ArrayRef<int64_t>{1} ||
      view.getStaticSizes() != ArrayRef<int64_t>{1} ||
      view.getStaticStrides() != ArrayRef<int64_t>{1})
    return std::nullopt;
  for (Operation &op : thenBlock.without_terminator()) {
    if (&op == load || &op == view)
      continue;
    // Hoist only address arithmetic. In particular, do not speculate another
    // memory access, a division that may trap, or a volatile annotation.
    if (op.getName().getDialectNamespace() != "arith" ||
        !isMemoryEffectFree(&op) || !isSpeculatable(&op))
      return std::nullopt;
  }
  return MaskedMetadataLoad{
      ifOp, load, view,
      cast<scf::YieldOp>(elseBlock.getTerminator()).getOperand(0)};
}

void copyBlockAssignment(Operation *from, Operation *to) {
  for (StringRef name : {CVPipeline::kBlockId, CVPipeline::kCoreType})
    if (Attribute attr = from->getAttr(name))
      to->setAttr(name, attr);
}

// A known active metadata read supplies a valid address for masked-off reads
// later in the same iteration. Select the address *before* loading, then select
// `other` afterwards. This preserves tail pages without a branch per page.
void simplifyMetadataMasks(
    ArrayRef<SmallVector<CVPipeline::CubePageLoader>> chains) {
  llvm::MapVector<Block *, llvm::SetVector<Operation *>> candidates;
  for (const auto &pages : chains) {
    for (auto page : pages) {
      Block *block = page.copy->getBlock();
      SmallVector<Value> worklist{page.copy.getSource()};
      llvm::SmallPtrSet<Operation *, 32> visited;
      while (!worklist.empty()) {
        Operation *op = worklist.pop_back_val().getDefiningOp();
        if (!op || op->getBlock() != block || !visited.insert(op).second)
          continue;
        if (isa<scf::IfOp>(op)) {
          candidates[block].insert(op);
          continue;
        }
        if (isMemoryEffectFree(op))
          llvm::append_range(worklist, op->getOperands());
      }
    }
  }
  for (auto &[block, ifOps] : candidates) {
    auto loop = dyn_cast<scf::ForOp>(block->getParentOp());
    if (!loop || loop->hasAttr("unsignedCmp"))
      continue;
    SmallVector<Operation *> ordered(ifOps.begin(), ifOps.end());
    llvm::sort(ordered, [](Operation *a, Operation *b) {
      return a->isBeforeInBlock(b);
    });
    struct SafeRead {
      memref::ReinterpretCastOp view;
      Operation *after;
    };
    SmallVector<SafeRead> safeReads;
    AliasAnalysis aliases(loop->getParentOp());
    for (Operation *op : ordered) {
      auto metadata = getMaskedMetadataLoad(cast<scf::IfOp>(op));
      if (!metadata)
        continue;
      auto [ifOp, load, view, other] = *metadata;
      bool active = isTrueInLoop(ifOp.getCondition(), loop, aliases);
      SafeRead *safe = nullptr;
      if (!active) {
        for (auto &read : llvm::reverse(safeReads)) {
          if (read.view.getSource() != view.getSource() ||
              read.view.getType() != view.getType())
            continue;
          bool clobbered = false;
          for (Operation *between = read.after->getNextNode(); between != op;
               between = between->getNextNode()) {
            if (mayModifyOrFree(between, view.getSource(), aliases)) {
              clobbered = true;
              break;
            }
          }
          if (!clobbered) {
            safe = &read;
            break;
          }
        }
        if (!safe)
          continue;
      }
      OpBuilder builder(ifOp);
      IRMapping mapping;
      for (Operation &addressOp :
           ifOp.getThenRegion().front().without_terminator()) {
        if (&addressOp == load || &addressOp == view)
          continue;
        copyBlockAssignment(ifOp, builder.clone(addressOp, mapping));
      }
      OpFoldResult offset = view.getMixedOffsets()[0];
      if (auto value = dyn_cast<Value>(offset))
        offset = mapping.lookupOrDefault(value);
      if (!active) {
        Value wanted =
            getValueOrCreateConstantIndexOp(builder, ifOp.getLoc(), offset);
        Value fallback = getValueOrCreateConstantIndexOp(
            builder, ifOp.getLoc(), safe->view.getMixedOffsets()[0]);
        auto selected = builder.create<arith::SelectOp>(
            ifOp.getLoc(), ifOp.getCondition(), wanted, fallback);
        copyBlockAssignment(ifOp, selected);
        offset = selected.getResult();
      }
      auto safeView = builder.create<memref::ReinterpretCastOp>(
          view.getLoc(), view.getType(), view.getSource(), offset,
          view.getMixedSizes(), view.getMixedStrides());
      copyBlockAssignment(ifOp, safeView);
      auto scalar = builder.create<memref::LoadOp>(
          load.getLoc(), safeView,
          mapping.lookupOrDefault(load.getIndices()[0]));
      scalar->setAttrs(load->getAttrs());
      copyBlockAssignment(ifOp, scalar);
      Value result = scalar;
      if (active) {
        safeReads.push_back({safeView, scalar});
      } else {
        safe->after = scalar;
        auto selected = builder.create<arith::SelectOp>(
            ifOp.getLoc(), ifOp.getCondition(), result, other);
        copyBlockAssignment(ifOp, selected);
        result = selected;
      }
      ifOp.getResult(0).replaceAllUsesWith(result);
      ifOp.erase();
    }
  }
}

// Only share preparation whose scalar GM reads are already shared in SSA.
// In particular, a V address derived from QK/softmax must not move before QK.
bool collectPagePreparation(Value value, Block *body,
                            llvm::SetVector<Operation *> &ops,
                            llvm::DenseSet<Operation *> &reads) {
  auto *op = value.getDefiningOp();
  if (!op) {
    auto arg = cast<BlockArgument>(value);
    if (arg.getOwner() != body)
      return true;
    auto loop = dyn_cast<scf::ForOp>(body->getParentOp());
    return loop && value == loop.getInductionVar();
  }
  if (op->getBlock() != body)
    return !body->findAncestorOpInBlock(*op);
  if (!ops.insert(op))
    return true;

  bool valid = true;
  op->walk([&](Operation *nested) {
    for (Value result : nested->getResults()) {
      if (!result.getType().isIntOrIndexOrFloat() &&
          !(isa<MemRefType>(result.getType()) && isGMSource(result)))
        valid = false;
    }
    if (auto load = dyn_cast<memref::LoadOp>(nested)) {
      if (!load.getType().isIntOrIndexOrFloat() ||
          !isGMSource(load.getMemRef()) || load->hasAttr("volatile") ||
          llvm::any_of(load->getUsers(), [](Operation *user) {
            return isa<annotation::MarkOp>(user);
          }))
        valid = false;
      reads.insert(load);
    } else if (!isa<scf::IfOp, scf::YieldOp>(nested) &&
               !isMemoryEffectFree(nested)) {
      valid = false;
    }
    if (nested->getNumRegions() && !isa<scf::IfOp>(nested))
      valid = false;
    for (Value operand : nested->getOperands()) {
      Operation *def = operand.getDefiningOp();
      if (def && (def == op || op->isProperAncestor(def)))
        continue;
      if (!collectPagePreparation(operand, body, ops, reads))
        valid = false;
    }
  });
  return valid;
}

bool canSharePreparation(ArrayRef<CVPipeline::CubePageLoader> first,
                         ArrayRef<CVPipeline::CubePageLoader> second) {
  auto firstRoot = first.back().insert;
  auto secondRoot = second.back().insert;
  Block *body = firstRoot->getBlock();
  auto loop = dyn_cast<scf::ForOp>(body->getParentOp());
  if (!loop || secondRoot->getBlock() != body ||
      !firstRoot->isBeforeInBlock(secondRoot) ||
      firstRoot.getType() != secondRoot.getType() ||
      first.size() != second.size() || !firstRoot->hasOneUse() ||
      !secondRoot->hasOneUse())
    return false;
  auto producerId = firstRoot->getAttr(CVPipeline::kBlockId);
  auto consumerId = secondRoot->getAttr(CVPipeline::kBlockId);
  if (!producerId || !consumerId || producerId == consumerId)
    return false;
  if ((*firstRoot->getUsers().begin())->getAttr(CVPipeline::kBlockId) !=
          producerId ||
      (*secondRoot->getUsers().begin())->getAttr(CVPipeline::kBlockId) !=
          consumerId)
    return false;
  // MarkMainLoop keeps only innermost pipeline loops. Sharing in an outer
  // loop would otherwise lose the counter that protects this buffer.
  bool nestedLoop = false;
  loop.walk([&](Operation *op) {
    if (op != loop && isa<scf::ForOp, scf::WhileOp>(op))
      nestedLoop = true;
  });
  if (nestedLoop)
    return false;
  AliasAnalysis aliases(loop->getParentOp());
  for (unsigned i = 0; i < first.size(); ++i) {
    auto left = first[i];
    auto right = second[i];
    if (left.insert.getStaticOffsets() != right.insert.getStaticOffsets() ||
        left.insert.getSourceType() != right.insert.getSourceType())
      return false;
    llvm::SetVector<Operation *> leftOps, rightOps;
    llvm::DenseSet<Operation *> leftReads, rightReads;
    if (!collectPagePreparation(left.copy.getSource(), body, leftOps,
                                leftReads) ||
        !collectPagePreparation(right.copy.getSource(), body, rightOps,
                                rightReads) ||
        leftReads.empty() || leftReads != rightReads)
      return false;
    // Conservatively require an unchanged payload while pairing the chains.
    // Shared metadata alone does not establish alias independence.
    if (!left.copy->isBeforeInBlock(right.copy))
      return false;
    for (Operation *between = left.copy; between != right.copy;
         between = between->getNextNode()) {
      if (mayModifyOrFree(between, right.copy.getSource(), aliases))
        return false;
    }
    for (Operation *op : rightOps) {
      auto id = op->getAttr(CVPipeline::kBlockId);
      if (id != producerId && id != consumerId &&
          !op->isBeforeInBlock(first.front().copy))
        return false;
      if (id != consumerId)
        continue;
      for (Operation *user : op->getUsers()) {
        auto *child = body->findAncestorOpInBlock(*user);
        if (!child || child->getAttr(CVPipeline::kBlockId) != consumerId)
          return false;
      }
    }
  }
  return true;
}

struct MaterializedPages {
  memref::AllocOp buffer;
  arith::ConstantOp zero;
  linalg::FillOp fill;
  memref::MemorySpaceCastOp reader;
  SmallVector<hivm::ND2NZOp> loads;
};

MaterializedPages materializePages(ArrayRef<CVPipeline::CubePageLoader> pages) {
  MaterializedPages result;
  auto root = pages.back().insert;
  auto type = root.getType();
  int64_t rows = type.getDimSize(0), cols = type.getDimSize(1);
  OpBuilder builder(root.getContext());
  auto cbuf = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
  auto nzType = MemRefType::get({cols / 16, rows / 16, 16, 16},
                                type.getElementType(), nullptr, cbuf);
  // The whole aggregate belongs to the matmul's compute block. Keeping the
  // buffer inside the same iteration also prevents writes racing with MMAD.
  auto tag = [&](Operation *op) {
    op->setAttr(CVPipeline::kCoreType, builder.getStringAttr("CUBE"));
    if (auto id = root->getAttr(CVPipeline::kBlockId))
      op->setAttr(CVPipeline::kBlockId, id);
  };
  auto indexAttrs = [&](ArrayRef<int64_t> values) {
    return llvm::map_to_vector(values, [&](int64_t value) -> OpFoldResult {
      return builder.getIndexAttr(value);
    });
  };
  Location loc = root.getLoc();
  Operation *firstCopy = pages.front().copy;
  for (const auto &page : pages)
    if (page.copy->isBeforeInBlock(firstCopy))
      firstCopy = page.copy;
  builder.setInsertionPoint(firstCopy);
  auto aggregate = builder.create<memref::AllocOp>(loc, nzType);
  result.buffer = aggregate;
  tag(aggregate);
  auto zero = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(type.getElementType()));
  result.zero = zero;
  tag(zero);
  auto fill = builder.create<linalg::FillOp>(loc, ValueRange{zero},
                                             ValueRange{aggregate});
  result.fill = fill;
  tag(fill);

  for (auto page : pages) {
    builder.setInsertionPoint(page.copy);
    int64_t row = page.insert.getStaticOffsets()[0];
    int64_t pageRows = page.insert.getSourceType().getDimSize(0);
    SmallVector<OpFoldResult> offsets = indexAttrs({0, row / 16, row % 16, 0});
    SmallVector<OpFoldResult> sizes = indexAttrs(
        {cols / 16, (pageRows + 15) / 16, std::min(pageRows, int64_t{16}), 16});
    SmallVector<OpFoldResult> strides = indexAttrs({1, 1, 1, 1});
    auto view = builder.create<memref::SubViewOp>(loc, aggregate, offsets,
                                                  sizes, strides);
    tag(view);
    Value source = page.copy.getSource();
    // ND2NZ uses the full aggregate's N-stride, so a short page writes only
    // its own rows. The initial zero fill covers the remaining masked rows.
    auto load = builder.create<hivm::ND2NZOp>(loc, TypeRange{}, source, view,
                                              builder.getUnitAttr());
    tag(load);
    result.loads.push_back(load);
  }

  builder.setInsertionPoint(root);
  auto ndType =
      MemRefType::get(type.getShape(), type.getElementType(), nullptr, cbuf);
  auto layout = builder.create<hivm::ConvertLayoutOp>(
      loc, ndType, aggregate,
      builder.getAttr<hivm::DataLayoutAttr>(hivm::DataLayout::nZ),
      builder.getAttr<hivm::DataLayoutAttr>(hivm::DataLayout::ND));
  tag(layout);
  auto cast = builder.create<memref::MemorySpaceCastOp>(
      loc, MemRefType::get(type.getShape(), type.getElementType()),
      layout.getResult());
  tag(cast);
  result.reader = cast;
  auto tensor =
      builder.create<bufferization::ToTensorOp>(loc, type, cast, true, true);
  tag(tensor);
  root.replaceAllUsesWith(tensor.getResult());

  llvm::SetVector<Operation *> cleanup;
  for (const auto &page : llvm::reverse(pages)) {
    for (Operation *op : page.bufferOps)
      cleanup.insert(op->getParentOp());
    page.insert->erase();
    page.tensor->erase();
    page.copy->erase();
    for (Operation *op : llvm::reverse(page.bufferOps))
      op->erase();
  }
  for (Operation *op : cleanup)
    if (isa<scf::IfOp>(op) && isOpTriviallyDead(op))
      op->erase();
  return result;
}

// Share only raw metadata. Moving all V payload loads into QK couples their
// buffer lifetimes and prevents QK from running ahead of PV. A small SSBuffer
// ring preserves one GM metadata read per page while each matmul owns its L1
// buffer and DMA schedule. Occupancy prevents a slot from being overwritten
// until the consumer finishes; each block indexes its own logical iteration.
bool sharePageMetadata(MaterializedPages &first, MaterializedPages &second,
                       int groupId) {
  constexpr int64_t slots = 2;
  auto loop = cast<scf::ForOp>(first.buffer->getBlock()->getParentOp());
  auto producerId = first.fill->getAttr(CVPipeline::kBlockId);
  auto consumerId = second.fill->getAttr(CVPipeline::kBlockId);
  SmallVector<memref::LoadOp> metadata;
  llvm::SmallPtrSet<Operation *, 32> oldPreparation;
  for (auto load : second.loads) {
    llvm::SetVector<Operation *> ops;
    llvm::DenseSet<Operation *> reads;
    if (!collectPagePreparation(load.getSrc(), loop.getBody(), ops, reads) ||
        reads.size() != 1)
      return false;
    auto scalar = cast<memref::LoadOp>(*reads.begin());
    if (!scalar.getType().isInteger(32) || scalar->getBlock() != loop.getBody())
      return false;
    metadata.push_back(scalar);
    for (Operation *op : ops)
      if (op->getAttr(CVPipeline::kBlockId) == consumerId)
        oldPreparation.insert(op);
  }
  OpBuilder builder(loop);
  if (!loop->hasAttr(CVPipeline::kBlockId))
    loop->setAttr(CVPipeline::kBlockId,
                  builder.getI32IntegerAttr(CVPipeline::getAvailableBlockId(
                      loop->getParentOfType<ModuleOp>())));
  auto tag = [&](Operation *op, Attribute id) {
    op->setAttr(CVPipeline::kBlockId, id);
    op->setAttr(CVPipeline::kCoreType, builder.getStringAttr("CUBE"));
  };
  auto module = loop->getParentOfType<ModuleOp>();
  auto base = triton::SSBufferManager::reserveBytes(
      module, slots * metadata.size() * sizeof(int32_t));
  if (!base)
    return false;
  auto addr = builder.create<arith::ConstantIntOp>(loop.getLoc(), *base, 64);
  tag(addr, loop->getAttr(CVPipeline::kBlockId));
  auto cache = builder.create<hivm::PointerCastOp>(
      loop.getLoc(),
      MemRefType::get(
          {slots, static_cast<int64_t>(metadata.size())}, builder.getI32Type(),
          nullptr,
          builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::SSBUF)),
      addr.getResult());
  tag(cache, loop->getAttr(CVPipeline::kBlockId));
  auto slot = [&](Operation *before, Attribute id) {
    builder.setInsertionPoint(before);
    auto loc = before->getLoc();
    auto relative = builder.create<arith::SubIOp>(loc, loop.getInductionVar(),
                                                  loop.getLowerBound());
    tag(relative, id);
    // The trip distance is nonnegative and scf.for has a positive step.
    // Unsigned arithmetic also lets power-of-two slots lower to a bit mask.
    auto iteration =
        builder.create<arith::DivUIOp>(loc, relative, loop.getStep());
    tag(iteration, id);
    auto capacity = builder.create<arith::ConstantOp>(
        loc, builder.getIntegerAttr(iteration.getType(), slots));
    tag(capacity, id);
    auto rem = builder.create<arith::RemUIOp>(loc, iteration, capacity);
    tag(rem, id);
    auto index = builder.createOrFold<arith::IndexCastOp>(
        loc, builder.getIndexType(), rem);
    if (auto *op = index.getDefiningOp())
      tag(op, id);
    return index;
  };
  Value producerSlot = slot(first.loads.front(), producerId);
  Value consumerSlot = slot(second.loads.front(), consumerId);
  for (unsigned i = 0; i < metadata.size(); ++i) {
    auto raw = metadata[i];
    builder.setInsertionPointAfter(first.loads[i]);
    auto page = builder.create<arith::ConstantIndexOp>(raw.getLoc(), i);
    tag(page, producerId);
    auto store = builder.create<memref::StoreOp>(
        raw.getLoc(), raw, cache, ValueRange{producerSlot, page});
    tag(store, producerId);
    store->setAttr(CVPipeline::kSharedPageWrite, builder.getUnitAttr());
    if (i == 0) {
      store->setAttr(CVPipeline::kSharedPageSlots,
                     builder.getI32IntegerAttr(slots));
      store->setAttr(
          CVPipeline::kIntraDeps,
          builder.getI32ArrayAttr({groupId, CVPipeline::crossCoreProducerId}));
    }
    builder.setInsertionPoint(second.loads[i]);
    auto read = builder.create<memref::LoadOp>(raw.getLoc(), cache,
                                               ValueRange{consumerSlot, page});
    tag(read, consumerId);
    if (i == 0)
      read->setAttr(
          CVPipeline::kIntraDeps,
          builder.getI32ArrayAttr({groupId, CVPipeline::crossCoreConsumerId}));
    auto mark =
        builder.create<annotation::MarkOp>(raw.getLoc(), read.getResult());
    tag(mark, consumerId);
    mark->setAttr(triton::kMemrefExtVolatile, builder.getUnitAttr());
    // Rebuild the consumer view from its cached raw value. Retain all masks,
    // selects, and row-count clamping rather than caching a partially decoded
    // address that could lose the masked-off page semantics.
    IRMapping mapping;
    mapping.map(raw.getResult(), read.getResult());
    std::function<Value(Value)> clone = [&](Value value) -> Value {
      if (mapping.contains(value))
        return mapping.lookup(value);
      Operation *op = value.getDefiningOp();
      if (!op || op->getBlock() != loop.getBody())
        return value;
      op->walk([&](Operation *nested) {
        for (Value operand : nested->getOperands()) {
          Operation *def = operand.getDefiningOp();
          if (def && (def == op || op->isProperAncestor(def)))
            continue;
          mapping.map(operand, clone(operand));
        }
      });
      auto *copied = builder.clone(*op, mapping);
      copied->walk([&](Operation *nested) { tag(nested, consumerId); });
      return mapping.lookup(value);
    };
    second.loads[i].getSrcMutable().assign(clone(second.loads[i].getSrc()));
  }
  // Dead original views would keep the GM metadata chain live when CloneOps
  // makes each compute block independent. Remove only replaced preparation.
  for (Operation &op :
       llvm::make_early_inc_range(llvm::reverse(*loop.getBody())))
    if (oldPreparation.contains(&op) && isOpTriviallyDead(&op))
      op.erase();
  return true;
}

class MaterializeCubePageLoadersPass
    : public PassWrapper<MaterializeCubePageLoadersPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeCubePageLoadersPass)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<annotation::AnnotationDialect, hivm::HIVMDialect,
                    arith::ArithDialect, memref::MemRefDialect,
                    bufferization::BufferizationDialect, linalg::LinalgDialect,
                    scf::SCFDialect>();
  }
  StringRef getArgument() const override {
    return "materialize-cube-page-loaders";
  }
  StringRef getDescription() const override {
    return "Load unrolled matmul pages directly into one CUBE L1 buffer";
  }
  void runOnOperation() override {
    if (CVPipeline::hasFallbackAttr(getOperation()))
      return;
    SmallVector<SmallVector<CVPipeline::CubePageLoader>> chains;
    getOperation().walk([&](tensor::InsertSliceOp root) {
      if (CVPipeline::getOpCoreType(root) != CVPipeline::CoreType::CUBE_ONLY)
        return;
      if (llvm::any_of(root->getUsers(), [](Operation *user) {
            return isa<tensor::InsertSliceOp>(user);
          }))
        return;
      auto pages = CVPipeline::getCubePageLoaders(root);
      if (!pages || llvm::any_of(*pages, [&](const auto &page) {
            return page.copy->getAttr(CVPipeline::kBlockId) !=
                       root->getAttr(CVPipeline::kBlockId) ||
                   page.insert->getAttr(CVPipeline::kBlockId) !=
                       root->getAttr(CVPipeline::kBlockId);
          })) {
        CVPipeline::setFallbackAttr(getOperation(), CVPipeline::ERRCODE_FAILED);
        return;
      }
      chains.push_back(std::move(*pages));
    });
    if (CVPipeline::hasFallbackAttr(getOperation()))
      return;
    simplifyMetadataMasks(chains);
    SmallVector<std::pair<unsigned, unsigned>> shared;
    llvm::DenseSet<unsigned> paired;
    for (unsigned i = 0; i < chains.size(); ++i) {
      if (paired.contains(i))
        continue;
      for (unsigned j = i + 1; j < chains.size(); ++j) {
        if (!paired.contains(j) && canSharePreparation(chains[i], chains[j])) {
          shared.emplace_back(i, j);
          paired.insert(i);
          paired.insert(j);
          break;
        }
      }
    }
    int groupId = 0;
    getOperation().walk([&](Operation *op) {
      auto deps = op->getAttrOfType<ArrayAttr>(CVPipeline::kIntraDeps);
      if (deps && deps.size() == 2)
        groupId = std::max(
            groupId, static_cast<int>(cast<IntegerAttr>(deps[0]).getInt()) + 1);
    });
    SmallVector<MaterializedPages> materialized;
    for (const auto &pages : chains)
      materialized.push_back(materializePages(pages));
    SmallVector<hivm::ND2NZOp> loads;
    for (auto [first, second] : shared) {
      if (!sharePageMetadata(materialized[first], materialized[second],
                             groupId))
        continue;
      ++groupId;
      llvm::append_range(loads, materialized[first].loads);
      llvm::append_range(loads, materialized[second].loads);
    }
    // Empty and masked-off pages are already zero-filled. Avoid submitting a
    // zero-row DMA; the metadata read and ring write still occur for every
    // page.
    for (auto load : loads) {
      OpBuilder builder(load);
      auto zero = builder.create<arith::ConstantIndexOp>(load.getLoc(), 0);
      auto rows = builder.createOrFold<memref::DimOp>(
          load.getLoc(), load.getSrc(), zero.getResult());
      auto active = builder.create<arith::CmpIOp>(
          load.getLoc(), arith::CmpIPredicate::ugt, rows, zero);
      for (Value value : {rows, Value(zero), Value(active)})
        if (auto *op = value.getDefiningOp())
          copyBlockAssignment(load, op);
      auto guarded = builder.create<scf::IfOp>(load.getLoc(), active, false);
      copyBlockAssignment(load, guarded);
      load->moveBefore(guarded.thenYield());
    }
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createMaterializeCubePageLoadersPass() {
  return std::make_unique<MaterializeCubePageLoadersPass>();
}
