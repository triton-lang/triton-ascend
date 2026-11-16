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

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/Support/Casting.h"
#include "llvm/Support/Debug.h"

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/Visitors.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"

#include "ascend/include/DynamicCVPipeline/Common/IROrder.h"
#include "ascend/include/DynamicCVPipeline/Common/LoopScheduleAttrs.h"

#include "DynamicCVPipeline/Common/ScalarClosure.h"
#include "DynamicCVPipeline/Common/Utils.h"

static constexpr const char *DEBUG_TYPE = "ScalarClosure";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LDBG(...)                                                              \
  LLVM_DEBUG({                                                                 \
    DBGS();                                                                    \
    llvm::dbgs() << __VA_ARGS__ << "\n";                                       \
  })

using namespace mlir;
using namespace CVPipeline;

/// A scalar producer still touches a buffer when it loads through a memref or
/// reports a read/write effect, so a clone would repeat the access.
static bool touchesMemory(Operation *op) {
  if (isa<memref::LoadOp, memref::StoreOp>(op)) {
    return true;
  }
  auto effects = dyn_cast<MemoryEffectOpInterface>(op);
  return effects && !effects.hasNoEffect();
}

/// A producer is cloned by `capture`, so it has to be a pure scalar
/// computation: anything that touches a buffer would repeat the access once
/// per clone. The judgement walks the op itself, because whether an op may be
/// copied is a property of the op, not of the memory effects it reports.
///
/// nested op     | clone-safe because
/// scf dialect   | control flow over scalars
/// terminator    | closes a nested block, no access of its own
/// view-like     | aliases a buffer without creating one
/// scalar result | every result is a non-shaped value and no effect is reported
static bool isCloneSafeScalar(Operation *op) {
  bool safe = true;
  op->walk([&](Operation *nested) {
    bool isScalar = llvm::all_of(nested->getResultTypes(), [](Type type) {
      return !isa<ShapedType>(type);
    });
    if (isa<scf::SCFDialect>(nested->getDialect()) ||
        nested->hasTrait<OpTrait::IsTerminator>() ||
        isa<ViewLikeOpInterface>(nested) ||
        (isScalar && !touchesMemory(nested))) {
      return WalkResult::advance();
    }
    safe = false;
    return WalkResult::interrupt();
  });
  return safe;
}

void ScalarClosure::collectOuterDependency(Operation *op) {
  op->walk([this](Operation *nestedOp) {
    for (auto operand : nestedOp->getOperands()) {
      collectScalarClosure(operand);
    }
  });
};

void ScalarClosure::collectScalarClosure(Value val) {
  if (!val || isa<TensorType>(val.getType())) {
    return;
  }
  Operation *defOp = val.getDefiningOp();
  if (!defOp || llvm::isa<memref::AllocOp, memref::AllocaOp>(defOp)) {
    return;
  }

  // A closure op is cloned into another block by `capture`, so a clone repeats
  // any memory effect it has. `collectImpure` opts out for a caller that never
  // clones; every other caller keeps to a pure scalar computation.
  if (!collectImpure && !isCloneSafeScalar(defOp)) {
    return;
  }

  auto *defBlock = defOp->getBlock();
  if (!defBlock || (defBlock != block &&
                    // keep memref dependencies of the main loop out
                    (!includeParent || defBlock != parentBlock))) {
    return;
  }

  // A scalar result does not make its producer a scalar op.
  // Keep multi-result producers outside and capture their values.
  if (defOp->getNumResults() != 1) {
    return;
  }

  collectOuterDependency(defOp);
  scalarOps.insert(defOp);
}

void ScalarClosure::collect() {
  if (block == nullptr) {
    return;
  }
  for (Operation *op : ops) {
    collectOuterDependency(op);
  }
}

ScalarClosure::ScalarClosure(Block *block, ArrayRef<Operation *> ops,
                             bool includeParent, bool collectImpure)
    : block(block), ops(ops), includeParent(includeParent),
      collectImpure(collectImpure) {
  if (!block) {
    return;
  }
  Operation *parentOp = block->getParentOp();
  if (!parentOp) {
    return;
  }
  parentBlock = parentOp->getBlock();
}

llvm::ArrayRef<Operation *> ScalarClosure::getScalars() const {
  return scalarOps.getArrayRef();
}

std::pair<SmallVector<Operation *>, IRMapping>
ScalarClosure::capture(int blockId) {
  if (scalarOps.empty()) {
    return {};
  }

  IRMapping mapper;
  // A clone reading an earlier clone must come after it, otherwise it would
  // reference an original that may sit after the insertion point.
  SmallVector<Operation *> opsToClone(scalarOps.begin(), scalarOps.end());
  llvm::sort(opsToClone, CVPipeline::isBeforeInIr);

  Operation *firstOp = nullptr;
  for (auto *op : ops) {
    if (op->getBlock() != block) {
      continue;
    }
    if (firstOp == nullptr || op->isBeforeInBlock(firstOp)) {
      firstOp = op;
    }
  }
  OpBuilder builder(firstOp);
  builder.setInsertionPoint(firstOp);

  SmallVector<Operation *> newOps;

  auto module = firstOp->getParentOfType<ModuleOp>();
  auto outerLoop = CVPipeline::hasOuterMainLoop(module)
                       ? firstOp->getParentOfType<scf::ForOp>()
                       : scf::ForOp();
  const bool outerEvent =
      outerLoop && outerLoop->hasAttr(CVPipeline::kOuterEventControlCount);

  LDBG("Collecting scalar closures for blockId: " << blockId);
  for (Operation *scalarOp : opsToClone) {
    if (llvm::is_contained(ops, scalarOp)) {
      // In `ops`, so not cloned: keep the clones that follow after it.
      builder.setInsertionPointAfter(scalarOp);
      continue;
    }

    LDBG("Clone: " << *scalarOp);

    Operation *cloned = builder.clone(*scalarOp, mapper);
    auto destinationId =
        Builder(scalarOp->getContext()).getI32IntegerAttr(blockId);
    cloned->setAttr(CVPipeline::kBlockId, destinationId);
    if (outerEvent && cloned->getNumRegions() &&
        scalarOp->hasAttr(CVPipeline::kBlockId)) {
      auto sourceId = scalarOp->getAttr(CVPipeline::kBlockId);
      // Keep a cloned scalar region in its consumer's scheduling group.
      // Differently tagged nested groups still retain their own identity.
      cloned->walk([&](Operation *nested) {
        if (nested->getAttr(CVPipeline::kBlockId) == sourceId) {
          nested->setAttr(CVPipeline::kBlockId, destinationId);
        }
      });
    }
    newOps.push_back(cloned);
  }

  return {newOps, mapper};
}
