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

#include "TritonToLinalg/ConvertModuloToMask.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Pass/Pass.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/Debug.h"

#define DEBUG_TYPE "convert-modulo-to-mask"

namespace ConvertModuloToMask {

using namespace mlir;
using namespace triton;

namespace {

static Value getScalarThroughSplat(Value v) {
  if (auto splat = v.getDefiningOp<triton::SplatOp>())
    return splat.getSrc();
  return nullptr;
}

static int64_t getMakeRangeSize(Value v) {
  if (auto mr = v.getDefiningOp<triton::MakeRangeOp>()) {
    return mr.getEnd() - mr.getStart();
  }
  if (auto addi = v.getDefiningOp<arith::AddIOp>()) {
    int64_t l = getMakeRangeSize(addi.getLhs());
    if (l > 0) return l;
    return getMakeRangeSize(addi.getRhs());
  }
  return 0;
}

static int64_t getDivisibility(Value scalar) {
  auto blockArg = dyn_cast<BlockArgument>(scalar);
  if (!blockArg)
    return 0;
  auto funcOp =
      dyn_cast<triton::FuncOp>(blockArg.getOwner()->getParentOp());
  if (!funcOp)
    return 0;
  unsigned idx = blockArg.getArgNumber();
  if (auto attr = funcOp.getArgAttrOfType<IntegerAttr>(idx, "tt.divisibility"))
    return attr.getInt();
  return 0;
}

static bool isSameScalar(Value a, Value b) {
  if (a == b)
    return true;
  if (auto ba = dyn_cast<BlockArgument>(a))
    if (auto bb = dyn_cast<BlockArgument>(b))
      return ba.getOwner() == bb.getOwner() &&
             ba.getArgNumber() == bb.getArgNumber();
  return false;
}

static bool feedsIntoStoreMask(Value cmpResult,
                               SmallPtrSetImpl<Operation *> &visited) {
  for (OpOperand &use : cmpResult.getUses()) {
    Operation *user = use.getOwner();
    if (!visited.insert(user).second)
      continue;
    if (auto store = dyn_cast<triton::StoreOp>(user)) {
      if (store.getMask() == cmpResult)
        return true;
    }
    if (isa<triton::BroadcastOp>(user)) {
      if (feedsIntoStoreMask(user->getResult(0), visited))
        return true;
    }
    if (isa<triton::ExpandDimsOp>(user)) {
      if (feedsIntoStoreMask(user->getResult(0), visited))
        return true;
    }
    if (isa<arith::AndIOp>(user)) {
      if (feedsIntoStoreMask(user->getResult(0), visited))
        return true;
    }
  }
  return false;
}

static bool hasStoreMaskGuard(Value unModulod, Value boundScalar) {
  for (OpOperand &use : unModulod.getUses()) {
    Operation *user = use.getOwner();
    auto cmp = dyn_cast<arith::CmpIOp>(user);
    if (!cmp)
      continue;
    if (cmp.getPredicate() != arith::CmpIPredicate::slt)
      continue;
    Value rhs = cmp.getRhs();
    Value rhsScalar = getScalarThroughSplat(rhs);
    if (rhsScalar && isSameScalar(rhsScalar, boundScalar)) {
      SmallPtrSet<Operation *, 16> visited;
      if (feedsIntoStoreMask(cmp.getResult(), visited))
        return true;
    }
  }
  for (OpOperand &use : unModulod.getUses()) {
    Operation *user = use.getOwner();
    if (!isa<triton::ExpandDimsOp>(user))
      continue;
    Value expanded = user->getResult(0);
    for (OpOperand &use2 : expanded.getUses()) {
      Operation *user2 = use2.getOwner();
      auto cmp = dyn_cast<arith::CmpIOp>(user2);
      if (!cmp)
        continue;
      if (cmp.getPredicate() != arith::CmpIPredicate::slt)
        continue;
      Value rhs = cmp.getRhs();
      Value rhsScalar = getScalarThroughSplat(rhs);
      if (rhsScalar && isSameScalar(rhsScalar, boundScalar)) {
        SmallPtrSet<Operation *, 16> visited;
        if (feedsIntoStoreMask(cmp.getResult(), visited))
          return true;
      }
    }
  }
  return false;
}

// Collect all tt.load operations reachable from a value through address
// computation chains (expand_dims, muli, broadcast, addi, addptr, scf.for
// iter_args, scf.yield).
static void collectLoadsFromAddress(Value addr,
                                    SmallVectorImpl<triton::LoadOp> &loads,
                                    SmallPtrSetImpl<Value> &visited) {
  if (!visited.insert(addr).second)
    return;
  for (OpOperand &use : addr.getUses()) {
    Operation *user = use.getOwner();
    if (auto load = dyn_cast<triton::LoadOp>(user)) {
      if (load.getPtr() == addr)
        loads.push_back(load);
      continue;
    }
    if (isa<triton::ExpandDimsOp, triton::BroadcastOp, triton::SplatOp,
            arith::MulIOp, arith::AddIOp, triton::AddPtrOp>(user)) {
      for (Value res : user->getResults())
        collectLoadsFromAddress(res, loads, visited);
      continue;
    }
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      // Find which init_arg position this value feeds into.
      for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
        if (initArg == addr) {
          // The corresponding block argument inside the loop body.
          Value innerArg = forOp.getRegionIterArg(idx);
          collectLoadsFromAddress(innerArg, loads, visited);
        }
      }
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
      if (!forOp)
        continue;
      unsigned idx = use.getOperandNumber();
      if (idx < forOp.getNumResults()) {
        Value forResult = forOp.getResult(idx);
        collectLoadsFromAddress(forResult, loads, visited);
      }
      continue;
    }
  }
}

// Verify that all uses of the remsi result flow ONLY into address computations
// (expand_dims, muli, broadcast, addi, addptr, load) and NOT into stores,
// control flow, or other ops where incorrect values would be observable.
static bool allUsesAreAddressComputation(Value val,
                                         SmallPtrSetImpl<Operation *> &visited) {
  for (OpOperand &use : val.getUses()) {
    Operation *user = use.getOwner();
    if (!visited.insert(user).second)
      continue;

    if (isa<triton::ExpandDimsOp, triton::BroadcastOp, triton::SplatOp,
            arith::MulIOp, arith::AddIOp, triton::AddPtrOp>(user)) {
      for (Value res : user->getResults()) {
        if (!allUsesAreAddressComputation(res, visited))
          return false;
      }
      continue;
    }
    if (isa<triton::LoadOp>(user))
      continue;
    if (auto forOp = dyn_cast<scf::ForOp>(user)) {
      for (auto [idx, initArg] : llvm::enumerate(forOp.getInitArgs())) {
        if (initArg != val)
          continue;
        Value innerArg = forOp.getRegionIterArg(idx);
        if (!allUsesAreAddressComputation(innerArg, visited))
          return false;
      }
      continue;
    }
    if (auto yieldOp = dyn_cast<scf::YieldOp>(user)) {
      auto forOp = dyn_cast<scf::ForOp>(yieldOp->getParentOp());
      if (!forOp)
        return false;
      unsigned idx = use.getOperandNumber();
      if (idx < forOp.getNumResults()) {
        if (!allUsesAreAddressComputation(forOp.getResult(idx), visited))
          return false;
      }
      continue;
    }

    return false;
  }
  return true;
}

// Determine the expand_dims axis used on the remsi result. This tells us
// which dimension of the 2D load tensor corresponds to our 1D mask.
static int getExpandDimsAxis(Value remsiResult) {
  for (OpOperand &use : remsiResult.getUses()) {
    if (auto expd = dyn_cast<triton::ExpandDimsOp>(use.getOwner()))
      return expd.getAxis();
  }
  return -1;
}

// Match the decomposed modulo pattern subi(x, muli(divsi(x, d), d)), which is
// semantically equivalent to x % d by the signed-division identity
// a == (a / b) * b + a % b, hence a - (a / b) * b == a % b.
//
// On success, `dividend` is set to x, `divisor` to d, and the three ops
// (subi, muli, divsi) are appended to `deadOps` for later cleanup.
static bool matchDecomposedModulo(arith::SubIOp subi, Value &dividend,
                                  Value &divisor,
                                  SmallVectorImpl<Operation *> &deadOps) {
  Value x = subi.getLhs();
  auto muli = subi.getRhs().getDefiningOp<arith::MulIOp>();
  if (!muli)
    return false;

  // muli must be divsi(x, d) * d, in either operand order.
  arith::DivSIOp divsi;
  Value d;
  if (auto lhsDiv = muli.getLhs().getDefiningOp<arith::DivSIOp>()) {
    divsi = lhsDiv;
    d = muli.getRhs();
  } else if (auto rhsDiv = muli.getRhs().getDefiningOp<arith::DivSIOp>()) {
    divsi = rhsDiv;
    d = muli.getLhs();
  } else {
    return false;
  }

  // divsi must be x / d with the same x as subi's lhs and the same d.
  if (divsi.getLhs() != x)
    return false;
  if (divsi.getRhs() != d)
    return false;

  dividend = x;
  divisor = d;
  deadOps.push_back(subi.getOperation());
  deadOps.push_back(muli.getOperation());
  deadOps.push_back(divsi.getOperation());
  return true;
}

// A unified modulo candidate: either a direct arith.remsi or a decomposed
// subi(x, muli(divsi(x, d), d)). `deadOps` lists the ops that become dead once
// `result` is replaced by the linear `dividend`.
struct ModuloCandidate {
  Value dividend;
  Value divisor;
  Value result;
  SmallVector<Operation *, 3> deadOps;
};

}  // namespace

void rewriteConvertModuloToMask(ModuleOp moduleOp) {
  SmallVector<ModuloCandidate> candidates;

  // Scan 1: direct arith.remsi ops.
  moduleOp.walk([&](arith::RemSIOp remsi) {
    candidates.push_back({remsi.getLhs(), remsi.getRhs(), remsi.getResult(),
                          {remsi.getOperation()}});
  });

  // Scan 2: decomposed modulo pattern subi(x, muli(divsi(x, d), d)).
  moduleOp.walk([&](arith::SubIOp subi) {
    Value dividend, divisor;
    SmallVector<Operation *, 3> deadOps;
    if (matchDecomposedModulo(subi, dividend, divisor, deadOps))
      candidates.push_back(
          {dividend, divisor, subi.getResult(), std::move(deadOps)});
  });

  for (auto &cand : candidates) {
    Value dividend = cand.dividend;
    Value divisor = cand.divisor;
    Value result = cand.result;

    // Condition 1: must be a 1-D i32 tensor.
    auto tensorTy = dyn_cast<RankedTensorType>(result.getType());
    if (!tensorTy || tensorTy.getRank() != 1)
      continue;
    if (!tensorTy.getElementType().isInteger(32))
      continue;

    // Condition 2: dividend must contain a make_range (linear tile offset).
    int64_t tileSize = getMakeRangeSize(dividend);
    if (tileSize <= 0)
      continue;

    // Condition 3: divisor must be splat of a scalar.
    Value boundScalar = getScalarThroughSplat(divisor);
    if (!boundScalar)
      continue;

    // Condition 4: the un-modulo'd dividend must have a cmpi slt guard against
    // the same bound that feeds into a store mask.
    if (!hasStoreMaskGuard(dividend, boundScalar))
      continue;

    // Condition 5: all uses of the remsi result flow only into address
    // computation (load addresses via addptr chains).
    SmallPtrSet<Operation *, 32> visited;
    if (!allUsesAreAddressComputation(result, visited))
      continue;

    // Condition 6: determine the expand_dims axis (needed for mask shaping).
    int axis = getExpandDimsAxis(result);
    if (axis < 0)
      continue;

    // Check divisibility to decide strategy.
    int64_t div = getDivisibility(boundScalar);
    bool needMaskInjection = (div < tileSize);

    if (needMaskInjection) {
      // Strategy B: remove modulo AND inject boundary mask into downstream
      // loads. This handles the general case where the bound might not be a
      // multiple of the tile size.

      SmallVector<triton::LoadOp> affectedLoads;
      SmallPtrSet<Value, 32> visitedAddrs;
      collectLoadsFromAddress(result, affectedLoads, visitedAddrs);

      if (affectedLoads.empty())
        continue;

      // Safety precheck: all affected loads must be 2-D tensor loads.
      // Loads without a mask are accepted — we will inject a fresh mask
      // with a zero "other" value so that OOB positions contribute zero
      // to downstream dot products. The store-mask guard (Condition 4)
      // guarantees those zero contributions are never written out.
      bool allLoadsSafe = true;
      for (auto load : affectedLoads) {
        auto loadTy = dyn_cast<RankedTensorType>(load.getResult().getType());
        if (!loadTy || loadTy.getRank() != 2) {
          allLoadsSafe = false;
          break;
        }
      }
      if (!allLoadsSafe)
        continue;

      IRRewriter rw(moduleOp.getContext());
      Location defLoc = result.getLoc();
      rw.setInsertionPointAfter(result.getDefiningOp());
      Value boundaryMask = rw.create<arith::CmpIOp>(
          defLoc, arith::CmpIPredicate::slt, dividend, divisor);

      // Clamp the address index to [0, divisor - 1] instead of using the raw
      // linear dividend. Dropping the modulo removes the address wrap that kept
      // every access inside the valid buffer; a plain mask only zeroes the
      // loaded VALUES, it does not stop a coalesced block DMA from physically
      // reading out-of-bounds columns. Clamping keeps every generated address
      // in-bounds (OOB lanes read the last valid element, whose value is then
      // discarded by the injected mask/other), so block DMA stays safe.
      auto dividendTy = cast<RankedTensorType>(dividend.getType());
      Attribute oneAttr = rw.getI32IntegerAttr(1);
      auto denseOne = DenseElementsAttr::get(dividendTy, oneAttr);
      Value one = rw.create<arith::ConstantOp>(defLoc, denseOne);
      Value upperBound = rw.create<arith::SubIOp>(defLoc, divisor, one);
      Value clampedIndex =
          rw.create<arith::MinSIOp>(defLoc, dividend, upperBound);

      for (auto load : affectedLoads) {
        auto loadTy = dyn_cast<RankedTensorType>(load.getResult().getType());
        if (!loadTy || loadTy.getRank() != 2)
          continue;

        rw.setInsertionPoint(load);
        Location loc = load.getLoc();

        Value expandedMask = rw.create<triton::ExpandDimsOp>(
            loc, boundaryMask, axis);

        auto fullMaskTy = RankedTensorType::get(
            loadTy.getShape(), rw.getI1Type());
        Value broadcastedMask = rw.create<triton::BroadcastOp>(
            loc, fullMaskTy, expandedMask);

        if (Value existingMask = load.getMask()) {
          Value combinedMask = rw.create<arith::AndIOp>(
              loc, existingMask, broadcastedMask);
          load.getMaskMutable().assign(combinedMask);
        } else {
          load.getMaskMutable().assign(broadcastedMask);
          auto elemTy = loadTy.getElementType();
          Attribute zeroAttr;
          if (isa<FloatType>(elemTy))
            zeroAttr = FloatAttr::get(elemTy, 0.0);
          else
            zeroAttr = IntegerAttr::get(elemTy, 0);
          auto denseZero = DenseElementsAttr::get(loadTy, zeroAttr);
          Value otherVal = rw.create<arith::ConstantOp>(loc, denseZero);
          load.getOtherMutable().assign(otherVal);
        }
      }

      result.replaceAllUsesWith(clampedIndex);
      for (auto *op : cand.deadOps) {
        if (op->use_empty())
          rw.eraseOp(op);
      }

      LLVM_DEBUG({
        llvm::dbgs() << "ConvertModuloToMask: replaced modulo with clamped "
                        "index + mask injection ("
                     << affectedLoads.size() << " loads)\n";
      });
    } else {
      // Strategy A: divisibility >= tileSize, no boundary tiles possible.
      // Simply remove the modulo — all accesses are guaranteed in-bounds.
      LLVM_DEBUG({
        llvm::dbgs() << "ConvertModuloToMask: removing modulo (aligned)\n";
      });
      result.replaceAllUsesWith(dividend);
      for (auto *op : cand.deadOps) {
        if (op->use_empty())
          op->erase();
      }
    }
  }
}

}  // namespace ConvertModuloToMask

namespace mlir {
namespace triton {

struct ConvertModuloToMaskPass
    : public PassWrapper<ConvertModuloToMaskPass, OperationPass<ModuleOp>> {
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(ConvertModuloToMaskPass)

  StringRef getArgument() const override {
    return "convert-modulo-to-mask";
  }
  StringRef getDescription() const override {
    return "Replace modulo (arith.remsi or decomposed divsi+muli+subi) in "
           "tile offset address computation with load-mask injection to "
           "enable block DMA";
  }
  void runOnOperation() override {
    ConvertModuloToMask::rewriteConvertModuloToMask(getOperation());
  }
};

std::unique_ptr<OperationPass<ModuleOp>> createConvertModuloToMaskPass() {
  return std::make_unique<ConvertModuloToMaskPass>();
}

}  // namespace triton
}  // namespace mlir
