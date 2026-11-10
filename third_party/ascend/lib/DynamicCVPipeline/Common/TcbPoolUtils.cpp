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

#include "DynamicCVPipeline/Common/TcbPoolUtils.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Interfaces/LoopLikeInterface.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <map>
#include <set>

namespace mlir {
namespace CVPipeline {

// Declarations live outside the scheduling ifs. Follow aliases to identify
// the one ssbuffer.if that owns each UB TCB's uses in this scope.
static Operation *getTcbSchedulingIf(Value buffer, scope::ScopeOp scopeOp) {
  SmallVector<Value> pending{buffer};
  llvm::SmallPtrSet<Operation *, 8> visited;
  Operation *owner = nullptr;
  while (!pending.empty()) {
    Value value = pending.pop_back_val();
    for (Operation *user : value.getUsers()) {
      if (isa<annotation::MarkOp>(user) || !visited.insert(user).second)
        continue;
      if (user->getParentOfType<scope::ScopeOp>() != scopeOp)
        return nullptr;
      Operation *block = user;
      while (block != scopeOp.getOperation() &&
             !(isa<scf::IfOp>(block) && block->hasAttr(CVPipeline::kIf)))
        block = block->getParentOp();
      if (block != scopeOp.getOperation()) {
        if (owner && owner != block)
          return nullptr;
        owner = block;
        continue;
      }
      if (!isa<ViewLikeOpInterface, memref::MemorySpaceCastOp,
               bufferization::ToTensorOp>(user))
        return nullptr;
      llvm::append_range(pending, user->getResults());
    }
  }
  return owner;
}

// The downstream planner handles exactly one Cube/Vector pair with two
// matching ping-pong UB queues. CBUF TCBs are independent of this fallback.
bool canEnableDynamicPlan(ModuleOp module) {
  std::set<std::set<int>> queues[2];
  bool seen[2] = {false, false};
  auto result = module.walk([&](scope::ScopeOp scopeOp) -> WalkResult {
    auto core =
        scopeOp->getAttrOfType<hivm::TCoreTypeAttr>(hivm::TCoreTypeAttr::name);
    if (!core)
      return WalkResult::advance();
    unsigned side;
    if (core.getTcoretype() == hivm::TCoreType::CUBE)
      side = 0;
    else if (core.getTcoretype() == hivm::TCoreType::VECTOR)
      side = 1;
    else
      return WalkResult::interrupt();
    if (seen[side])
      return WalkResult::interrupt();
    seen[side] = true;
    std::map<Operation *, std::set<int>> groups;
    std::set<int> ids;
    auto members = scopeOp.walk([&](memref::AllocOp alloc) -> WalkResult {
      auto space = dyn_cast_or_null<hivm::AddressSpaceAttr>(
          alloc.getType().getMemorySpace());
      if (!space || space.getAddressSpace() != hivm::AddressSpace::UB)
        return WalkResult::advance();
      auto id = CVPipeline::getTightlyCoupledBufferId(alloc.getResult());
      if (!id)
        return WalkResult::advance();
      Operation *owner = getTcbSchedulingIf(alloc.getResult(), scopeOp);
      if (!owner || !ids.insert(*id).second)
        return WalkResult::interrupt();
      groups[owner].insert(*id);
      return WalkResult::advance();
    });
    if (members.wasInterrupted() || ids.size() != 4 || groups.size() != 2)
      return WalkResult::interrupt();
    Operation *schedulingLoop = nullptr;
    for (const auto &group : groups) {
      if (group.second.size() != 2)
        return WalkResult::interrupt();
      // Both queues must belong to the same nearest scheduling loop.
      auto loop = group.first->getParentOfType<LoopLikeOpInterface>();
      if (!loop || (schedulingLoop && schedulingLoop != loop.getOperation()))
        return WalkResult::interrupt();
      schedulingLoop = loop.getOperation();
      queues[side].insert(group.second);
    }
    return WalkResult::advance();
  });
  return !result.wasInterrupted() && seen[0] && seen[1] &&
         queues[0] == queues[1];
}

} // namespace CVPipeline
} // namespace mlir
