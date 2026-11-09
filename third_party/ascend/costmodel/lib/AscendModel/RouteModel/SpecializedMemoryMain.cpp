// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "AscendModel/RouteModel/SpecializedMemoryMain.h"
#include "AscendModel/RouteModel/OrdinaryStoreRound4MainAnalysis.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <functional>

using namespace mlir;
using namespace mlir::ascend;
using namespace mlir::ascend::ordinary_store_main;

namespace {
// In addition to axis slopes, prove a zero origin. The generic axis summary
// deliberately omits uniform offsets; those must not silently alter a full
// fragment's contiguous source/sink or index-table boundary.
std::optional<int64_t> zeroOriginImpl(Value v, unsigned depth,
                                      unsigned &remaining) {
  if (depth > 48 || !remaining)
    return std::nullopt;
  --remaining;
  if (auto c = constant(v))
    return c;
  if (isa<BlockArgument>(v) && isa<triton::PointerType>(v.getType()))
    return 0;
  auto *op = v.getDefiningOp();
  if (!op)
    return std::nullopt;
  auto n = op->getName().getStringRef();
  if (n == "tt.load")
    return 0; // A symbolic index at zero, never an actual tensor read.
  if (n == "tt.make_range") {
    auto start = op->getAttrOfType<IntegerAttr>("start");
    return start ? std::optional<int64_t>(start.getInt()) : std::nullopt;
  }
  if (op->getNumOperands() == 1 && (n == "tt.splat" || n == "tt.expand_dims" ||
                                    n == "tt.broadcast" || n == "arith.extsi"))
    return zeroOriginImpl(op->getOperand(0), depth + 1, remaining);
  if (op->getNumOperands() != 2)
    return std::nullopt;
  auto a = zeroOriginImpl(op->getOperand(0), depth + 1, remaining);
  auto b = zeroOriginImpl(op->getOperand(1), depth + 1, remaining);
  int64_t result;
  if (!a || !b)
    return std::nullopt;
  if ((n == "tt.addptr" || n == "arith.addi") &&
      !__builtin_add_overflow(*a, *b, &result))
    return result;
  if (n == "arith.muli" && !__builtin_mul_overflow(*a, *b, &result))
    return result;
  return std::nullopt;
}
std::optional<int64_t> zeroOrigin(Value v) {
  unsigned remaining = 2048;
  return zeroOriginImpl(v, 0, remaining);
}

// i64 index loads are admitted only by this complete-envelope proof. The
// generic target-increment model continues to admit its original i32 domain.
SmallVector<Axis> directAxesImpl(Value v, unsigned depth, unsigned &remaining) {
  auto result = unknown(v);
  if (depth > 48 || !remaining)
    return result;
  --remaining;
  auto *op = v.getDefiningOp();
  if (!op)
    return result;
  if (auto load = dyn_cast<triton::LoadOp>(op)) {
    auto t = dyn_cast<RankedTensorType>(v.getType());
    if (t && t.getRank() == 1 &&
        (t.getElementType().isInteger(32) || t.getElementType().isInteger(64)))
      return {Axis{0, 1, v, true}};
    return result;
  }
  if (auto shaped = propagateAxisShape(
          v, result, Axis{}, Axis{1, 0, {}, true}, Axis{}, [&](Value source) {
            return directAxesImpl(source, depth + 1, remaining);
          }))
    return *shaped;
  auto n = op->getName().getStringRef();
  if (op->getNumOperands() != 2 ||
      (n != "tt.addptr" && n != "arith.addi" && n != "arith.muli"))
    return result;
  auto a = directAxesImpl(op->getOperand(0), depth + 1, remaining);
  auto b = directAxesImpl(op->getOperand(1), depth + 1, remaining);
  if (a.size() != result.size() || b.size() != result.size())
    return result;
  auto ac = constant(op->getOperand(0)), bc = constant(op->getOperand(1));
  for (size_t i = 0; i < result.size(); ++i) {
    if (!a[i].valid || !b[i].valid)
      continue;
    Axis x;
    if (n == "arith.muli") {
      if (!ac && !bc)
        continue;
      const auto &source = ac ? b[i] : a[i];
      int64_t scale = ac ? *ac : *bc;
      if (__builtin_mul_overflow(source.stride, scale, &x.stride) ||
          __builtin_mul_overflow(source.loadedScale, scale, &x.loadedScale))
        continue;
      x.loaded = source.loaded;
    } else {
      if (a[i].loaded && b[i].loaded && a[i].loaded != b[i].loaded)
        continue;
      if (__builtin_add_overflow(a[i].stride, b[i].stride, &x.stride) ||
          __builtin_add_overflow(a[i].loadedScale, b[i].loadedScale,
                                 &x.loadedScale))
        continue;
      x.loaded = a[i].loaded ? a[i].loaded : b[i].loaded;
    }
    if (!(x.loaded && x.stride))
      result[i] = x;
  }
  return result;
}
SmallVector<Axis> directAxes(Value v) {
  unsigned remaining = 2048;
  return directAxesImpl(v, 0, remaining);
}
} // namespace

std::optional<SpecializedMemoryGeometry>
mlir::ascend::proveSpecializedMemoryFragment(ModuleOp module) {
  SmallVector<triton::FuncOp> functions;
  for (auto fn : module.getOps<triton::FuncOp>())
    functions.push_back(fn);
  if (functions.size() != 1)
    return std::nullopt;
  auto fn = functions.front();
  if (fn.getNumArguments() != 3 || fn.getBody().empty() ||
      !llvm::hasSingleElement(fn.getBody()))
    return std::nullopt;
  for (Value arg : fn.getArguments())
    if (!isa<triton::PointerType>(arg.getType()))
      return std::nullopt;
  SmallVector<triton::LoadOp> loads;
  triton::StoreOp store;
  unsigned count = 0;
  for (Operation &op : fn.getBody().front()) {
    if (++count > 256 || op.getNumRegions())
      return std::nullopt;
    auto n = op.getName().getStringRef();
    if (auto load = dyn_cast<triton::LoadOp>(op)) {
      if (load.getMask() || load.getOther() || load.getIsVolatile() ||
          load.getCache() != triton::CacheModifier::NONE ||
          load.getEvict() != triton::EvictionPolicy::NORMAL ||
          !load.getBoundaryCheck().empty())
        return std::nullopt;
      loads.push_back(load);
    } else if (auto s = dyn_cast<triton::StoreOp>(op)) {
      if (store || s.getMask() || s.getCache() != triton::CacheModifier::NONE ||
          s.getEvict() != triton::EvictionPolicy::NORMAL ||
          !s.getBoundaryCheck().empty())
        return std::nullopt;
      store = s;
    } else if (n != "arith.constant" && n != "arith.addi" &&
               n != "arith.muli" && n != "arith.extsi" &&
               n != "tt.make_range" && n != "tt.splat" &&
               n != "tt.expand_dims" && n != "tt.broadcast" &&
               n != "tt.addptr" && n != "tt.return")
      return std::nullopt;
  }
  if (loads.size() != 2 || !store)
    return std::nullopt;
  auto payload = store.getValue().getDefiningOp<triton::LoadOp>();
  auto t = dyn_cast<RankedTensorType>(store.getValue().getType());
  if (!payload || !t || !t.hasStaticShape() || t.getRank() != 2 ||
      !t.getElementType().isF32() || t.getDimSize(0) < 2 ||
      t.getDimSize(1) < 2 || t.getDimSize(0) > 4096 || t.getDimSize(1) > 4096 ||
      t.getDimSize(0) > 8192 / t.getDimSize(1))
    return std::nullopt;
  auto index = loads[0] == payload ? loads[1] : loads[0];
  auto it = dyn_cast<RankedTensorType>(index.getType());
  if (!it || it.getRank() != 1 || it.getDimSize(0) != t.getDimSize(0) ||
      (!it.getElementType().isInteger(32) &&
       !it.getElementType().isInteger(64)))
    return std::nullopt;
  // Every entry pointer must have the exact logical pointee type, with three
  // distinct SSA roots; pointer reinterpretations and hidden source/sink ops
  // do not acquire an envelope certificate.
  SmallVector<Value> roots;
  for (auto pair : {std::make_pair(index.getPtr(), it.getElementType()),
                    std::make_pair(payload.getPtr(), t.getElementType()),
                    std::make_pair(store.getPtr(), t.getElementType())}) {
    auto root = dyn_cast_or_null<BlockArgument>(pointerRoot(pair.first));
    auto ptr = root ? dyn_cast<triton::PointerType>(root.getType()) : nullptr;
    if (!root || root.getOwner() != &fn.getBody().front() || !ptr ||
        ptr.getPointeeType() != pair.second ||
        llvm::is_contained(roots, root) ||
        zeroOrigin(pair.first) != std::optional<int64_t>(0))
      return std::nullopt;
    roots.push_back(root);
  }
  const int64_t columns = t.getDimSize(1);
  auto ia = directAxes(index.getPtr());
  if (ia.size() != 1 || !ia[0].valid || ia[0].loaded || ia[0].stride != 1)
    return std::nullopt;
  auto kind = [&](Value ptr) {
    auto a = directAxes(ptr);
    if (a.size() != 2 || !a[0].valid || !a[1].valid || a[1].loaded ||
        a[1].stride != 1)
      return 0;
    if (!a[0].loaded && a[0].stride == columns)
      return 1; // Contiguous source/sink.
    if (a[0].loaded == index.getResult() && !a[0].stride &&
        a[0].loadedScale == columns)
      return 2; // DS target using precisely this rank1 index load.
    return 0;
  };
  int source = kind(payload.getPtr()), sink = kind(store.getPtr());
  if (!((source == 1 && sink == 2) || (source == 2 && sink == 1)))
    return std::nullopt;
  // Prove the entire body belongs to this store's backward slice. An extra
  // unused computation, load, return value or side effect is not priced away.
  llvm::SmallPtrSet<Operation *, 32> owned;
  std::function<void(Operation *)> visit = [&](Operation *op) {
    if (!op || !owned.insert(op).second)
      return;
    for (Value v : op->getOperands())
      visit(v.getDefiningOp());
  };
  visit(store.getOperation());
  for (Operation &op : fn.getBody().front()) {
    if (op.getName().getStringRef() == "tt.return") {
      if (op.getNumOperands())
        return std::nullopt;
    } else if (!owned.contains(&op))
      return std::nullopt;
  }
  SpecializedMemoryGeometry g;
  g.shape = {t.getDimSize(0), columns};
  g.pattern = "DS";
  g.regularStep = 1;
  g.regularAxisCoefficients = {std::nullopt, 1};
  g.discreteAxisCoefficients = {columns, std::nullopt};
  g.operation = source == 2 ? SpecializedMemoryOperation::Load
                            : SpecializedMemoryOperation::Store;
  g.completeFragmentBoundary = g.directThreePointerEnvelope = true;
  return g;
}
