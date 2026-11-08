// Finite main70f Host/device address proof. No tensor contents are inspected.
#ifndef ORDINARY_STORE_ROUND4_VISIBILITY_ANALYSIS_H
#define ORDINARY_STORE_ROUND4_VISIBILITY_ANALYSIS_H
#include "OrdinaryStoreRound4MainAnalysis.h"
#include "llvm/Support/JSON.h"
#include <string>

namespace mlir::ascend::ordinary_store_main {
enum class PitchKind { Invalid, Constant, HostArgument, DeviceScalarLoad };
struct Pitch {
  PitchKind kind = PitchKind::Invalid;
  int64_t scale = 1;
  Value source;
  unsigned argument = 0;
};
struct VisibilityAxis {
  bool valid = false;
  bool coordinate = false;
  Value loadedIndex; // Empty means tt.make_range, not a loaded D coordinate.
  Pitch pitch{PitchKind::Constant, 1, {}, 0};
};

// A scalar device coefficient must come from an exact entry pointer plus a
// bounded constant offset. Program/lane-dependent or tensor pointers fail.
inline bool scalarDevicePointer(Value value, triton::FuncOp entry,
                                unsigned depth = 0) {
  if (depth > 24 || !isa<triton::PointerType>(value.getType()))
    return false;
  if (auto a = dyn_cast<BlockArgument>(value))
    return a.getOwner() == &entry.getBody().front() &&
           a.getArgNumber() < entry.getNumArguments();
  auto *op = value.getDefiningOp();
  if (!op || op->getName().getStringRef() != "tt.addptr" ||
      op->getNumOperands() != 2)
    return false;
  auto offset = constant(op->getOperand(1));
  return offset && *offset >= 0 && *offset <= 4096 &&
         scalarDevicePointer(op->getOperand(0), entry, depth + 1);
}

inline Pitch uniformPitch(Value value, triton::FuncOp entry,
                          unsigned depth = 0) {
  if (depth > 32)
    return {};
  if (auto c = constant(value))
    return Pitch{PitchKind::Constant, *c, {}, 0};
  if (auto a = dyn_cast<BlockArgument>(value)) {
    auto type = dyn_cast<IntegerType>(a.getType());
    if (a.getOwner() == &entry.getBody().front() && type &&
        (type.getWidth() == 32 || type.getWidth() == 64))
      return Pitch{PitchKind::HostArgument, 1, value, a.getArgNumber()};
    return {};
  }
  auto *op = value.getDefiningOp();
  if (!op)
    return {};
  const auto name = op->getName().getStringRef();
  if ((name == "tt.splat" || name == "tt.broadcast" ||
       name == "tt.expand_dims") &&
      op->getNumOperands() == 1)
    return uniformPitch(op->getOperand(0), entry, depth + 1);
  if (name == "arith.extsi" && op->getNumOperands() == 1) {
    Type source = op->getOperand(0).getType(), target = value.getType();
    if (auto t = dyn_cast<RankedTensorType>(source))
      source = t.getElementType();
    if (auto t = dyn_cast<RankedTensorType>(target))
      target = t.getElementType();
    auto s = dyn_cast<IntegerType>(source), t = dyn_cast<IntegerType>(target);
    if (!s || !t || s.getWidth() < 32 || t.getWidth() > 64 ||
        s.getWidth() >= t.getWidth())
      return {};
    return uniformPitch(op->getOperand(0), entry, depth + 1);
  }
  if (name == "tt.load" && op->getNumResults() == 1 &&
      !isa<RankedTensorType>(value.getType())) {
    auto type = dyn_cast<IntegerType>(value.getType());
    auto load = dyn_cast<triton::LoadOp>(op);
    if (!type || (type.getWidth() != 32 && type.getWidth() != 64) || !load ||
        load.getMask() || !scalarDevicePointer(load.getPtr(), entry) ||
        load.getCache() != triton::CacheModifier::NONE ||
        load.getEvict() != triton::EvictionPolicy::NORMAL)
      return {};
    if (auto a = op->getAttrOfType<BoolAttr>("isVolatile"))
      if (a.getValue())
        return {};
    return Pitch{PitchKind::DeviceScalarLoad, 1, value, 0};
  }
  // Only a positive constant times one direct scalar is admitted. Two dynamic
  // scalars, sums, casts that truncate/zero-extend, and tensor loads fail
  // closed.
  if (name == "arith.muli" && op->getNumOperands() == 2) {
    auto a = uniformPitch(op->getOperand(0), entry, depth + 1);
    auto b = uniformPitch(op->getOperand(1), entry, depth + 1);
    if (a.kind == PitchKind::Invalid || b.kind == PitchKind::Invalid ||
        (a.kind != PitchKind::Constant && b.kind != PitchKind::Constant))
      return {};
    auto result = a.kind == PitchKind::Constant ? b : a;
    const auto factor = a.kind == PitchKind::Constant ? a.scale : b.scale;
    if (factor <= 0 ||
        __builtin_mul_overflow(result.scale, factor, &result.scale))
      return {};
    return result;
  }
  return {};
}

inline SmallVector<VisibilityAxis> visibilityUnknown(Value value) {
  auto type = dyn_cast<RankedTensorType>(value.getType());
  return type ? SmallVector<VisibilityAxis>(type.getRank())
              : SmallVector<VisibilityAxis>{};
}
inline SmallVector<VisibilityAxis> visibilityAxesImpl(Value value,
                                                      triton::FuncOp entry,
                                                      unsigned depth,
                                                      unsigned &remaining) {
  auto result = visibilityUnknown(value);
  if (depth > 48 || !remaining)
    return result;
  --remaining;
  auto *op = value.getDefiningOp();
  if (!op)
    return result;
  const auto name = op->getName().getStringRef();
  const size_t rank = result.size();
  VisibilityAxis zero, range, loaded;
  zero.valid = true;
  range = zero;
  range.coordinate = true;
  loaded = range;
  loaded.loadedIndex = value;
  if (auto shaped = propagateAxisShape(
          value, result, zero, range, loaded, [&](Value source) {
            return visibilityAxesImpl(source, entry, depth + 1, remaining);
          }))
    return *shaped;
  if (op->getNumOperands() != 2 ||
      (name != "tt.addptr" && name != "arith.addi" && name != "arith.muli"))
    return result;
  // For multiplication one operand must be a proved uniform coefficient.
  // A loaded D coordinate times a scalar device load is intentionally allowed;
  // it is not the prohibited product of two axis-bearing coordinates.
  if (name == "arith.muli") {
    auto a = uniformPitch(op->getOperand(0), entry);
    auto b = uniformPitch(op->getOperand(1), entry);
    if (a.kind == PitchKind::Invalid && b.kind == PitchKind::Invalid)
      return result;
    const bool leftPitch = a.kind != PitchKind::Invalid;
    const auto pitch = leftPitch ? a : b;
    auto source = visibilityAxesImpl(op->getOperand(leftPitch ? 1 : 0), entry,
                                     depth + 1, remaining);
    if (source.size() != rank || pitch.scale <= 0)
      return result;
    for (size_t i = 0; i < rank; ++i) {
      auto axis = source[i];
      if (!axis.valid)
        continue;
      if (!axis.coordinate) {
        result[i].valid = true;
        continue;
      }
      // An axis may carry one dynamic coefficient, never a product of two.
      if (axis.pitch.kind != PitchKind::Constant &&
          pitch.kind != PitchKind::Constant)
        continue;
      Pitch combined =
          axis.pitch.kind == PitchKind::Constant ? pitch : axis.pitch;
      const int64_t scale = axis.pitch.kind == PitchKind::Constant
                                ? axis.pitch.scale
                                : pitch.scale;
      if (__builtin_mul_overflow(combined.scale, scale, &combined.scale) ||
          combined.scale <= 0)
        continue;
      axis.pitch = combined;
      result[i] = axis;
    }
    return result;
  }
  auto a = visibilityAxesImpl(op->getOperand(0), entry, depth + 1, remaining);
  auto b = visibilityAxesImpl(op->getOperand(1), entry, depth + 1, remaining);
  if (a.size() != rank || b.size() != rank)
    return result;
  for (size_t i = 0; i < rank; ++i) {
    if (!a[i].valid || !b[i].valid || (a[i].coordinate && b[i].coordinate))
      continue;
    result[i] = a[i].coordinate ? a[i] : b[i];
  }
  return result;
}
inline SmallVector<VisibilityAxis> visibilityAxes(Value pointer,
                                                  triton::FuncOp entry) {
  unsigned remaining = 2048;
  return visibilityAxesImpl(pointer, entry, 0, remaining);
}

// The existing public compile caller attaches these facts on actual entry args.
// Require the real IR attr and the exact option record together. A
// free-standing vector, arg name, source-case identifier or device value is
// never consulted.
inline std::optional<int64_t>
hostPitchValue(const Pitch &pitch, triton::FuncOp entry,
               const llvm::json::Object &capability) {
  if (pitch.kind == PitchKind::Constant)
    return pitch.scale;
  if (pitch.kind != PitchKind::HostArgument ||
      pitch.argument >= entry.getNumArguments() ||
      pitch.source != entry.getArgument(pitch.argument))
    return std::nullopt;
  auto attr = entry.getArgAttrOfType<IntegerAttr>(
      pitch.argument, "ascend.costmodel.runtime_value");
  auto type = dyn_cast<IntegerType>(pitch.source.getType());
  const auto *facts = capability.getArray("runtime_scalar_values");
  if (!attr || attr.getValue().getBitWidth() != 64 || !type || !facts ||
      facts->size() > entry.getNumArguments())
    return std::nullopt;
  std::optional<int64_t> exact;
  SmallVector<int64_t> seen;
  for (const auto &value : *facts) {
    auto *item = value.getAsObject();
    if (!item)
      return std::nullopt;
    auto index = item->getInteger("argument_index"),
         scalar = item->getInteger("value");
    if (!index || !scalar || *index < 0 ||
        static_cast<uint64_t>(*index) >= entry.getNumArguments() ||
        llvm::is_contained(seen, *index))
      return std::nullopt;
    seen.push_back(*index);
    if (static_cast<uint64_t>(*index) == pitch.argument)
      exact = *scalar;
  }
  if (!exact || *exact != attr.getInt() || *exact <= 0 ||
      (type.getWidth() < 64 &&
       *exact > (int64_t(1) << (type.getWidth() - 1)) - 1))
    return std::nullopt;
  int64_t result;
  if (__builtin_mul_overflow(*exact, pitch.scale, &result) || result <= 0)
    return std::nullopt;
  return result;
}
} // namespace mlir::ascend::ordinary_store_main
#endif
