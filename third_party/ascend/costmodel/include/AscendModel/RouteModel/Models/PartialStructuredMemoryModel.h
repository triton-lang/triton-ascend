//===- PartialStructuredMemoryModel.h -------------------------*- C++ -*-===//
#ifndef ASCENDMODEL_PARTIALSTRUCTUREDMEMORYMODEL_H
#define ASCENDMODEL_PARTIALSTRUCTUREDMEMORYMODEL_H

#include <cstdint>
#include <optional>
#include <string>
#include <unordered_map>
#include <vector>

namespace mlir::ascend::partial_memory {

enum class Model { SimtLoad, SimtStore, SimdLoad, SimdStore };

struct Axis {
  int64_t extent = 0;
  // Element-address stride for S, coefficient of one opaque index for D.
  int64_t pitch = 0;
  bool discrete = false;
};

struct Geometry {
  std::vector<Axis> axes;
  int64_t bits = 0;
  int64_t warps = 0;
  bool floatingPoint = false;
};

using Features = std::unordered_map<std::string, double>;
// Deterministic TTIR geometry only. No index samples or backend instructions.
Features extractFeatures(const Geometry &geometry);
// Nullopt preserves legacy pricing outside the calibrated q=1 domain.
std::optional<double> predict(const Geometry &geometry, Model model);

} // namespace mlir::ascend::partial_memory
#endif
