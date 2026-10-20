//===- PartialStructuredSimtLoadModel.h -----------------------*- C++ -*-===//
#ifndef ASCENDMODEL_PARTIALSTRUCTUREDSIMTLOADMODEL_H
#define ASCENDMODEL_PARTIALSTRUCTUREDSIMTLOADMODEL_H

#include "AscendModel/RouteModel/Models/PartialStructuredMemoryModel.h"

namespace mlir::ascend::partial_memory {

struct FactorLoadGeometry {
  Geometry geometry;
  int64_t factor = 1;
  // Known additive program_id coefficient, in element addresses. Never infer
  // this from the allocation size or from loaded index values.
  int64_t programStride = 0;
  std::vector<int64_t> lower, upper;
};

// Same TTIR-only reference descriptors as the frozen 112-coefficient model.
// These are synthetic geometry, not runtime transactions or cache hit rates.
// Internal static feature algebra. Caller must validate the geometry domain.
Features extractFactorReferenceFeatures(const FactorLoadGeometry &geometry);
Features extractFactorLoadFeatures(const FactorLoadGeometry &geometry);
// Complete physical F-program A-B increment, in SYS_CNT ticks. Callers must
// not apply F again. Returns nullopt outside the measured representative
// domain.
std::optional<double> predictFactorLoad(const FactorLoadGeometry &geometry);

} // namespace mlir::ascend::partial_memory
#endif
