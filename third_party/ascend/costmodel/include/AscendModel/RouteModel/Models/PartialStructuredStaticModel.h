#ifndef ASCENDMODEL_PARTIALSTRUCTUREDSTATICMODEL_H
#define ASCENDMODEL_PARTIALSTRUCTUREDSTATICMODEL_H
#include "AscendModel/RouteModel/Models/PartialStructuredSimtLoadModel.h"
namespace mlir::ascend::partial_memory {
// Complete static rectangle and known inter-program coefficient. Coordinates
// of loaded indices remain unknown. SIMD accepts F1 only.
Features extractStaticFeatures(const FactorLoadGeometry &, Model);
// SYS_CNT ticks: SIMT is a full F-program increment; SIMD is a native fragment.
// No sampling, q, U/V, kernel-name or timing inputs. Nullopt retains fallback.
std::optional<double> predictStatic(const FactorLoadGeometry &, Model);
} // namespace mlir::ascend::partial_memory
#endif
