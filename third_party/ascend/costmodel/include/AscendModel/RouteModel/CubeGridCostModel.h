#ifndef ASCENDMODEL_ROUTEMODEL_CUBEGRIDCOSTMODEL_H
#define ASCENDMODEL_ROUTEMODEL_CUBEGRIDCOSTMODEL_H

#include "mlir/IR/Operation.h"
#include <cstdint>
#include <optional>
#include <string>

namespace mlir::ascend {

/// Frozen v9 parameters. Resource features are kcycles; the result is cycles.
struct CubeGridParameters {
  double inputBytesPerCycle = 202.25;
  double transactionStartup = 243.7737815366516;
  double cyclesPerQuantum = 0.826162107321598;
  int64_t coreCount = 28;
  double fillBase = 1216.524774849831;
  double fillInput = 1601.4886674916365;
  double fillMmad = 305.10768642386216;
  double fillFix = 1269.1947613953987;
  double occupancyBase = 117.41982321965838;
  double occupancyInput = 1.60300120268717;
  double occupancyFix = 3.6922483637846195;
  double serializedInput = 126.4043151942917;
  double serializedMmad = 14.114387736861522;
  double serializedFix = 105.37662470768385;
  double waveBase = 0;
  double waveInput = 205.18294116510623;
  double waveMmad = 0;
  double waveFix = 0;
  bool isValid() const;
};

struct CubeGridGeometry {
  int64_t m = 0;
  int64_t n = 0;
  int64_t k = 0;
  int64_t programCount = 0;
};

struct CubeGridCost {
  CubeGridGeometry geometry;
  /// Full kernel elapsed cycles, including all memory and launch costs.
  /// Do not add this to leaf costs or multiply it by the route wave count.
  double elapsedCycles = 0;
};

bool isCubeGridCalibrationDomain(const CubeGridGeometry &geometry);
double evaluateCubeGridCycles(const CubeGridGeometry &geometry,
                              const CubeGridParameters &parameters = {});

/// Recognize a whole direct GEMM; never match by symbol name. The caller must
/// separately check target/compiler/launch domain. Disjointness is a runtime
/// contract, not something distinct SSA pointer arguments establish.
std::optional<CubeGridCost>
matchCubeGridCost(Operation *root, int64_t programCount,
                  const CubeGridParameters &parameters,
                  bool disjointPointerArguments, std::string &rejectionReason);

} // namespace mlir::ascend
#endif
