//===- NormalSegmentsStoreModel.h --------------------------------*- C++
//-*-===//
// Frozen normal-lowering Store price for the independently validated first
// cells. This header contains no research/legacy dispatcher and performs no
// unit conversion.
#ifndef ASCENDMODEL_NORMAL_SEGMENTS_STORE_MODEL_H
#define ASCENDMODEL_NORMAL_SEGMENTS_STORE_MODEL_H

#include <cmath>
#include <cstdint>
#include <optional>

namespace mlir::ascend::normal_segments_store {

enum class Payload { F32, I32, F16, BF16, U8, I64 };

struct Geometry {
  int64_t rows;
  int64_t columns;
  int64_t activeRows;
  int64_t activeColumns;
  int64_t innerStride;
  int64_t rowPitchElements;
};

struct Quote {
  double profileSystemCycles;
  const char *modelSHA256;
  const char *independentDEVSHA256;
};

// The reference unit belongs to the calibrated hardware profile, not to a
// runtime physical-card identifier or an unobserved device counter frequency.
inline constexpr double referenceSystemCounterMHz = 1000.0;

inline std::optional<Quote> evaluate(Payload payload, const Geometry &g,
                                     unsigned numWarps,
                                     unsigned superblockFactor,
                                     int64_t logicalProgramCount,
                                     double actualProfileSystemCounterMHz) {
  if ((payload != Payload::F32 && payload != Payload::I32 &&
       payload != Payload::F16 && payload != Payload::BF16 &&
       payload != Payload::U8 && payload != Payload::I64) ||
      numWarps != 4 || superblockFactor != 1 || logicalProgramCount != 1 ||
      !std::isfinite(actualProfileSystemCounterMHz) ||
      actualProfileSystemCounterMHz != referenceSystemCounterMHz ||
      g.rows != 16 || g.columns != 256 || g.innerStride != 1 ||
      g.rowPitchElements != 256 ||
      !((g.activeRows == 16 && g.activeColumns == 256) ||
        (g.activeRows == 12 && g.activeColumns == 192)))
    return std::nullopt;

  // Signless i8 storage uses the independently fitted and validated U8 model.
  if (payload == Payload::U8)
    return Quote{
        2541.5 + 2121.125 * g.activeRows +
            0.0 * (g.activeRows * g.activeColumns),
        "9f55ede0c214dc4681fecad537bd066f22b936259699aa491b77f62654efdda0",
        "c6d9da771081b205c81b4bce702eabf351a0c67c4e96aacfaa656b411720a2a0"};

  // Coefficients were frozen on TRAIN before the independent DEV measurements.
  // I32 uses the independently validated fixed-coefficient F32 migration.
  const double payloadBytes =
      payload == Payload::I64                                 ? 8.0
      : (payload == Payload::F16 || payload == Payload::BF16) ? 2.0
                                                              : 4.0;
  const double cycles =
      2709.0 + 2696.6875 * g.activeRows +
      0.11572265625 * (payloadBytes * g.activeRows * g.activeColumns);
  return Quote{
      cycles,
      "1fd563f915938ab285e21944304b6d9aaef24281b028b2cfbb50a3118ff0cb97",
      payload == Payload::I64
          ? "3092201b0adcb9c2d3d8283ee9626b5eb598ce62b738c2ae59640214bf91cd70"
      : payload == Payload::I32
          ? "ad837d7e950b63d902eaf0dc69ecf57677665953f50aaaf2fba65ee58b4cabd5"
      : payload == Payload::F16
          ? "0e81ac03b536e4f0dca066d84d111697a2a3fb488c66d62a968af494215d6a5f"
      : payload == Payload::BF16
          ? "f8ba07185a32902cabc7e298ec0a6d43f3fd51294bb28848250d223fdd306f82"
          : "8caf72244245c9f013884891d2208dba803fdb33bd55ee710ae36d8e45a43eb0"};
}
} // namespace mlir::ascend::normal_segments_store
#endif
