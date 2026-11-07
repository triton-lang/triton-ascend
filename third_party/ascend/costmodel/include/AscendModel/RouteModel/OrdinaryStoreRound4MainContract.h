// Main-TA adapter contract. Pure C++17; no runtime/device or geometry inference.
#ifndef ORDINARY_STORE_ROUND4_MAIN_CONTRACT_H
#define ORDINARY_STORE_ROUND4_MAIN_CONTRACT_H
#include "OrdinaryStoreRound4.h"
#include <array>

namespace indirect_store_round4::main_ta {
struct CallerFacts {
  bool enabled = false;
  bool firstStore = false;
  bool allocationFresh = false;
  bool initializationGuardsOnly = false;
  bool exclusiveAlias = false;
  bool notReused = false;
  bool retainedUntilCompletion = false;
  bool entryAndIRBound = false;
  bool transformBound = false;
  bool addressIntegerDomainDeclared = false;
  std::string sourceSha256, launchId, allocationId;
  std::array<std::string, 5> lifecycleProofSha256;
  int64_t allocationEpoch = 0;
  std::string incrementContract, accountingBoundary, unitContract;
  std::string profileSha256, unitSourceSha256;
  double ticksPerSystemCycle = 0.0;
};
inline bool sha256(const std::string &s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
    return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
  });
}
inline std::string validateCaller(const CallerFacts &c,
                                  const std::string &actualProfile,
                                  const std::string &actualUnitSource) {
  if (!c.enabled) return "candidate_default_off";
  if (!c.firstStore || !c.allocationFresh || !c.initializationGuardsOnly ||
      !c.exclusiveAlias || !c.notReused || !c.retainedUntilCompletion)
    return "explicit_caller_first_store_facts_missing";
  if (!sha256(c.sourceSha256) || c.launchId.empty() || c.allocationId.empty() ||
      c.allocationEpoch <= 0 ||
      !std::all_of(c.lifecycleProofSha256.begin(), c.lifecycleProofSha256.end(), sha256))
    return "explicit_caller_lifecycle_identity_or_proof_missing";
  if (!c.entryAndIRBound) return "entry_or_transformed_IR_not_bound";
  if (!c.transformBound) return "unsupported_transformed_route_contract";
  if (!c.addressIntegerDomainDeclared)
    return "loaded_index_bounds_or_integer_no_wrap_caller_contract_missing";
  if (c.incrementContract != "round4_first_store_matched_increment_syscnt_v1" ||
      c.accountingBoundary != "round4_matched_target_only")
    return "matched_target_only_contract_missing";
  if (c.unitContract != "syscnt_ticks_to_active_profile_system_cycles_v1" ||
      !std::isfinite(c.ticksPerSystemCycle) || c.ticksPerSystemCycle != 1.0)
    return "active_SYS_CNT_profile_unit_contract_missing";
  if (!sha256(actualProfile) || !sha256(actualUnitSource) ||
      c.profileSha256 != actualProfile || c.unitSourceSha256 != actualUnitSource)
    return "caller_units_not_bound_to_actual_loaded_profile";
  return {};
}
// Auxiliary costs must be formed after excluding exactly the target store.
// Never insert the fitted target into a resource max or a whole-kernel floor.
inline std::optional<double> composeTargetOnly(double auxiliarySystemCycles,
                                               double targetTicks,
                                               double ticksPerSystemCycle) {
  if (!std::isfinite(auxiliarySystemCycles) || auxiliarySystemCycles < 0 ||
      !std::isfinite(targetTicks) || targetTicks < 0 ||
      !std::isfinite(ticksPerSystemCycle) || ticksPerSystemCycle != 1.0)
    return std::nullopt;
  const double total = auxiliarySystemCycles + targetTicks / ticksPerSystemCycle;
  return std::isfinite(total) ? std::optional<double>(total) : std::nullopt;
}
} // namespace indirect_store_round4::main_ta
#endif
