// Main-TA JSON caller contract; no runtime/device or geometry inference.
#ifndef ORDINARY_STORE_ROUND4_MAIN_CONTRACT_H
#define ORDINARY_STORE_ROUND4_MAIN_CONTRACT_H
#include "OrdinaryStoreRound4.h"
#include "llvm/Support/JSON.h"

namespace indirect_store_round4::main_ta {
inline bool sha256(const std::string &s) {
  return s.size() == 64 && std::all_of(s.begin(), s.end(), [](char c) {
           return (c >= '0' && c <= '9') || (c >= 'a' && c <= 'f');
         });
}
inline std::string validateDirectCaller(const llvm::json::Object &c,
                                        bool entryAndIRBound,
                                        bool transformBound,
                                        bool addressIntegerDomainDeclared,
                                        const std::string &actualProfile,
                                        const std::string &actualUnitSource) {
  auto text = [](const llvm::json::Object &o, llvm::StringRef key) {
    return o.getString(key).value_or(llvm::StringRef{});
  };
  auto hash = [&](const llvm::json::Object &o, llvm::StringRef key) {
    return indirect_store_round4::main_ta::sha256(text(o, key).str());
  };
  if (!c.getBoolean("enabled").value_or(false))
    return "candidate_default_off";
  const auto *life = c.getObject("lifecycle");
  if (!life || !life->getBoolean("first_store").value_or(false) ||
      text(*life, "allocation") != "fresh_for_this_launch" ||
      text(*life, "initialization") != "uninitialized_payload_guards_only" ||
      text(*life, "aliasing") != "exclusive_target_no_external_alias" ||
      text(*life, "reuse") != "not_reused" ||
      text(*life, "retention") != "valid_until_kernel_completion")
    return "explicit_caller_first_store_facts_missing";
  if (!hash(c, "source_sha256") || text(*life, "launch_id").empty() ||
      text(*life, "allocation_id").empty() ||
      life->getInteger("allocation_epoch").value_or(0) <= 0)
    return "explicit_caller_lifecycle_identity_or_proof_missing";
  for (auto key :
       {"allocation_proof_sha256", "initialization_proof_sha256",
        "alias_proof_sha256", "reuse_proof_sha256", "retention_proof_sha256"})
    if (!hash(*life, key))
      return "explicit_caller_lifecycle_identity_or_proof_missing";
  if (!entryAndIRBound)
    return "entry_or_transformed_IR_not_bound";
  if (!transformBound)
    return "unsupported_transformed_route_contract";
  if (!addressIntegerDomainDeclared)
    return "loaded_index_bounds_or_integer_no_wrap_caller_contract_missing";
  if (text(c, "increment_contract") !=
          "round4_first_store_matched_increment_syscnt_v1" ||
      text(c, "accounting_boundary") != "round4_matched_target_only")
    return "matched_target_only_contract_missing";
  const auto ticks = c.getNumber("syscnt_ticks_per_system_cycle").value_or(0.);
  if (text(c, "unit_contract") !=
          "syscnt_ticks_to_active_profile_system_cycles_v1" ||
      !std::isfinite(ticks) || ticks != 1.)
    return "active_SYS_CNT_profile_unit_contract_missing";
  if (!indirect_store_round4::main_ta::sha256(actualProfile) ||
      !indirect_store_round4::main_ta::sha256(actualUnitSource) ||
      text(c, "profile_sha256") != actualProfile ||
      text(c, "unit_source_sha256") != actualUnitSource)
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
  const double total =
      auxiliarySystemCycles + targetTicks / ticksPerSystemCycle;
  return std::isfinite(total) ? std::optional<double>(total) : std::nullopt;
}
} // namespace indirect_store_round4::main_ta
#endif
