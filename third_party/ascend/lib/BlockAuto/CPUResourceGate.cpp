// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "Selection.h"
#include <set>
#include <sstream>

namespace mlir::triton::ascend::blockauto {
std::vector<std::string> matches(StringRef text, const Regex &pattern,
                                 unsigned group) {
  std::vector<std::string> result;
  SmallVector<StringRef> groups;
  while (pattern.match(text, &groups)) {
    result.push_back(groups[group].str());
    text = text.drop_front(groups[0].end() - text.begin());
  }
  return result;
}
int64_t spec(StringRef ir, StringRef key) {
  std::set<int64_t> values;
  static const Regex value(
      R"(^,[[:space:]]*([0-9]+)[[:space:]]*:[[:space:]]*i[0-9]+)");
  StringRef remaining(ir);
  std::string needle = "\"" + key.str() + "\"";
  SmallVector<StringRef> groups;
  for (size_t at; (at = remaining.find(needle)) != StringRef::npos;) {
    remaining = remaining.drop_front(at + needle.size());
    if (value.match(remaining, &groups))
      values.insert(std::stoll(groups[1].str()));
  }
  check(values.size() == 1 && *values.begin() > 0,
        "missing or ambiguous target specification: " + key.str());
  return *values.begin();
}
O resources(const O &item) {
  const O &metadata = sub(item, "metadata");
  StringRef ir = item.getString("final_ir").value_or("");
  std::string mode = str(metadata, "parallel_mode");
  check(str(sub(metadata, "target"), "arch") == "Ascend950PR_9579",
        "cost model target is outside the validated A5 profile");
  bool vv = str(metadata, "mix_mode") == "aiv",
       mixed = vv && mode == "mix_simd_simt", bound = false, function = false;
  int64_t capacity = spec(ir, "UB_SIZE"),
          alignment = vv ? spec(ir, "UB_ALIGN_SIZE") : 8, end = -1;
  check(alignment >= 8 && alignment % 8 == 0, "invalid UB alignment");
  std::map<std::string, int64_t> constants;
  static const Regex functionPattern(
      R"(^[[:space:]]*func\.func[[:space:]]+(private[[:space:]]+)?@([^[:space:](]+))"),
      constantPattern(
          R"(^[[:space:]]*(%[[:alnum:]_.$-]+) = llvm\.mlir\.constant\((-?[0-9]+) : (i[0-9]+|index)\))"),
      pointerPattern(
          R"(hivm\.hir\.pointer_cast\((%[[:alnum:]_.$-]+)\)[[:space:]]*:[[:space:]]*memref<(([0-9]+x)*)(bf16|[if][0-9]+),[[:space:]]*#hivm\.address_space<ub>>)");
  std::istringstream lines(ir.str());
  for (std::string line; std::getline(lines, line);) {
    SmallVector<StringRef> m;
    if (StringRef(line).contains("func.func") && functionPattern.match(line)) {
      function = true;
      constants.clear();
    }
    if (StringRef(line).contains("llvm.mlir.constant(") &&
        constantPattern.match(line, &m))
      constants[m[1].str()] = std::stoll(m[2].str());
    if (line.find("#hivm.address_space<ub>") == std::string::npos)
      continue;
    check(line.find("memref.alloc") == std::string::npos,
          "unresolved UB allocation remains in final IR");
    if (line.find("hivm.hir.pointer_cast") == std::string::npos)
      continue;
    check(pointerPattern.match(line, &m) && constants.count(m[1].str()) &&
              function,
          "UB pointer has an unproved offset, shape or layout");
    int64_t offset = constants[m[1].str()],
            bits = m[4] == "bf16" ? 16 : std::stoll(m[4].drop_front().str());
    bool conservative = vv && bits == 1;
    check(offset >= 0 && (conservative || (bits >= 8 && bits % 8 == 0)),
          "unsupported UB element width or negative address");
    A factors{conservative ? 8 : bits};
    StringRef shape = m[2];
    while (!shape.empty()) {
      auto dim = shape.split('x');
      factors.push_back(std::stoll(dim.first.str()));
      shape = dim.second;
    }
    bits = product(factors);
    check(bits <= INT64_MAX - alignment + 1 &&
              offset <= (INT64_MAX - bits - alignment + 1) / 8,
          "UB extent overflow");
    end = std::max(end,
                   offset * 8 + (bits + alignment - 1) / alignment * alignment);
    bound |= conservative;
  }
  if (end < 0) {
    // Cube-only output may retain mix_mode=mix in compiler metadata.
    check(function && !vv && !item.get("llvm_aiv") &&
              !ir.contains("address_space<ub>"),
          "no verifiable static UB allocation in final IR");
    end = 0;
  }
  O usage{{"capacity_bits", capacity},
          {"address_extent_bits", end},
          {"conservative_upper_bound", bound}};
  if (vv) {
    check(mode == "simd" || mixed,
          "only AIV SIMD and MIX resource proofs are supported");
    usage["backend_usable_bits"] = capacity;
    if (mixed) {
      int64_t dynamic = integer(metadata, "shared_mem_dynamic_size"),
              full = capacity;
      check(dynamic > 0 && dynamic <= INT64_MAX / 8,
            "missing runtime dynamic UB size");
      capacity = dynamic * 8;
      check(full - spec(ir, "MAXIMUM_D_CACHE_SIZE") <= capacity &&
                capacity <= full - spec(ir, "MINIMAL_D_CACHE_SIZE"),
            "MIX dynamic UB conflicts with D-cache reservation");
      // The compiler owns the effective shared-memory setting in metadata.
      usage["reserved_cache_bits"] = full - capacity;
      usage["capacity_bits"] = capacity;
    }
  }
  check(end <= capacity, bound ? "i1 conservative UB upper bound exceeds "
                                 "capacity; actual overflow unconfirmed"
                               : "final UB address exceeds capacity");
  return usage;
}
} // namespace mlir::triton::ascend::blockauto
