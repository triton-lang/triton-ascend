// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "Selection.h"
#include "llvm/Support/MemoryBuffer.h"
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
int64_t spec(const std::string &ir, StringRef key) {
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
  const std::string ir = str(item, "final_ir"),
                    target = str(sub(metadata, "target"), "arch"),
                    mode = str(metadata, "parallel_mode");
  check(target == "Ascend950PR_9579",
        "cost model target is outside the validated A5 profile");
  static const Regex targetPattern(
      "hacc\\.target[[:space:]]*=[[:space:]]*#hacc\\.target<\"([^\"\n]+)\">");
  auto targets = matches(ir, targetPattern);
  std::set<std::string> unique(targets.begin(), targets.end());
  check(unique.size() == 1 &&
            (*unique.begin() == target || *unique.begin() == "Ascend950"),
        "unexpected allocation target");
  bool vv = str(metadata, "mix_mode") == "aiv",
       mixed = vv && mode == "mix_simd_simt", bound = false, function = false;
  int64_t capacity = spec(ir, "UB_SIZE"),
          alignment = vv ? spec(ir, "UB_ALIGN_SIZE") : 8, end = -1;
  check(alignment >= 8 && alignment % 8 == 0, "invalid UB alignment");
  std::map<std::string, int64_t> constants;
  std::vector<std::string> functions, entries;
  static const Regex functionPattern(
      R"(^[[:space:]]*func\.func[[:space:]]+(private[[:space:]]+)?@([^[:space:](]+))"),
      constantPattern(
          R"(^[[:space:]]*(%[[:alnum:]_.$-]+) = llvm\.mlir\.constant\((-?[0-9]+) : (i[0-9]+|index)\))"),
      pointerPattern(
          R"(hivm\.hir\.pointer_cast\((%[[:alnum:]_.$-]+)\)[[:space:]]*:[[:space:]]*memref<(([0-9]+x)*)(bf16|[if][0-9]+),[[:space:]]*#hivm\.address_space<ub>>)");
  std::istringstream lines(ir);
  for (std::string line; std::getline(lines, line);) {
    SmallVector<StringRef> m;
    if (StringRef(line).contains("func.func") && functionPattern.match(line)) {
      function = true;
      constants.clear();
      functions.push_back(line);
      if (line.find("hacc.entry") != std::string::npos)
        entries.push_back(line);
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
  static const Regex coreType(
      R"(hivm\.func_core_type[[:space:]]*=[[:space:]]*#hivm\.func_core_type<(AIC|AIV)>)");
  auto allCore = [&](StringRef core) {
    return !functions.empty() &&
           std::all_of(functions.begin(), functions.end(),
                       [&](const std::string &line) {
                         return matches(line, coreType) ==
                                std::vector<std::string>{core.str()};
                       });
  };
  if (end < 0) {
    check(allCore("AIC") &&
              !Regex(R"(address_space[[:space:]]*<[[:space:]]*ub[[:space:]]*>)")
                   .match(ir),
          "no verifiable static UB allocation in final IR");
    end = 0;
  }
  O usage{{"capacity_bits", capacity},
          {"address_extent_bits", end},
          {"conservative_upper_bound", bound}};
  if (vv) {
    check(mode == "simd" || mixed,
          "only AIV SIMD and MIX resource proofs are supported");
    check(entries.size() == 1 &&
              entries[0].find(mixed
                                  ? "hivm.vf_mode<MIX>"
                                  : "hivm.vf_mode<SIMD>") != std::string::npos,
          "compile metadata and final IR vector modes differ");
    usage["backend_usable_bits"] = capacity;
    if (mixed) {
      check(allCore("AIV"),
            "MIX metadata does not describe a single AIV MIX entry");
      int64_t dynamic = integer(metadata, "shared_mem_dynamic_size"),
              full = capacity;
      check(dynamic > 0 && dynamic <= INT64_MAX / 8,
            "missing runtime dynamic UB size");
      capacity = dynamic * 8;
      check(full - spec(ir, "MAXIMUM_D_CACHE_SIZE") <= capacity &&
                capacity <= full - spec(ir, "MINIMAL_D_CACHE_SIZE"),
            "MIX dynamic UB conflicts with D-cache reservation");
      std::vector<int64_t> explicitSizes;
      if (const auto *command =
              optional(metadata, "block_auto_evidence").getArray("command"))
        for (const V &arg : *command)
          if (auto s = arg.getAsString();
              s && s->starts_with("--simt-vf-dynamic-size=")) {
            int64_t size;
            check(!s->split('=').second.getAsInteger(10, size),
                  "invalid compile-time dynamic UB size");
            explicitSizes.push_back(size);
          }
      check(explicitSizes.size() <= 1,
            "ambiguous compile-time dynamic UB size");
      if (explicitSizes.empty()) {
        std::string defaults = str(item, "backend_defaults");
        if (defaults.empty()) {
          auto buffer = MemoryBuffer::getFile(
              str(optional(sub(item, "files"), "backend_defaults"), "path"));
          check(bool(buffer), "missing compiler dynamic-UB default");
          defaults = (*buffer)->getBuffer().str();
        }
        auto values = matches(
            defaults,
            Regex("--simt-vf-dynamic-size=<int>[^\n]*Default is ([0-9]+)\\."));
        check(values.size() == 1, "unknown compiler dynamic-UB default");
        explicitSizes.push_back(std::stoll(values[0]));
      }
      check(explicitSizes[0] > 0 && explicitSizes[0] <= INT64_MAX / 1024 &&
                explicitSizes[0] * 1024 == dynamic,
            "compile, resource proof and runtime UB capacities differ");
      usage["reserved_cache_bits"] = full - capacity;
      usage["capacity_bits"] = capacity;
    }
    usage["evidence"] =
        "backend allocation, aligned extents and effective runtime capacity";
  }
  check(end <= capacity, bound ? "i1 conservative UB upper bound exceeds "
                                 "capacity; actual overflow unconfirmed"
                               : "final UB address exceeds capacity");
  return usage;
}
} // namespace mlir::triton::ascend::blockauto
