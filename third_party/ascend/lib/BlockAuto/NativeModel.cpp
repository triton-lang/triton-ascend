// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "Selection.h"
#include <set>
#include <sstream>

namespace mlir::triton::ascend::blockauto {
static std::vector<std::string> argumentNames(const std::string &ttir) {
  std::istringstream lines(ttir);
  static const Regex names(R"(%([[:alnum:]_.$-]+):)");
  for (std::string line; std::getline(lines, line);)
    if (line.find("tt.func public @") != std::string::npos)
      return matches(line, names);
  throw std::runtime_error("candidate TTIR entry unavailable");
}
A launchArguments(const O &contract, const std::string &ttir) {
  const O &arguments = sub(contract, "arguments");
  const A &order = array(contract, "argument_order"),
          &grid = array(contract, "grid");
  check(grid.size() == 3 && order.size() == arguments.size(),
        "launch/TTIR ABI mismatch");
  A args{nullptr, nullptr}; // TA synchronization and workspace pointers.
  std::set<std::string> seen;
  unsigned host = 0;
  for (const auto &name : argumentNames(ttir)) {
    if (StringRef(name).starts_with("__hacc_original_grid_")) {
      size_t axis = std::string("xyz").find(name.back());
      check(axis < 3, "invalid original grid argument");
      args.push_back(grid[axis]);
      continue;
    }
    check(host < order.size() && order[host].getAsString(),
          "launch/TTIR ABI mismatch");
    std::string key = order[host++].getAsString()->str();
    check(seen.insert(key).second, "launch/TTIR ABI mismatch");
    const O &arg = sub(arguments, key);
    check(str(arg, "kind") == "tensor" || str(arg, "kind") == "scalar",
          "unsupported host argument");
    args.push_back(str(arg, "kind") == "scalar" ? get(arg, "value")
                                                : V(nullptr));
  }
  check(host == order.size(), "launch/TTIR ABI mismatch");
  return args;
}
bool preservesGrid(const O &metadata, int64_t factor) {
  const O &report = optional(metadata, "block_batching"),
          &guard = optional(report, "guard");
  bool grouped = str(report, "vv_lowering") == "sequential_iterations_v1";
  if (!grouped && !guard.get("loop_schedule"))
    return false;
  check(factor != 1 && integer(report, "lane_width") == factor,
        "VV_LOOP_GRID_CONTRACT_MISMATCH");
  const O &schedule = sub(guard, "loop_schedule");
  check(schedule.size() == 5 && integer(schedule, "version") == 1 &&
            str(schedule, "kind") == "sequential_iterations_v1" &&
            str(schedule, "grid_policy") == "preserve" &&
            integer(schedule, "factor") == factor,
        "VV_LOOP_SCHEDULE_MISMATCH");
  const O &loop = sub(schedule, "loop");
  bool found = false;
  if (const A *accesses = guard.getArray("accesses"))
    for (const V &access : *accesses)
      if (const A *loops = object(access).getArray("loops"))
        for (const V &fact : *loops)
          found |= fact == V(O(loop));
  const V &layouts = get(guard, "layout_checks");
  check(loop.size() == 5 && loop.get("id") && loop.get("bits") &&
            loop.get("lower") && loop.get("upper") && loop.get("step") &&
            found &&
            (layouts.kind() == V::Null ||
             (layouts.getAsArray() && layouts.getAsArray()->empty())),
        "VV_LOOP_SCHEDULE_MISMATCH");
  check(grouped && str(report, "cv_mode") == "disabled" &&
            get(metadata, "program_grid_transforms").kind() == V::Null &&
            integer(metadata, "coalesce_factor", 1) == 1,
        "VV_LOOP_GRID_CONTRACT_MISMATCH");
  for (StringRef key : {"program_grid_mapping_applied",
                        "row_coalescing_applied", "ptsm_cap_authorized"})
    check(metadata.getBoolean(key) && !*metadata.getBoolean(key),
          "VV_LOOP_GRID_CONTRACT_MISMATCH");
  return true;
}
std::string rejection(const O &item, const O &contract) {
  const O &metadata = sub(item, "metadata");
  A grid(array(contract, "grid"));
  check(grid.size() == 3, "invalid launch grid");
  int64_t tasks = product(grid), factor = integer(item, "factor"),
          axis = integer(item, "axis");
  check(factor == 1 || factor == 2 || factor == 4, "invalid factor");
  const V *floor = contract.get("runtime_core_floor");
  if (!floor)
    floor = contract.get("core_floor");
  if (floor && floor->getAsString())
    check(!floor->getAsString()->empty() &&
              floor->getAsString()->find_first_not_of("0123456789") ==
                  StringRef::npos,
          "invalid core floor");
  int64_t cores =
      floor && floor->kind() != V::Null
          ? (floor->getAsString() ? std::stoll(floor->getAsString()->str())
                                  : integer(*floor))
          : spec(str(item, "final_ir"), "VECTOR_CORE_COUNT");
  check(cores >= 0, "invalid core floor");
  if (factor == 1)
    return "";
  check(axis >= 0 && axis < 3, "invalid axis");
  bool preserve = preservesGrid(metadata, factor);
  if (!preserve)
    grid[axis] = (integer(grid[axis]) - 1) / factor + 1;
  int64_t merged = product(grid);
  if (!preserve && merged >= tasks)
    return "GRID_NOT_REDUCED";
  return merged < cores ? "GRID_BELOW_CORE_FLOOR" : "";
}
O modelConfig(const O &item, const O &contract) {
  const O &metadata = sub(item, "metadata");
  std::string mode = str(metadata, "parallel_mode"), ir = str(item, "final_ir");
  bool vv = str(metadata, "mix_mode") == "aiv";
  int64_t factor = integer(item, "factor"), axis = integer(item, "axis"),
          cores = spec(ir, vv ? "VECTOR_CORE_COUNT" : "CUBE_CORE_COUNT");
  check(str(sub(metadata, "target"), "arch") == "Ascend950PR_9579",
        "unsupported target profile");
  check(factor == 1 || factor == 2 || factor == 4,
        "factor outside automatic shortlist");
  check(mode == "simd" || (vv && mode == "mix_simd_simt"),
        "unmodelled execution mode");
  check(!flag(metadata, "ptsm_cap_authorized") &&
            !flag(metadata, "row_coalescing_applied") &&
            integer(metadata, "coalesce_factor", 1) == 1,
        "additional grid transform");
  A grid(array(contract, "grid")), args(array(item, "arguments"));
  check(grid.size() == 3, "invalid launch grid");
  product(grid);
  if (factor != 1 && !(vv && preservesGrid(metadata, factor))) {
    check(axis >= 0 && axis < 3, "invalid axis");
    grid[axis] = (integer(grid[axis]) - 1) / factor + 1;
  } else if (vv && factor == 1)
    preservesGrid(metadata, factor);
  for (const V &n : grid)
    args.push_back(n);
  int64_t blocks = product(grid);
  check(bool(metadata.getBoolean("auto_blockify_enabled")),
        "missing automatic block dispatch setting");
  bool automatic = flag(metadata, "auto_blockify_enabled");
  if (automatic)
    blocks = std::min(blocks, cores);
  O modules;
  if (vv)
    modules["aiv"] = "";
  else {
    for (StringRef role : {"aic", "aiv"})
      if (item.get("llvm_" + role.str()))
        modules[role] = "";
    if (modules.empty() && item.get("llvm"))
      modules["aic"] = "";
    if (modules.get("aiv"))
      check(spec(ir, "VECTOR_CORE_COUNT") == cores * 2,
            "only one Cube with two Vector subblocks is modelled");
  }
  O config{{"mode", vv ? "vv" : "cv"},
           {"modules", std::move(modules)},
           {"name", get(metadata, "name")},
           {"arguments", std::move(args)},
           {"grid", std::move(grid)},
           {"cores", cores},
           {"blocks", blocks},
           {"auto_blockify", automatic}};
  if (vv) {
    if (mode == "mix_simd_simt")
      config["mixed"] = O{{"simt_transaction_bytes", 64},
                          {"simt_transition_cycles", 32},
                          {"simt_instruction_cycles", 1}};
  }
  return config;
}
} // namespace mlir::triton::ascend::blockauto
