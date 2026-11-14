// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "IRAnalysis.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include <set>

namespace mlir::triton::ascend::blockauto {
A CandidateIRAnalysis::launchArguments(const O &contract,
                                       const O &metadata) const {
  const O &arguments = sub(contract, "arguments");
  const A &order = array(contract, "argument_order"),
          &grid = array(contract, "grid");
  check(grid.size() == 3 && order.size() == arguments.size(),
        "launch ABI mismatch");
  A args{nullptr, nullptr}; // TA synchronization and workspace pointers.
  std::set<std::string> seen;
  for (const V &name : order) {
    check(bool(name.getAsString()), "missing host argument name");
    std::string key = name.getAsString()->str();
    check(seen.insert(key).second, "launch ABI mismatch");
    const O &arg = sub(arguments, key);
    check(str(arg, "kind") == "tensor" || str(arg, "kind") == "scalar",
          "unsupported host argument");
    args.push_back(str(arg, "kind") == "scalar" ? get(arg, "value")
                                                : V(nullptr));
  }
  // The compiler records hidden extents in ABI order; SSA names are irrelevant.
  if (auto *axes = optional(metadata, "program_grid_transforms")
                       .getArray("hidden_extent_axes"))
    for (const V &value : *axes) {
      int64_t axis = integer(value);
      check(axis >= 0 && axis < 3, "invalid original grid axis");
      args.push_back(grid[axis]);
    }
  auto entry = dyn_cast_or_null<FunctionOpInterface>(
      SymbolTable::lookupSymbolIn(module, str(metadata, "name")));
  if (!entry)
    entry = dyn_cast_or_null<FunctionOpInterface>(SymbolTable::lookupSymbolIn(
        module, str(metadata, "name") + "_mix_aic"));
  check(entry && entry.getNumArguments() == args.size() + 3,
        "launch/final IR ABI mismatch");
  return args;
}
bool preservesGrid(const O &metadata) {
  // Lowering legality and loop proofs belong to the compiler pass.
  return str(optional(metadata, "block_batching"), "vv_lowering") ==
         "sequential_iterations_v1";
}
std::string rejection(const O &item, const O &contract,
                      const CandidateIRAnalysis &ir) {
  const O &metadata = sub(item, "metadata");
  A grid(array(contract, "grid"));
  int64_t tasks = product(grid), factor = integer(item, "factor"),
          axis = integer(item, "axis");
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
          : ir.spec("VECTOR_CORE_COUNT");
  check(cores >= 0, "invalid core floor");
  if (factor == 1)
    return "";
  bool preserve = preservesGrid(metadata);
  if (!preserve)
    grid[axis] = (integer(grid[axis]) - 1) / factor + 1;
  int64_t merged = product(grid);
  if (!preserve && merged >= tasks)
    return "GRID_NOT_REDUCED";
  return merged < cores ? "GRID_BELOW_CORE_FLOOR" : "";
}
O modelConfig(const O &item, const O &contract, const A &arguments,
              const CandidateIRAnalysis &ir) {
  const O &metadata = sub(item, "metadata");
  std::string mode = str(metadata, "parallel_mode");
  bool vv = str(metadata, "mix_mode") == "aiv",
       preserve = vv && preservesGrid(metadata);
  int64_t factor = integer(item, "factor"), axis = integer(item, "axis"),
          cores = ir.spec(vv ? "VECTOR_CORE_COUNT" : "CUBE_CORE_COUNT");
  check(mode == "simd" || (vv && mode == "mix_simd_simt"),
        "unmodelled execution mode");
  check(!flag(metadata, "ptsm_cap_authorized") &&
            !flag(metadata, "row_coalescing_applied") &&
            (!preserve ||
             (!flag(metadata, "program_grid_mapping_applied") &&
              get(metadata, "program_grid_transforms").kind() == V::Null)) &&
            integer(metadata, "coalesce_factor", 1) == 1,
        "additional grid transform");
  A grid(array(contract, "grid")), args(arguments);
  product(grid);
  if (factor != 1 && !preserve)
    grid[axis] = (integer(grid[axis]) - 1) / factor + 1;
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
      check(ir.spec("VECTOR_CORE_COUNT") == cores * 2,
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
