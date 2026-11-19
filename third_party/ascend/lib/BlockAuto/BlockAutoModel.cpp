// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "IRAnalysis.h"
#include "llvm/Support/raw_ostream.h"
#include <tuple>

namespace mlir::triton::ascend::blockauto {
bool excluded(StringRef status, bool vv) {
  return status == "UB_REJECTED" || status == "COMPILE_REJECTED" ||
         status == "FRONTEND_REJECTED" || status == "NO_MERGE_OPPORTUNITY" ||
         (vv && status == "RUNTIME_REJECTED");
}
static std::string key(const O &item) {
  return integer(item, "factor") == 1
             ? "unmerged"
             : "a" + std::to_string(integer(item, "axis")) + "_f" +
                   std::to_string(integer(item, "factor"));
}
static O prepare(const O &item, const O &contract, MLIRContext &context) {
  O state{
      {"key", key(item)}, {"resource_checked", false}, {"needs_llvm", false}};
  bool vv = false;
  try {
    if (!item.get("metadata")) {
      if (!excluded(str(item, "status")))
        state["status"] = "RESOURCE_UNCONFIRMED";
      return state;
    }
    auto module = parseCandidateIR(str(item, "final_ir"), context);
    ModuleAnalysisManager manager(module->getOperation(), nullptr);
    auto &ir = AnalysisManager(manager).getAnalysis<CandidateIRAnalysis>();
    ir.checkResources(item);
    A arguments = ir.launchArguments(contract, sub(item, "metadata"));
    state["resource_checked"] = true;
    vv = str(sub(item, "metadata"), "mix_mode") == "aiv";
    state["vv"] = vv;
    int64_t factor = integer(item, "factor"), axis = integer(item, "axis");
    check((factor == 1 || factor == 2 || factor == 4) &&
              (factor == 1 || (axis >= 0 && axis < 3)),
          "invalid merge factor or axis");
    if (vv) {
      std::string reason = rejection(item, contract, ir);
      if (!reason.empty()) {
        update(state, O{{"status", "RUNTIME_REJECTED"}, {"error", reason}});
        return state;
      }
      auto path = ir.metaLibrary();
      if (!path.empty())
        state["meta_library"] = path;
    } else
      state["merge_kind"] =
          mergeKind(sub(item, "metadata"), integer(item, "factor"));
    state["config"] = modelConfig(item, contract, arguments, ir);
    state["needs_llvm"] = vv;
  } catch (const std::exception &error) {
    if (!flag(state, "resource_checked"))
      state["status"] = "RESOURCE_UNCONFIRMED";
    else if (vv)
      state["status"] = "STATIC_ANALYSIS_UNSUPPORTED";
    state["error"] = std::string(error.what());
  }
  return state;
}
using Items = std::map<std::string, const O *>;
static O rank(const Items &items, const O &contract) {
  O result{{"model", "jit-cv-vv-native-v1"},
           {"selected_key", "unmerged"},
           {"selected_axis", -1},
           {"model_complete", false},
           {"reason", "BASELINE_FALLBACK_INCOMPLETE_MODEL"},
           {"candidates", O{}},
           {"device_timing_used", false},
           {"factor", 1}};
  try {
    const O &baseline = *items.at("unmerged");
    if (!flag(baseline, "resource_checked")) {
      O states;
      for (const auto &[key, value] : items) {
        O row;
        const O &item = *value;
        for (StringRef field :
             {"factor", "axis", "status", "error", "resource_checked"})
          if (item.get(field))
            row[field] = get(item, field);
        states[key] = std::move(row);
      }
      update(result, O{{"reason", "BASELINE_RESOURCE_UNCONFIRMED"},
                       {"candidates", std::move(states)}});
      return result;
    }
    bool vv = flag(baseline, "vv");
    if (!vv) {
      bool tensor = false;
      for (const auto &[name, value] : sub(contract, "arguments")) {
        const O &arg = object(value);
        if (str(arg, "kind") != "tensor")
          continue;
        tensor = true;
        std::string type = str(arg, "dtype");
        if (type != "fp32" && type != "float32" && type != "torch.float32") {
          result["reason"] = "CV_DTYPE_NOT_COVERED";
          return result;
        }
      }
      if (!tensor) {
        result["reason"] = "CV_DTYPE_NOT_COVERED";
        return result;
      }
    }
    O records;
    for (const auto &[key, value] : items) {
      const O &item = *value;
      O row{{"factor", get(item, "factor")},
            {"axis", get(item, "axis")},
            {"resource_checked", vv && flag(item, "resource_checked")},
            {"status", str(item, "status",
                           vv ? "STATIC_ANALYSIS_UNSUPPORTED"
                              : "STATIC_ANALYSIS_UNCONFIRMED")}};
      if (vv)
        row["error"] = get(item, "error");
      try {
        check(flag(item, "resource_checked"),
              str(item, "error", "candidate resource evidence unavailable"));
        if (vv && (!flag(item, "needs_llvm") || !item.get("optimized_llvm"))) {
          records[key] = std::move(row);
          continue;
        }
        check(item.getObject("config"),
              str(item, "error", "missing analysis configuration"));
        O config(sub(item, "config"));
        for (auto &[role, source] : *config.getObject("modules"))
          source = item.getString(vv ? "optimized_llvm" : "llvm_" + role.str())
                       .value_or(item.getString("llvm").value_or(""));
        O trace = analyzeBlockCandidate(config);
        if (vv)
          update(row, std::move(trace));
        else {
          double score = number(trace, "score");
          const O &counts = sub(trace, "counts");
          check(number(counts, "mma_calls") > 0 && score > 0,
                "incomplete Cube accounting");
          update(row, O{{"status", "STATIC_FEATURES_READY"},
                        {"resource_checked", true},
                        {"estimated_cycles", score},
                        {"merge_kind", get(item, "merge_kind")},
                        {"batch_grid", get(trace, "logical_grid")},
                        {"mode", get(trace, "mode")},
                        {"useful_flops", get(counts, "useful_flops")},
                        {"cube_core_count", get(trace, "cube_core_count")}});
        }
      } catch (const std::exception &error) {
        row["error"] = std::string(error.what());
        if (vv && flag(item, "resource_checked"))
          row["status"] = "STATIC_ANALYSIS_UNSUPPORTED";
      }
      records[key] = std::move(row);
    }
    if (vv) {
      bool complete = !records.empty();
      std::string best;
      auto order = [&](const std::string &name) {
        const O &o = sub(records, name);
        return std::make_tuple(number(o, "score"), integer(o, "factor"), name);
      };
      for (const auto &[key, value] : records) {
        const O &row = object(value);
        bool scored = row.getNumber("score") &&
                      str(row, "status") == "STATIC_COST_ESTIMATED";
        complete &= scored || excluded(str(row, "status"));
        if (scored && (best.empty() || order(key.str()) < order(best)))
          best = key.str();
      }
      if (complete && !best.empty())
        update(result, O{{"selected_key", best},
                         {"model_complete", true},
                         {"reason", "LOWEST_STATIC_SCORE"}});
    } else
      update(result, rankCV(records, contract));
    result["candidates"] = std::move(records);
    const O &selected = *items.at(str(result, "selected_key"));
    result["selected_axis"] = get(selected, "axis");
    result["factor"] = get(selected, "factor");
  } catch (const std::exception &error) {
    result["error"] = std::string(error.what());
  }
  return result;
}
} // namespace mlir::triton::ascend::blockauto
namespace mlir::triton::ascend {
std::string selectBlockCandidates(llvm::StringRef request) {
  using namespace blockauto;
  auto parsed = json::parse(request);
  if (!parsed)
    throw std::runtime_error(toString(parsed.takeError()));
  const O &input = object(*parsed), &contract = sub(input, "contract");
  std::string action = str(input, "action");
  O result;
  if (action == "plan")
    result = planCV(input);
  else if (action == "prepare") {
    DialectRegistry registry;
    registerCandidateDialects(registry);
    MLIRContext context(registry, MLIRContext::Threading::DISABLED);
    context.allowUnregisteredDialects();
    A states;
    for (const V &value : array(input, "candidates"))
      states.push_back(prepare(object(value), contract, context));
    bool baselineReady =
        std::any_of(states.begin(), states.end(), [](const V &v) {
          return str(object(v), "key") == "unmerged" &&
                 flag(object(v), "resource_checked");
        });
    if (!baselineReady)
      for (V &v : states)
        (*v.getAsObject())["needs_llvm"] = false;
    result = O{{"candidates", std::move(states)}};
  } else {
    check(action == "rank", "unknown selector action");
    Items items;
    for (const V &value : array(input, "candidates")) {
      const O &item = object(value);
      items[key(item)] = &item;
    }
    result = rank(items, contract);
  }
  std::string output;
  raw_string_ostream stream(output);
  stream << V(std::move(result));
  return output;
}
} // namespace mlir::triton::ascend
