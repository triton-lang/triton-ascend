// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "BlockAuto/Analysis.h"
#include "TritonToGraph/ResourceCostModel.h"
#include "mlir/Analysis/DataFlow/DeadCodeAnalysis.h"
#include "mlir/Analysis/DataFlow/IntegerRangeAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/IR/TypeUtilities.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/Passes.h"
#include "triton/Analysis/AxisInfo.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <cmath>
#include <map>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace mlir::triton::ascend::blockauto {
using namespace llvm;
using O = json::Object;
using A = json::Array;
using V = json::Value;
inline void check(bool valid, StringRef reason) {
  if (!valid)
    throw std::runtime_error(reason.str());
}
inline const V &get(const O &o, StringRef key) {
  static const V missing = nullptr;
  auto *v = o.get(key);
  return v ? *v : missing;
}
inline const O &object(const V &v) {
  check(v.getAsObject(), "expected an object");
  return *v.getAsObject();
}
inline const O &sub(const O &o, StringRef key) { return object(get(o, key)); }
inline const O &optional(const O &o, StringRef key) {
  static const O empty;
  auto *v = o.getObject(key);
  return v ? *v : empty;
}
inline const A &array(const O &o, StringRef key) {
  auto *v = o.getArray(key);
  check(v, "missing array: " + key.str());
  return *v;
}
inline std::string str(const O &o, StringRef key, StringRef fallback = "") {
  return o.getString(key).value_or(fallback).str();
}
inline int64_t integer(const V &v) {
  auto n = v.getAsInteger();
  check(bool(n), "expected an integer");
  return *n;
}
inline int64_t integer(const O &o, StringRef key, int64_t fallback = 0) {
  return o.get(key) ? integer(*o.get(key)) : fallback;
}
inline double number(const O &o, StringRef key) {
  auto n = o.getNumber(key);
  check(n && std::isfinite(*n), "missing finite number: " + key.str());
  return *n;
}
inline bool flag(const O &o, StringRef key) {
  return o.getBoolean(key).value_or(false);
}
inline void update(O &o, O values) {
  for (auto &v : values)
    o[v.first] = std::move(v.second);
}
// Keep this list in step with uses_grouping_policy in block_batching.py.
inline bool usesGroupingPolicy(StringRef arch) {
  return arch.starts_with("Ascend910_93") || arch.starts_with("Ascend910B");
}
inline int64_t product(const A &values) {
  int64_t n = 1;
  for (const V &v : values) {
    int64_t x = integer(v);
    check(x > 0 && n <= INT64_MAX / x, "nonpositive or overflowing extent");
    n *= x;
  }
  return n;
}

static std::string key(const O &item) {
  return integer(item, "factor") == 1
             ? "unmerged"
             : "a" + std::to_string(integer(item, "axis")) + "_f" +
                   std::to_string(integer(item, "factor"));
}

struct Rate {
  double interval = 1, latency = 4;
  unsigned slots = 2;
};

struct Profile {
  std::map<std::string, Rate> costs;
  double localBandwidth = 32, globalBandwidth = 512;

  explicit Profile(bool cv) {
    // A5 assumptions: 56 Vector cores, 1650 MHz, 16 GB/s per Vector core.
    if (!cv) {
      localBandwidth = 16.0 * 1e9 / (1650.0 * 1e6);
      globalBandwidth = localBandwidth * 56;
    }
#define AVE_COST_ENTRY(OP, SRC, DST, INTERVAL, LATENCY, SLOTS)                 \
  costs[#OP ":" #SRC ":" #DST] = {INTERVAL, LATENCY, SLOTS};
#include "bishengir/Dialect/HIVMAVE/CostModelInfo/AVECostModelInfo.def"
#undef AVE_COST_ENTRY
  }

  Rate cost(StringRef instruction, StringRef dtype) const {
    Rate fallback;
    if (instruction == "vdiv" || instruction == "vsqrt" ||
        instruction == "vrsqrt")
      fallback = {4, 14, 2};
    else if (instruction == "vexp")
      fallback = {4, 13, 2};
    else if (instruction == "vln")
      fallback = {4, 15, 2};
    std::string opcode = llvm::StringSwitch<std::string>(instruction)
                             .Case("vadd", "VFAddOp")
                             .Case("vsub", "VFSubOp")
                             .Case("vmul", "VFMulOp")
                             .Case("vdiv", "VFDivOp")
                             .Case("vsqrt", "VFSqrtOp")
                             .Case("vexp", "VFExpOp")
                             .Case("vln", "VFLnOp")
                             .Case("vmax", "VFMaxOp")
                             .Case("vmin", "VFMinOp")
                             .Case("vabs", "VFAbsOp")
                             .Case("vneg", "VFNegOp")
                             .Case("vsel", "VFSelectOp")
                             .Default("");
    auto found = costs.find(opcode + ":" + dtype.str() + ":" + dtype.str());
    return found == costs.end() ? fallback : found->second;
  }
};

std::string mergeKind(const O &metadata, int64_t factor, bool frontend) {
  if (factor == 1)
    return "unmerged";
  const O &report = frontend ? metadata : optional(metadata, "block_batching");
  check(!frontend || str(report, "compile_status") == "REWRITTEN",
        "frontend did not admit the CV rewrite");
  if (str(report, "cv_mode") == "independent_batch_v1")
    return frontend ? "batch" : "independent_batch";
  std::string lowering = str(report, "cv_lowering");
  if (frontend && StringRef(lowering).ends_with("_uniform_control"))
    lowering.resize(lowering.size() - 16);
  if (str(report, "cv_mode") == "spatial_concat_v1") {
    if (lowering == "spatial_m_v1")
      return "M";
    if (lowering == "spatial_n_v1")
      return "N";
  }
  throw std::runtime_error("compiler CV axis role is not covered");
}
O planCV(const O &request) {
  const A &grid = array(sub(request, "contract"), "grid");
  int64_t selectedAxis = integer(request, "axis", -1);
  check(grid.size() == 3 && selectedAxis >= -1 && selectedAxis < 3,
        "invalid CV launch grid or axis");
  product(grid);
  A probes;
  auto result = [](StringRef reason, A candidates = {}) {
    return O{{"reason", reason}, {"candidates", std::move(candidates)}};
  };
  if (!request.get("reports")) {
    for (int64_t axis = 0; axis < 3; ++axis)
      for (int64_t factor : {2, 4})
        if ((selectedAxis == -1 || selectedAxis == axis) &&
            integer(grid[axis]) >= factor && integer(grid[axis]) % factor == 0)
          probes.push_back(O{{"axis", axis}, {"factor", factor}});
    return O{{"probes", std::move(probes)}};
  }
  std::map<std::string, A> admitted;
  std::map<int64_t, std::string> roles;
  for (const V &value : array(request, "reports")) {
    try {
      O item(object(value));
      int64_t axis = integer(item, "axis"), factor = integer(item, "factor");
      check(axis >= 0 && axis < 3 && (factor == 2 || factor == 4) &&
                (selectedAxis == -1 || axis == selectedAxis) &&
                integer(grid[axis]) >= factor &&
                integer(grid[axis]) % factor == 0,
            "invalid probe");
      const O &report = sub(item, "report");
      check(integer(report, "selected_axis", -1) == axis &&
                integer(report, "requested_factor") == factor,
            "frontend selected a different axis or factor");
      std::string kind = mergeKind(report, factor, true);
      auto [role, inserted] = roles.emplace(axis, kind);
      if (!inserted && role->second != kind)
        return result("CV_AXIS_ROLE_DEPENDS_ON_FACTOR");
      admitted[kind].push_back(
          O{{"axis", axis},
            {"factor", factor},
            {"key", "a" + std::to_string(axis) + "_f" + std::to_string(factor)},
            {"merge_kind", kind}});
    } catch (const std::exception &) {
    }
  }
  auto order = [](const V &v) {
    const O &o = object(v);
    return std::make_pair(integer(o, "axis"), integer(o, "factor"));
  };
  for (auto &[kind, candidates] : admitted)
    std::sort(candidates.begin(), candidates.end(),
              [&](const V &a, const V &b) { return order(a) < order(b); });
  A spatial;
  const O *m = nullptr, *n = nullptr;
  for (const std::string kind : {"M", "N"})
    for (const V &v : admitted[kind]) {
      const O &o = object(v);
      spatial.push_back(v);
      if (integer(o, "factor") == 4) {
        auto &slot = kind == "M" ? m : n;
        if (!slot)
          slot = &o;
      }
    }
  if (flag(sub(request, "contract"), "legacy_resource_policy")) {
    auto reason = spatial.empty() ? "CV_NO_ADMITTED_SPATIAL_MERGE"
                                  : "LEGACY_CV_SPATIAL_CANDIDATES";
    return result(reason, std::move(spatial));
  }
  if (m && n)
    return result("CV_M4_N4", A{O(*m), O(*n)});
  if (m || n) {
    A candidates;
    for (const V &v : spatial)
      if (integer(object(v), "axis") == integer(m ? *m : *n, "axis"))
        candidates.push_back(v);
    return result("CV_ONE_SPATIAL_AXIS_F2_F4", std::move(candidates));
  }
  if (!spatial.empty()) {
    while (spatial.size() > 2)
      spatial.pop_back();
    return result("CV_SPATIAL_F2_ONLY", std::move(spatial));
  }
  A batch;
  for (const V &v : admitted["batch"])
    if (batch.empty() ||
        integer(object(v), "axis") == integer(object(batch.front()), "axis"))
      batch.push_back(v);
  StringRef reason = batch.empty() ? "CV_NO_ADMITTED_MERGE" : "CV_BATCH_F2_F4";
  return result(reason, std::move(batch));
}

O rankTtir(const O &request) {
  const O &contract = sub(request, "contract");
  const A &candidates = array(request, "candidates");
  O records;
  std::vector<const O *> ready;
  bool baselineReady = false;
  for (const V &value : candidates) {
    const O &candidate = object(value);
    std::string name = key(candidate);
    records[name] = O(candidate);
    if (str(candidate, "status") != "TTIR_COST_ESTIMATED")
      continue;
    int64_t factor = integer(candidate, "factor");
    baselineReady |= factor == 1;
    if (!flag(candidate, "cv") && factor > 1) {
      const V &floor = get(contract, "core_floor");
      int64_t minimum =
          floor.kind() == V::Null
              ? integer(candidate, "cores")
              : (floor.getAsString() ? std::stoll(floor.getAsString()->str())
                                     : integer(floor));
      if (integer(candidate, "blocks") < minimum)
        continue;
    }
    ready.push_back(&candidate);
  }
  std::sort(ready.begin(), ready.end(), [](const O *a, const O *b) {
    return std::make_tuple(number(*a, "score"), integer(*a, "factor"),
                           integer(*a, "axis")) <
           std::make_tuple(number(*b, "score"), integer(*b, "factor"),
                           integer(*b, "axis"));
  });
  const O *selected = baselineReady && !ready.empty() ? ready.front() : nullptr;
  O result{{"model", "ttir-relative-v3"},
           {"device_timing_used", false},
           {"model_complete", baselineReady},
           {"factor", 1},
           {"selected_axis", -1},
           {"selected_key", "unmerged"},
           {"reason", "TTIR_BASELINE_MODEL_UNSUPPORTED"}};
  if (selected) {
    std::string reason = "LOWEST_TTIR_SCORE";
    bool spatial = llvm::any_of(ready, [](const O *row) {
      return str(*row, "merge_kind") == "M" || str(*row, "merge_kind") == "N";
    });
    if (!spatial &&
        contract.getBoolean("cv_empirical_short_batch").value_or(true))
      for (const O *row : ready)
        if (str(*row, "merge_kind") == "batch" &&
            integer(*row, "factor") == 2 &&
            product(array(contract, "grid")) <= integer(*row, "cores") &&
            llvm::count_if(array(contract, "grid"),
                           [](const V &v) { return integer(v) > 1; }) == 1) {
          selected = row;
          reason = "EMPIRICAL_SINGLE_TILE_SHORT_BATCH_X2";
          break;
        }
    update(result, O{{"factor", get(*selected, "factor")},
                     {"selected_axis", get(*selected, "axis")},
                     {"selected_key", key(*selected)},
                     {"reason", reason}});
  }
  result["candidates"] = std::move(records);
  A order;
  if (baselineReady)
    for (const O *row : ready)
      order.push_back(key(*row));
  if (selected) {
    auto chosen = std::find_if(order.begin(), order.end(), [&](const V &v) {
      return v.getAsString().value_or("") == key(*selected);
    });
    if (chosen != order.end())
      std::rotate(order.begin(), chosen, chosen + 1);
  }
  result["ranked_candidates"] = std::move(order);
  return result;
}

// A frontend estimate: these are relative costs, not final instruction counts.
struct TtirWork {
  double vector = 0, vectorPath = 0, cube = 0, scalar = 0;
  double simt = 0;
  double loads = 0, stores = 0, readBytes = 0, writeBytes = 0;
  bool boundedLoops = false;
};

// Seed launch dimensions, then let MLIR propagate all arithmetic ranges.
class TtirRanges : public dataflow::IntegerRangeAnalysis {
  const A &grid;

public:
  TtirRanges(DataFlowSolver &solver, const A &grid)
      : IntegerRangeAnalysis(solver), grid(grid) {}
  LogicalResult visitOperation(
      Operation *op,
      ArrayRef<const dataflow::IntegerValueRangeLattice *> operands,
      ArrayRef<dataflow::IntegerValueRangeLattice *> results) override {
    int axis = -1;
    bool count = false;
    if (auto id = dyn_cast<mlir::triton::GetProgramIdOp>(op))
      axis = id.getAxisAsInt();
    else if (auto size = dyn_cast<mlir::triton::GetNumProgramsOp>(op)) {
      axis = size.getAxisAsInt();
      count = true;
    }
    if (axis < 0)
      return IntegerRangeAnalysis::visitOperation(op, operands, results);
    int64_t extent = integer(grid[axis]);
    auto range = ConstantIntRanges::fromSigned(
        APInt(32, count ? extent : 0), APInt(32, count ? extent : extent - 1));
    propagateIfChanged(results[0], results[0]->join(IntegerValueRange(range)));
    return success();
  }
};

// Specialize only the analysis clone. The executable keeps its runtime ABI.
static OwningOpRef<ModuleOp> specializeTtir(ModuleOp module, const O &input) {
  OwningOpRef<ModuleOp> copy(cast<ModuleOp>(module->clone()));
  A arguments;
  if (auto *values = input.getArray("scalar_arguments"))
    arguments = A(*values);
  if (auto transforms = copy->getOperation()->getAttrOfType<DictionaryAttr>(
          "hacc.program_grid_transforms"))
    if (auto axes = transforms.getAs<ArrayAttr>("hidden_extent_axes"))
      for (Attribute axis : axes)
        arguments.push_back(array(sub(input, "contract"),
                                  "grid")[cast<IntegerAttr>(axis).getInt()]);
  for (auto fn : copy->getOps<mlir::triton::FuncOp>()) {
    if (fn.isExternal() || fn.getNumArguments() != arguments.size())
      continue;
    OpBuilder builder(fn.getContext());
    builder.setInsertionPointToStart(&fn.getBody().front());
    for (auto [arg, value] : llvm::zip(fn.getArguments(), arguments)) {
      Type type = arg.getType();
      TypedAttr constant;
      if (auto number = value.getAsInteger(); number && type.isIntOrIndex())
        constant = builder.getIntegerAttr(type, *number);
      else if (auto boolean = value.getAsBoolean();
               boolean && type.isInteger(1))
        constant = builder.getBoolAttr(*boolean);
      else if (auto number = value.getAsNumber();
               number && isa<FloatType>(type))
        constant = builder.getFloatAttr(type, *number);
      if (constant)
        arg.replaceAllUsesWith(
            builder.create<arith::ConstantOp>(fn.getLoc(), constant));
    }
  }
  PassManager pm(copy->getContext());
  pm.addPass(createSCCPPass());
  pm.addPass(createCanonicalizerPass());
  pm.addPass(createCSEPass());
  check(succeeded(pm.run(*copy)), "TTIR scalar specialization failed");
  return copy;
}

static int64_t ttirElements(Type type) {
  auto shaped = dyn_cast<ShapedType>(type);
  if (!shaped)
    return 1;
  check(shaped.hasStaticShape(), "dynamic tensor shape in TTIR cost model");
  return shaped.getNumElements();
}

static Type ttirElement(Type type) {
  if (auto shaped = dyn_cast<ShapedType>(type))
    type = shaped.getElementType();
  if (auto pointer = dyn_cast<mlir::triton::PointerType>(type))
    type = pointer.getPointeeType();
  return type;
}

static double ttirIterations(Operation *op, DataFlowSolver &solver,
                             bool &bounded) {
  auto bounds = [&](mlir::Value value) {
    APInt constant;
    if (matchPattern(value, m_ConstantInt(&constant)))
      return std::make_pair(constant.getSExtValue(), constant.getSExtValue());
    auto *state = solver.lookupState<dataflow::IntegerValueRangeLattice>(value);
    check(state && !state->getValue().isUninitialized(),
          "dynamic loop trip count in TTIR cost model");
    const auto &range = state->getValue().getValue();
    check(!(range == ConstantIntRanges::maxRange(range.smin().getBitWidth())),
          "dynamic loop trip count in TTIR cost model");
    bounded |= range.smin() != range.smax();
    return std::make_pair(range.smin().getSExtValue(),
                          range.smax().getSExtValue());
  };
  double count = 1;
  for (Operation *parent = op->getParentOp(); parent;
       parent = parent->getParentOp()) {
    if (auto loop = dyn_cast<scf::ForOp>(parent)) {
      auto lower = bounds(loop.getLowerBound()),
           upper = bounds(loop.getUpperBound()), step = bounds(loop.getStep());
      check(step.first > 0, "dynamic loop trip count in TTIR cost model");
      double trips =
          std::max(0.0, std::ceil((double(upper.second) - double(lower.first)) /
                                  double(step.first)));
      count *= trips;
      check(std::isfinite(count), "TTIR loop count overflow");
    }
    check(!isa<scf::WhileOp>(parent), "dynamic while loop in TTIR cost model");
  }
  return count;
}

static TtirWork ttirWork(ModuleOp module, const Profile &profile,
                         const A &grid) {
  TtirWork work;
  DenseMap<mlir::Value, double> ready;
  mlir::triton::ModuleAxisInfoAnalysis axes(module);
  DataFlowSolver solver;
  solver.load<dataflow::DeadCodeAnalysis>();
  solver.load<TtirRanges>(grid);
  check(succeeded(solver.initializeAndRun(module)),
        "TTIR integer range analysis failed");
  module.walk<WalkOrder::PreOrder>([&](Operation *op) {
    StringRef name = op->getName().getStringRef();
    if (name == "builtin.module" || name == "tt.func" || name == "scf.for" ||
        name == "scf.if")
      return WalkResult::advance();
    if (op->hasTrait<OpTrait::IsTerminator>())
      return WalkResult::skip();
    if (name == "tt.call" || name == "tt.dot_scaled" || name == "scf.while")
      throw std::runtime_error("unsupported TTIR cost operation: " +
                               name.str());
    double iterations = ttirIterations(op, solver, work.boundedLoops);
    auto record = [&](double throughput, double latency) {
      double begin = 0;
      for (mlir::Value value : op->getOperands())
        begin = std::max(begin, ready.lookup(value));
      double end = begin + iterations * latency;
      for (mlir::Value value : op->getResults())
        ready[value] = end;
      work.vector += iterations * throughput;
      work.vectorPath = std::max(work.vectorPath, end);
    };
    Type type;
    if (op->getNumResults())
      type = op->getResult(0).getType();
    else if (op->getNumOperands())
      type = op->getOperand(0).getType();
    if (!type)
      return WalkResult::advance();
    if (name == "tt.store")
      type = op->getOperand(1).getType();
    int64_t elements = ttirElements(type);
    Type element = ttirElement(type);
    unsigned bits =
        element.isIntOrFloat() ? element.getIntOrFloatBitWidth() : 64;
    unsigned lanes = std::max(1u, 2048u / std::max(8u, bits));
    if (name == "tt.load" || name == "tt.store") {
      auto *info = axes.getAxisInfo(op->getOperand(0));
      int64_t contiguous = info && !info->getContiguity().empty()
                               ? *llvm::max_element(info->getContiguity())
                               : 1;
      double bytes = elements * std::max(8u, bits) / 8.0;
      // Account for fragmented transactions, without pretending unknown
      // pointer strides are a contiguous full-bandwidth DMA.
      double packet = contiguous * std::max(8u, bits) / 8.0;
      bytes *= std::max(1.0, 32.0 / packet);
      if (name == "tt.load") {
        work.readBytes += iterations * bytes;
        work.loads += iterations;
      } else {
        work.writeBytes += iterations * bytes;
        work.stores += iterations;
      }
      return WalkResult::skip();
    }
    if (name == "tt.dot") {
      auto lhs = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto rhs = dyn_cast<RankedTensorType>(op->getOperand(1).getType());
      check(lhs && rhs && lhs.hasStaticShape() && rhs.hasStaticShape() &&
                lhs.getRank() >= 2 && rhs.getRank() == lhs.getRank(),
            "unsupported dot shape in TTIR cost model");
      int64_t m = lhs.getShape()[lhs.getRank() - 2];
      int64_t k = lhs.getShape().back(), n = rhs.getShape().back();
      double batch = double(lhs.getNumElements()) / (m * k);
      // Cube tile work; packing, conversion and scheduling are still estimates.
      work.cube += iterations * batch * std::ceil(m / 16.0) *
                   std::ceil(n / 16.0) * std::ceil(k / 16.0) * 16;
      return WalkResult::skip();
    }
    if (name == "tt.reduce" || name == "tt.scan") {
      auto input = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      check(input && input.hasStaticShape() && axis,
            "unknown aggregation domain in TTIR cost model");
      int64_t extent = input.getDimSize(axis.getInt());
      check(extent > 0, "empty aggregation domain");
      double groups = double(input.getNumElements()) / extent;
      double steps = std::ceil(std::log2(double(extent)));
      double combines = std::max<int64_t>(1, op->getNumOperands());
      double chunks = std::ceil(double(extent) / lanes);
      double throughput =
          groups * combines *
          (chunks +
           std::ceil(std::log2(double(std::min<int64_t>(extent, lanes)))));
      double latency =
          combines * (steps * Rate{}.latency + std::max(0.0, groups - 1));
      if (name == "tt.scan") {
        // A prefix scan retains every lane at every shuffle/combine step;
        // reduction shrinks the active domain and cannot share this cost.
        throughput = groups * combines * chunks * steps * 2;
        latency =
            combines * steps * (2 * Rate{}.latency + std::max(0.0, chunks - 1));
        // A5 cumsum lowers to a SIMT helper, independently for each batch
        // row (lowerBatchLaneScans). Retain its per-core UB round trip and
        // two 32-cycle mode transitions, as in the lowered SimtModel.
        work.simt += iterations * groups *
                     (64 + 2 * extent * std::max(8u, bits) / 8.0 /
                               profile.localBandwidth);
      }
      record(throughput, latency);
      return WalkResult::skip();
    }
    if (name == "arith.constant" || name == "tt.splat" ||
        name == "tt.make_range" || name == "tt.expand_dims" ||
        name == "tt.broadcast" || name == "tt.reshape" || name == "tt.trans" ||
        name == "tt.join" || name == "tt.split" || name == "tt.bitcast") {
      record(0, 0);
      return WalkResult::skip();
    }
    // A scalar result of a tensor reduction still executes on the Vector
    // core. Scalar floating-point SSA types are not AIC scalar instructions.
    if (!isa<RankedTensorType>(type) && !isa<FloatType>(type)) {
      work.scalar += iterations;
      record(0, 0);
      return WalkResult::advance();
    }
    if (name == "tt.extern_elementwise") {
      auto symbol = op->getAttrOfType<StringAttr>("symbol");
      check(bool(symbol), "external elementwise symbol missing");
      StringRef function = symbol.getValue();
      name = function.contains("tanh")   ? "math.tanh"
             : function.contains("exp2") ? "math.exp2"
             : function.contains("exp")  ? "math.exp"
             : function.contains("log2") ? "math.log2"
             : function.contains("log")  ? "math.log"
             : function.contains("pow")  ? "math.powf"
                                         : function;
    }
    if (auto shaped = dyn_cast<RankedTensorType>(type))
      if (op->getNumResults())
        if (auto *info = axes.getAxisInfo(op->getResult(0));
            info && info->getConstancy().size() == unsigned(shaped.getRank())) {
          // Broadcast index/mask arithmetic is computed per distinct value,
          // not once per replica in its tensor type.
          elements = 1;
          for (int i = 0; i < shaped.getRank(); ++i)
            elements *= (shaped.getDimSize(i) + info->getConstancy(i) - 1) /
                        info->getConstancy(i);
        }
    std::string opcode =
        llvm::StringSwitch<std::string>(name)
            .Cases("arith.addf", "arith.addi", "tt.addptr", "vadd")
            .Cases("arith.subf", "arith.subi", "vsub")
            .Cases("arith.mulf", "arith.muli", "vmul")
            .Cases("arith.divf", "arith.divsi", "arith.divui", "vdiv")
            .Case("math.exp", "vexp")
            .Case("math.log", "vln")
            .Case("math.sqrt", "vsqrt")
            .Case("math.rsqrt", "vrsqrt")
            .Case("arith.select", "vsel")
            .Cases("arith.maximumf", "arith.maxnumf", "arith.maxsi",
                   "arith.maxui", "vmax")
            .Cases("arith.minimumf", "arith.minnumf", "arith.minsi",
                   "arith.minui", "vmin")
            .Cases("arith.andi", "arith.ori", "arith.xori", "vadd")
            .Cases("arith.cmpf", "arith.cmpi", "vsel")
            .Cases("arith.extf", "arith.truncf", "arith.extsi", "arith.extui",
                   "vadd")
            .Cases("arith.trunci", "arith.sitofp", "arith.uitofp", "vadd")
            .Cases("arith.fptosi", "arith.fptoui", "arith.index_cast", "vadd")
            .Cases("arith.shli", "arith.shrsi", "arith.shrui", "vadd")
            .Cases("arith.remsi", "arith.remui", "vdiv")
            .Cases("math.absf", "math.absi", "vabs")
            .Case("arith.negf", "vneg")
            .Default("");
    std::string dtype =
        element.isBF16()
            ? "BF16"
            : (isa<FloatType>(element) ? "F" : "I") + std::to_string(bits);
    auto cost = [&](StringRef instruction) {
      return profile.cost(instruction, dtype);
    };
    SmallVector<StringRef> instructions;
    if (name == "math.tanh")
      instructions = {"vmul", "vexp", "vsub", "vadd", "vdiv"};
    else if (name == "math.exp2" || name == "math.log2")
      instructions = {name == "math.exp2" ? "vexp" : "vln", "vmul"};
    else if (name == "math.powf")
      instructions = {"vln", "vmul", "vexp"};
    else if (name == "math.fma")
      instructions = {"vmul", "vadd"};
    else {
      check(!opcode.empty(), "unsupported TTIR cost operation: " + name.str());
      instructions.push_back(opcode);
    }
    double throughput = 0, latency = 0;
    for (StringRef instruction : instructions) {
      Rate rate = cost(instruction);
      double repeats = std::ceil(double(elements) / lanes);
      throughput += repeats * rate.interval / rate.slots;
      latency += rate.latency +
                 std::max(0.0, repeats - 1) * rate.interval / rate.slots;
    }
    record(throughput, latency);
    return WalkResult::advance();
  });
  return work;
}
} // namespace mlir::triton::ascend::blockauto

namespace mlir::triton::ascend {
std::vector<unsigned> blockDataOnlyArguments(ModuleOp module, unsigned arity) {
  if (!llvm::hasSingleElement(*module.getBody()))
    return {};
  auto fn = dyn_cast<triton::FuncOp>(&module.getBody()->front());
  if (!fn || !fn.isPublic() || fn.getNumArguments() != arity ||
      !llvm::hasSingleElement(fn.getBody()))
    return {};

  // Follow every use, including both roles if a value is stored under itself
  // as the mask. An unrecognized consumer keeps the scalar exact.
  llvm::DenseSet<Operation *> dataConsumers;
  auto onlyDataUses = [&](Value value) {
    return llvm::all_of(value.getUses(), [&](OpOperand &use) {
      Operation *owner = use.getOwner();
      if (!dataConsumers.contains(owner))
        return false;
      auto store = dyn_cast<triton::StoreOp>(owner);
      return !store || &use == &store.getValueMutable();
    });
  };
  bool outputTail = true;
  for (Operation &op : llvm::reverse(fn.getBody().front())) {
    if (auto store = dyn_cast<triton::StoreOp>(op)) {
      if (outputTail && getElementTypeOrSelf(store.getValue()).isIntOrFloat())
        dataConsumers.insert(&op);
      continue;
    }
    if (isa<triton::ReturnOp>(op))
      continue;
    // A later read may observe the stored value through an alias.
    if (op.getNumRegions() || !isMemoryEffectFree(&op))
      outputTail = false;
    if (isa<triton::SplatOp, triton::BitcastOp, arith::BitcastOp,
            arith::ExtSIOp, arith::ExtUIOp, arith::TruncIOp, arith::AddIOp,
            arith::SubIOp, arith::MulIOp, arith::AndIOp, arith::OrIOp,
            arith::XOrIOp, arith::ShLIOp, arith::ShRSIOp, arith::ShRUIOp,
            arith::CmpIOp, arith::SelectOp>(op) &&
        onlyDataUses(op.getResult(0)))
      dataConsumers.insert(&op);
  }

  std::vector<unsigned> arguments;
  for (BlockArgument arg : fn.getArguments())
    if (isa<IntegerType>(arg.getType()) && onlyDataUses(arg))
      arguments.push_back(arg.getArgNumber());
  return arguments;
}

std::string scoreBlockTtir(ModuleOp module, llvm::StringRef request) {
  using namespace blockauto;
  auto parsed = json::parse(request);
  if (!parsed)
    throw std::runtime_error(toString(parsed.takeError()));
  const O &input = object(*parsed), &contract = sub(input, "contract");
  int64_t factor = integer(input, "factor", 1),
          axis = integer(input, "axis", -1);
  bool cv = flag(input, "cv");
  int64_t cores = integer(input, "cores");
  O output{{"factor", factor},
           {"axis", axis},
           {"resource_checked", false},
           {"model", "ttir-relative-v3"}};
  try {
    check(cores > 0, "invalid target core count");
    A grid(array(contract, "grid"));
    bool preserve = flag(input, "preserve_grid");
    if (factor > 1 && !preserve) {
      check(axis >= 0 && axis < 3, "invalid TTIR merge axis");
      grid[axis] = (integer(grid[axis]) - 1) / factor + 1;
    }
    int64_t blocks = product(grid);
    StringRef arch = input.getString("arch").value_or("");
    if (usesGroupingPolicy(arch)) {
      // The earlier generation uses TTIR legality and occupancy, not the A5
      // instruction costs priced below against an A5 register file and
      // unified buffer. Final allocation and launcher validation still run
      // on the selected binary.
      bool allowed =
          factor == 1 ||
          (!preserve && axis >= 0 && axis < 3 && (factor == 2 || factor == 4) &&
           integer(array(contract, "grid")[axis]) % factor == 0 &&
           blocks >= cores &&
           (!cv || str(input, "merge_kind") == "M" ||
            str(input, "merge_kind") == "N"));
      update(output, O{{"status", allowed ? "TTIR_COST_ESTIMATED"
                                          : "TTIR_MODEL_UNSUPPORTED"},
                       {"model", "ttir-legacy-grouping-v1"},
                       {"score", factor == 1 ? 5 : factor},
                       {"blocks", blocks},
                       {"cores", cores}});
      std::string result;
      llvm::raw_string_ostream stream(result);
      stream << json::Value(std::move(output));
      return result;
    }
    check(arch.starts_with("Ascend950") || arch.starts_with("Ascend910_95"),
          "unsupported automatic selection target");
    Profile profile(cv);
    auto specialized = specializeTtir(module, input);
    TtirWork work = ttirWork(*specialized, profile, grid);
    auto live = mlir::triton::cfg::estimatePeakLiveBytes(*specialized);
    // TTIR tensor liveness is a pressure estimate, never an actual UB proof.
    double pressure =
        live.known ? double(live.peakLiveBytes) / (248 * 1024) : 0;
    // Reuse the A5 lowered model's 32-cycle DMA and 8-cycle VF dispatch
    // costs. Do not assume loads, their dependent compute, and stores overlap.
    double memory =
        (work.readBytes + work.writeBytes) / profile.localBandwidth +
        (work.loads + work.stores) * 32;
    double compute =
        std::max(work.vector, work.vectorPath) + work.cube + work.simt + 8;
    double perBlock = memory + compute + work.scalar;
    double waves = std::ceil(double(blocks) / cores);
    double score =
        std::max(waves * perBlock, blocks * (work.readBytes + work.writeBytes) /
                                       profile.globalBandwidth);
    if (!cv) {
      // The VV per-core bandwidth is a share of the chip bandwidth, not a
      // private DMA limit. Charge traffic from actual blocks to this resource:
      // ceil(blocks / cores) would charge idle cores in the final wave.
      // Per-core compute, request startup and SIMT work still take full waves.
      double bandwidth =
          std::min(profile.globalBandwidth,
                   std::min(blocks, cores) * profile.localBandwidth);
      score = blocks * (work.readBytes + work.writeBytes) / bandwidth +
              waves * (32 * (work.loads + work.stores) + compute + work.scalar);
    }
    score *= 1 + std::pow(std::max(0.0, pressure - 0.5), 2);
    update(output, O{{"status", "TTIR_COST_ESTIMATED"},
                     {"score", score},
                     {"blocks", blocks},
                     {"waves", waves},
                     {"cores", cores},
                     {"vector_work", work.vector},
                     {"cube_work", work.cube},
                     {"vector_critical_path", work.vectorPath},
                     {"memory_cycles", memory},
                     {"compute_cycles", compute},
                     {"simt_work", work.simt},
                     {"loop_trip_upper_bound", work.boundedLoops},
                     {"scalar_work", work.scalar},
                     {"memory_requests", work.loads + work.stores},
                     {"bytes_per_program", work.readBytes + work.writeBytes},
                     {"live_bytes_known", live.known},
                     {"estimated_live_bytes", int64_t(live.peakLiveBytes)}});
  } catch (const std::exception &error) {
    update(output, O{{"status", "TTIR_MODEL_UNSUPPORTED"},
                     {"error", std::string(error.what())}});
  }
  std::string result;
  llvm::raw_string_ostream stream(result);
  stream << V(std::move(output));
  return result;
}
} // namespace mlir::triton::ascend

namespace mlir::triton::ascend {
std::string selectBlockCandidates(llvm::StringRef request) {
  using namespace blockauto;
  auto parsed = json::parse(request);
  if (!parsed)
    throw std::runtime_error(toString(parsed.takeError()));
  const O &input = object(*parsed);
  std::string action = str(input, "action");
  check(action == "plan" || action == "rank_ttir",
        "unknown TTIR selection action");
  O result = action == "plan" ? planCV(input) : rankTtir(input);
  std::string output;
  llvm::raw_string_ostream stream(output);
  stream << V(std::move(result));
  return output;
}
} // namespace mlir::triton::ascend
