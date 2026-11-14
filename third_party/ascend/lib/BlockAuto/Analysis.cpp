// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "BlockAuto/Analysis.h"
#include "bishengir/Dialect/HACC/IR/HACC.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "mlir/Analysis/DataLayoutAnalysis.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/ControlFlow/IR/ControlFlowOps.h"
#include "mlir/Dialect/DLTI/DLTI.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/ControlFlowInterfaces.h"
#include "mlir/Interfaces/DataLayoutInterfaces.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/AnalysisManager.h"
#include "mlir/Target/LLVMIR/Dialect/LLVMIR/LLVMIRToLLVMTranslation.h"
#include "mlir/Target/LLVMIR/Import.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/BinaryFormat/Dwarf.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/DIBuilder.h"
#include "llvm/IR/DebugInfo.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/Regex.h"
#include "llvm/Support/SourceMgr.h"
#include "llvm/Support/raw_ostream.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <deque>
#include <functional>
#include <initializer_list>
#include <limits>
#include <map>
#include <queue>
#include <set>
#include <stdexcept>
#include <string>
#include <tuple>
#include <variant>
#include <vector>

namespace mlir::triton::ascend {
llvm::json::Object analyzeBlockCandidate(const llvm::json::Object &config);
namespace blockauto {
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
inline int64_t product(const A &values) {
  int64_t n = 1;
  for (const V &v : values) {
    int64_t x = integer(v);
    check(x > 0 && n <= INT64_MAX / x, "nonpositive or overflowing extent");
    n *= x;
  }
  return n;
}
std::vector<std::string> matches(StringRef text, const Regex &pattern,
                                 unsigned group = 1);
class CandidateIRAnalysis;
bool preservesGrid(const O &metadata);
std::string rejection(const O &item, const O &contract,
                      const CandidateIRAnalysis &ir);
O modelConfig(const O &item, const O &contract, const A &arguments,
              const CandidateIRAnalysis &ir);
std::string mergeKind(const O &metadata, int64_t factor, bool frontend = false);
O planCV(const O &request);
O rankCV(const O &records, const O &contract);
bool excluded(StringRef status, bool vv = true);
} // namespace blockauto
} // namespace mlir::triton::ascend

namespace mlir::triton::ascend::blockauto {
// Allocation and lowering legality belong to the compiler passes.
class CandidateIRAnalysis {
public:
  CandidateIRAnalysis(Operation *op, AnalysisManager &manager);
  int64_t spec(StringRef key) const;
  void checkResources(const O &item) const;
  A launchArguments(const O &contract, const O &metadata) const;
  std::string metaLibrary() const;

private:
  ModuleOp module;
  const DataLayout &layout;
};
void registerCandidateDialects(DialectRegistry &registry);
OwningOpRef<ModuleOp> parseCandidateIR(StringRef source, MLIRContext &context);
} // namespace mlir::triton::ascend::blockauto

namespace mlir::triton::ascend {
namespace blockauto {
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
} // namespace blockauto
namespace {
using namespace llvm;
using Object = json::Object;
using Array = json::Array;
using Ref = std::pair<int, double>;
using Key = std::tuple<int, int, int>;
using blockauto::array;
using blockauto::check;
using blockauto::number;
using blockauto::object;
std::string text(const Object &o, StringRef key) {
  auto v = o.getString(key);
  check(bool(v), "missing string: " + key.str());
  return v->str();
}

struct Pointer {
  Operation *allocation = nullptr;
  int depth = 0;
  int64_t offset = 0;
  bool operator<(const Pointer &p) const {
    return std::tie(allocation, depth, offset) <
           std::tie(p.allocation, p.depth, p.offset);
  }
};
struct Scalar {
  std::variant<std::monostate, APInt, double, Pointer, std::vector<Scalar>>
      data;
  Scalar() = default;
  Scalar(APInt v) : data(std::move(v)) {}
  Scalar(int64_t v) : data(APInt(64, uint64_t(v))) {}
  Scalar(Pointer p) : data(p) {}
  Scalar(std::vector<Scalar> a) : data(std::move(a)) {}
  bool known() const { return data.index() != 0; }
  int64_t integer() const {
    auto *i = std::get_if<APInt>(&data);
    check(i && i->getBitWidth() <= 64, "unknown scalar integer");
    return i->getBitWidth() == 1 ? i->getZExtValue() : i->getSExtValue();
  }
  uint64_t bits() const {
    auto *i = std::get_if<APInt>(&data);
    check(i && i->getBitWidth() <= 64, "unknown scalar bit pattern");
    return i->getZExtValue();
  }
};
using Memory = std::map<Pointer, Scalar>;
using Environment = DenseMap<mlir::Value, Scalar>;
Scalar emptyAggregate(mlir::Type type) {
  std::vector<Scalar> values;
  if (auto s = dyn_cast<LLVM::LLVMStructType>(type))
    for (mlir::Type element : s.getBody())
      values.push_back(emptyAggregate(element));
  else if (auto a = dyn_cast<LLVM::LLVMArrayType>(type))
    values.assign(a.getNumElements(), emptyAggregate(a.getElementType()));
  else
    return {};
  return Scalar(std::move(values));
}
Scalar fromAttribute(Attribute attr) {
  if (auto integer = dyn_cast_or_null<IntegerAttr>(attr))
    return Scalar(integer.getValue());
  if (auto floating = dyn_cast_or_null<FloatAttr>(attr)) {
    Scalar result;
    double value = floating.getValueAsDouble();
    if (std::isfinite(value))
      result.data = value;
    return result;
  }
  if (auto dense = dyn_cast_or_null<DenseElementsAttr>(attr)) {
    std::vector<Scalar> values;
    for (auto value : dense.getValues<Attribute>())
      values.push_back(fromAttribute(value));
    return Scalar(std::move(values));
  }
  return {};
}
Attribute toAttribute(const Scalar &scalar, mlir::Type type) {
  if (auto *value = std::get_if<APInt>(&scalar.data))
    if (auto integer = dyn_cast<IntegerType>(type))
      return IntegerAttr::get(type, value->sextOrTrunc(integer.getWidth()));
  if (auto *value = std::get_if<double>(&scalar.data))
    if (isa<FloatType>(type))
      return FloatAttr::get(type, *value);
  return {};
}
Scalar insert(Scalar value, ArrayRef<int64_t> indices, const Scalar &element) {
  if (indices.empty())
    return element;
  auto *values = std::get_if<std::vector<Scalar>>(&value.data);
  if (!values)
    return {};
  check(indices.front() >= 0 && indices.front() < int64_t(values->size()),
        "invalid aggregate index");
  (*values)[indices.front()] =
      insert((*values)[indices.front()], indices.drop_front(), element);
  return value;
}
bool constant(Operation &op) {
  return isa<LLVM::ConstantOp, LLVM::UndefOp, LLVM::PoisonOp, LLVM::ZeroOp,
             LLVM::AddressOfOp>(op);
}
// LLVM constants become operations during import, but do not issue
// instructions.
bool scalarIntrinsic(Operation &op) {
  return op.hasAttr("block_auto.intrinsic") ||
         isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(op);
}
StringRef callName(LLVM::CallOp call) {
  auto name = call.getCallee().value_or("");
  return name.starts_with("a5.llvm.") ? name.drop_front(3) : name;
}
StringRef functionAttribute(LLVM::LLVMFuncOp fn, StringRef key) {
  if (auto attributes = fn.getPassthrough())
    for (Attribute attr : *attributes) {
      if (auto pair = dyn_cast<ArrayAttr>(attr)) {
        if (pair.size() == 2 && cast<StringAttr>(pair[0]).getValue() == key)
          return cast<StringAttr>(pair[1]).getValue();
      } else if (auto name = dyn_cast<StringAttr>(attr))
        if (name.getValue() == key)
          return key;
    }
  return {};
}
int callingConvention(LLVM::LLVMFuncOp fn) {
  int convention = 0;
  functionAttribute(fn, "block_auto_calling_conv").getAsInteger(10, convention);
  return convention;
}

struct Rate {
  double interval = 1, latency = 4;
  unsigned slots = 2;
  std::string unit = "vector_alu", pipe = "vector_alu";
};
struct Profile {
  std::map<std::string, Rate> costs, rates;
  mutable std::map<std::pair<std::string, bool>, Rate> instructionRates;
  std::set<std::string> covered;
  double localBandwidth = 32, globalBandwidth = 512;
  unsigned window = 64;
  void addRates(std::initializer_list<StringRef> operations,
                std::initializer_list<StringRef> types, Rate rate = {}) {
    for (StringRef op : operations) {
      covered.insert(op.str());
      for (StringRef type : types)
        rates[op.str() + ":" + type.str()] = rate;
    }
  }
  Profile(const Object &config) {
    // Preserve the existing 9579 model assumptions, not device measurements:
    // 56 Vector cores, 1650 MHz and 16 GB/s per core.
    if (text(config, "mode") == "vv") {
      check(number(config, "cores") == 56,
            "compiled core count differs from the A5 model");
      localBandwidth = 16.0 * 1e9 / (1650.0 * 1e6);
      globalBandwidth = localBandwidth * 56;
    }
#define AVE_COST_ENTRY(OP, SRC, DST, INTERVAL, LATENCY, SLOTS)                 \
  costs[#OP ":" #SRC ":" #DST] = {INTERVAL, LATENCY, SLOTS};
#include "bishengir/Dialect/HIVMAVE/CostModelInfo/AVECostModelInfo.def"
#undef AVE_COST_ENTRY
    // Fixed rates from the previously packaged instruction profile, revision
    // 55b286e9113551634e9bcda1bb552cefc6c8a772. Keep dtype exclusions explicit.
    addRates({"vadd", "vadds", "vsub", "vabs", "vneg"},
             {"int8", "int16", "int32", "int64"}, {2, 4, 2});
    addRates({"vmul", "vmuls"}, {"int16", "int32", "int64"}, {2, 4, 2});
    addRates({"vadd", "vadds", "vsub", "vmul"}, {"fp16", "bf16", "fp32"});
    addRates({"vmuls", "vabs", "vneg"}, {"fp16", "fp32"});
    addRates({"vmax", "vmaxs", "vmin", "vmins", "vsel", "vcmp"},
             {"int8", "fp16", "bf16", "int16", "fp32", "int32", "int64"});
    addRates({"vand", "vor", "vxor"}, {"int8", "int16", "int32", "int64"});
    addRates({"vnot"}, {"int8", "fp16", "int16", "fp32", "int32", "int64"});
    std::initializer_list<StringRef> broadTypes = {"int8", "fp16",  "bf16",
                                                   "fp8",  "fp4",   "int16",
                                                   "fp32", "int32", "int64"};
    addRates({"vdups"}, broadTypes);
    addRates({"vcvtff.bf162f32.x"}, {"bf16"});
    addRates({"vdiv", "vsqrt", "vexp", "vln"}, {"fp16"},
             {8, 4, 2, "vector_alu", "vector_transcendental"});
    Rate transcendental{4, 4, 2, "vector_alu", "vector_transcendental"};
    addRates({"vdiv", "vsqrt", "vexp", "vexpdif", "vln"}, {"fp32"},
             transcendental);
    addRates({"vdiv"}, {"int16", "int32", "int64"}, transcendental);
    addRates({"vexpdif"}, {"fp16"}, transcendental);
    Rate vector{1, 4, 1, "vector", "vector"};
    addRates({"vcadd", "vcmax"}, {"fp16", "int16", "fp32", "int32", "int64"},
             vector);
    addRates({"vbr"}, broadTypes, vector);
  }
  Rate cost(StringRef name, bool ready) const {
    auto key = std::make_pair(name.str(), ready);
    auto found = instructionRates.find(key);
    if (found == instructionRates.end())
      found = instructionRates.emplace(key, instructionCost(name, ready)).first;
    return found->second;
  }
  Rate instructionCost(StringRef name, bool ready) const {
    std::string op =
        name.drop_front(strlen("llvm.hivm.")).split('.').first.str();
    Rate result;
    if (op == "vdiv" || op == "vsqrt" || op == "vrsqrt")
      result = {4, 14, 2};
    if (op == "vexp")
      result = {4, 13, 2};
    if (op == "vln")
      result = {4, 15, 2};
    static const Regex vectorType(R"(\.v[0-9]+(f[0-9]+|i[0-9]+|bf16))");
    auto types = blockauto::matches(name, vectorType);
    for (auto &type : types) {
      std::transform(type.begin(), type.end(), type.begin(), ::toupper);
    }
    static const std::set<std::string> operations{
        "vadd", "vadds",   "vsub", "vmul", "vmuls", "vdiv", "vsqrt",
        "vexp", "vexpdif", "vln",  "vmax", "vmaxs", "vmin", "vmins",
        "vabs", "vneg",    "vand", "vor",  "vxor",  "vnot"};
    std::string opcode;
    if (operations.count(op)) {
      opcode = op.substr(1);
      opcode[0] = toupper(opcode[0]);
      opcode = "VF" + opcode + "Op";
    }
    if (op == "vsel")
      opcode = "VFSelectOp";
    if (op == "vintlv")
      opcode = "VFInterleaveOp";
    if (op == "vdintlv")
      opcode = "VFDeInterleaveOp";
    std::string dst = types.empty() ? "Unknown" : types[0],
                src = types.size() > 1 ? types[1] : dst;
    auto found = costs.find(opcode + ":" + src + ":" + dst);
    if (found != costs.end())
      result = found->second;
    if (StringRef(op).starts_with("vld"))
      return {1, 4, 2, "load", "load"};
    if (StringRef(op).starts_with("vst"))
      return {1, 4, 1, "store", "store"};
    if (!ready)
      return result;
    std::string dtype;
    if (name == "llvm.hivm.vcvtff.bf162f32.x") {
      op = "vcvtff.bf162f32.x";
      dtype = "bf16";
    } else if (covered.count(op)) {
      static const Regex pattern(R"(\.v[0-9]+(bf16|f[0-9]+|i[0-9]+|s[0-9]+))");
      for (const auto &type : blockauto::matches(name, pattern)) {
        dtype = type;
        if (dtype != "i1")
          break;
        dtype.clear();
      }
      if (StringRef(dtype).starts_with("f"))
        dtype = "fp" + dtype.substr(1);
      if (StringRef(dtype).starts_with("i") ||
          StringRef(dtype).starts_with("s"))
        dtype = "int" + dtype.substr(1);
    }
    if (!dtype.empty()) {
      auto rate = rates.find(op + ":" + dtype);
      check(rate != rates.end(),
            "unsupported instruction dtype: " + name.str());
      double latency = result.latency;
      result = rate->second;
      result.latency = latency;
    }
    return result;
  }
};

struct VectorPlan {
  struct Node {
    std::set<int> deps;
    Rate rate;
    double earliest = 0;
    bool executes = true;
  };
  const Profile &profile;
  bool readyMode;
  DenseMap<mlir::Value, std::set<int>> dependencies;
  DenseMap<mlir::Value, double> times;
  std::map<std::string, std::vector<double>> legacySlots{
      {"exec", {0, 0}}, {"load", {0, 0}}, {"store", {0}}};
  std::vector<Node> nodes;
  std::vector<int> retirement;
  std::set<int> control;
  std::vector<double> pending;
  double issue = 0, finish = 0;
  VectorPlan(const Profile &p, bool ready) : profile(p), readyMode(ready) {}
  static OperandRange readOperands(Operation &i) {
    // CF forwards loop-carried SSA values on edges. They are phi inputs, not
    // operands consumed by the branch instruction's issue resource.
    return i.getOperands().take_front(isa<cf::BranchOp>(i) ? 0
                                      : isa<cf::CondBranchOp>(i)
                                          ? 1
                                          : i.getNumOperands());
  }
  std::set<int> deps(Operation &i) const {
    std::set<int> result;
    for (mlir::Value operand : readOperands(i))
      if (auto at = dependencies.find(operand); at != dependencies.end())
        result.insert(at->second.begin(), at->second.end());
    return result;
  }
  double dependencyTime(Operation &i) {
    double result = 0;
    for (mlir::Value operand : readOperands(i))
      result = std::max(result, times[operand]);
    return result;
  }
  int node(Operation &i, Rate rate) {
    auto parents = deps(i);
    parents.insert(control.begin(), control.end());
    size_t ordinal = retirement.size();
    if (ordinal >= profile.window)
      parents.insert(retirement[ordinal - profile.window]);
    int id = nodes.size();
    nodes.push_back({parents, rate, ordinal * 0.5, true});
    std::set<int> retired{id};
    if (!retirement.empty())
      retired.insert(retirement.back());
    retirement.push_back(nodes.size());
    nodes.push_back({retired, {}, 0, false});
    if (i.getNumResults() != 0)
      dependencies[i.getResult(0)] = {id};
    return id;
  }
  void add(LLVM::CallOp call) {
    auto rate = profile.cost(callName(call), readyMode);
    if (readyMode) {
      node(*call, rate);
      return;
    }
    std::string pipe =
        rate.unit == "load" || rate.unit == "store" ? rate.unit : "exec";
    auto &slots = legacySlots[pipe];
    auto slot = std::min_element(slots.begin(), slots.begin() + rate.slots);
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [&](double t) { return t <= issue; }),
                  pending.end());
    std::sort(pending.begin(), pending.end());
    double queued = pending.size() >= profile.window ? pending.front() : 0;
    double start = std::max({issue, *slot, dependencyTime(*call), queued});
    *slot = start + rate.interval;
    issue = start + 0.5;
    finish = std::max(finish, start + rate.latency);
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [&](double t) { return t <= start; }),
                  pending.end());
    pending.push_back(start + rate.latency);
    if (call->getNumResults())
      times[call.getResult()] = start + rate.latency;
  }
  void scalar(Operation &i) {
    if (i.getNumResults()) {
      if (readyMode)
        dependencies[i.getResult(0)] = deps(i);
      else
        times[i.getResult(0)] = dependencyTime(i);
    }
    if (!readyMode ||
        isa<LLVM::CallOp, LLVM::AllocaOp, LLVM::LoadOp, LLVM::StoreOp,
            LLVM::BitcastOp, LLVM::ExtractValueOp, LLVM::InsertValueOp>(i) ||
        scalarIntrinsic(i))
      return;
    int id = node(i, {1, 1, 1, "vf_control", "vf_control"});
    if (isa<BranchOpInterface>(i))
      control = {id};
  }
  double complete() {
    if (!readyMode)
      return finish;
    std::vector<std::vector<int>> children(nodes.size());
    std::vector<int> remaining(nodes.size());
    std::vector<double> earliest(nodes.size());
    using Ready = std::pair<double, int>;
    std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> queue;
    for (unsigned i = 0; i < nodes.size(); ++i) {
      remaining[i] = nodes[i].deps.size();
      earliest[i] = nodes[i].earliest;
      for (int parent : nodes[i].deps)
        children[parent].push_back(i);
      if (!remaining[i])
        queue.emplace(earliest[i], i);
    }
    std::map<std::string, double> units;
    std::map<std::pair<std::string, std::string>, std::vector<double>> pipes;
    while (!queue.empty()) {
      auto [time, i] = queue.top();
      queue.pop();
      const auto &n = nodes[i];
      double end = time;
      if (n.executes) {
        auto &slots = pipes
                          .try_emplace({n.rate.unit, n.rate.pipe},
                                       std::vector<double>{0, 0})
                          .first->second;
        auto slot =
            std::min_element(slots.begin(), slots.begin() + n.rate.slots);
        double start = std::max({time, issue, units[n.rate.unit], *slot});
        if (!queue.empty() && start > queue.top().first) {
          queue.emplace(start, i);
          continue;
        }
        *slot = start + n.rate.interval;
        units[n.rate.unit] = start + 1.0 / n.rate.slots;
        issue = start + 0.5;
        end = start + n.rate.latency;
        finish = std::max(finish, end);
      }
      for (int child : children[i]) {
        earliest[child] = std::max(earliest[child], end);
        if (!--remaining[child])
          queue.emplace(earliest[child], child);
      }
    }
    return finish;
  }
};

struct Event {
  std::string kind;
  std::vector<Ref> deps;
  double duration = 0, bytes = 0, startup = 0;
};
using Args = ArrayRef<Scalar>;
int64_t integer(Args args, unsigned i) {
  check(i < args.size(), "helper argument missing");
  return args[i].integer();
}
double extent(Args args) {
  double size = 1;
  for (const Scalar &arg : args) {
    int64_t n = arg.integer();
    check(n >= 0, "negative helper extent");
    size *= n;
  }
  check(std::isfinite(size), "helper extent overflow");
  return size;
}
struct Plan {
  virtual ~Plan() = default;
  virtual void step() = 0;
  virtual void vf(double cycles) = 0;
  virtual void dma(bool load, double bytes) = 0;
  virtual void flag(bool set, int src, int dst, int id) = 0;
  virtual void barrier(int pipe) = 0;
  virtual bool extra(StringRef name, Args args) = 0;
};
struct KernelPlan : Plan {
  const Profile &profile;
  std::vector<Event> events;
  std::map<int, Ref> pipes{
      {0, {-1, 0}}, {1, {-1, 0}}, {4, {-1, 0}}, {5, {-1, 0}}};
  Ref scalar{-1, 0};
  std::map<Key, Ref> flags;
  bool ndPad = false;
  KernelPlan(const Profile &p) : profile(p) {}
  Ref add(std::string kind, std::vector<Ref> refs, double duration = 0,
          double bytes = 0, double startup = 0) {
    std::map<int, double> deps;
    for (auto [id, delay] : refs)
      deps[id] = std::max(deps[id], delay);
    Ref result{events.size(), 0};
    events.push_back({std::move(kind),
                      {deps.begin(), deps.end()},
                      duration,
                      bytes,
                      startup});
    return result;
  }
  void step() override {
    scalar.second += 1;
    pipes[0] = scalar;
  }
  void vf(double cycles) override {
    pipes[1] = add("vector", {scalar, pipes[1]}, cycles + 8);
  }
  void dma(bool load, double size) override {
    int pipe = load ? 4 : 5;
    pipes[pipe] = add("dma", {scalar, pipes[pipe]},
                      32 + size / profile.localBandwidth, size, 32);
  }
  void flag(bool set, int src, int dst, int id) override {
    check(pipes.count(src) && pipes.count(dst),
          "unmodelled synchronization pipe");
    Key key{src, dst, id};
    if (set)
      flags[key] = add("set", {scalar, pipes[src]});
    else {
      check(flags.count(key), "wait without statically observed set");
      pipes[dst] = add("wait", {scalar, pipes[dst], flags.at(key)}, 1);
      flags.erase(key);
      if (dst == 0)
        scalar = pipes[0];
    }
  }
  std::vector<Ref> all() const {
    std::vector<Ref> refs{scalar};
    for (auto [pipe, ref] : pipes)
      refs.push_back(ref);
    return refs;
  }
  void barrier(int) override { scalar = add("barrier", all()); }
  bool extra(StringRef name, Args args) override {
    if (name == "llvm.hivm.MOV.UB.TO.UB.v310") {
      check(args.size() == 3, "UB DMA ABI mismatch");
      uint64_t word = args[2].bits(), count = word & 65535,
               length = (word >> 16) & 65535;
      check(count && length, "zero UB DMA burst");
      vf(std::ceil(double(count * length * 32) / 256));
    } else if (name.starts_with("llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2.") ||
               name == "llvm.hivm.MOV.UB.TO.OUT.ALIGN.V2.DV") {
      static const std::set<std::string> types{"f32", "f16", "bf16",
                                               "u8",  "s8",  "u32"};
      bool load = name.contains(".OUT.TO.UB.");
      auto type = name.drop_front(strlen("llvm.hivm.MOV.OUT.TO.UB.ALIGN.V2."))
                      .split('.')
                      .first;
      check(args.size() == 4 &&
                (!load || (types.count(type.str()) && name.ends_with(".DV"))),
            "unknown DMA ABI");
      uint64_t first = args[2].bits();
      args[3].bits();
      uint64_t mask = (uint64_t(0xfff) << 4) | (uint64_t(0x1fffff) << 25) |
                      (load ? uint64_t(1) << 58 : 0);
      check(!(first & ~mask), "unsupported DMA control flags");
      uint64_t count = (first >> 4) & 0xfff, length = (first >> 25) & 0x1fffff;
      check(count && length, "zero DMA burst");
      dma(load, double(count * length));
    } else if (name.starts_with("llvm.hivm.ND.DMA.OUT.TO.UB.b")) {
      check(ndPad && args.size() == 4, "NDDMA padding or ABI unavailable");
      unsigned bits = 0;
      check(!name.drop_front(strlen("llvm.hivm.ND.DMA.OUT.TO.UB.b"))
                    .getAsInteger(10, bits) &&
                (bits == 8 || bits == 16 || bits == 32),
            "unknown NDDMA dtype");
      uint64_t first = args[2].bits(), second = args[3].bits();
      check(!(first & 15) &&
                !(second & ~((uint64_t(1) << 40) - 1 | uint64_t(1) << 56)),
            "unsupported NDDMA flags");
      double bytes = bits / 8;
      for (unsigned shift : {4, 24, 44})
        bytes *= (first >> shift) & 0xfffff;
      for (unsigned shift : {0, 20})
        bytes *= (second >> shift) & 0xfffff;
      check(bytes > 0, "zero NDDMA extent");
      dma(true, bytes);
    } else if (name == "llvm.hivm.SET.PAD.CNT.NDDMA") {
      check(args.size() == 1 && integer(args, 0) == 0, "nonzero NDDMA padding");
      ndPad = true;
    } else if (name == "llvm.hivm.SET.MOV.PAD.VAL" ||
               name == "llvm.hivm.SET.PAD.VAL.NDDMA" ||
               (name.starts_with("llvm.hivm.SET.LOOP") &&
                name.size() == strlen("llvm.hivm.SET.LOOP0.STRIDE.NDDMA") &&
                name[18] >= '0' && name[18] <= '4' &&
                name.ends_with(".STRIDE.NDDMA"))) {
      check(args.size() == 1, "unknown DMA setup ABI");
    } else if (!name.starts_with("llvm.lifetime.start.") &&
               !name.starts_with("llvm.lifetime.end."))
      return false;
    return true;
  }
  void complete() { add("end", all()); }
};

struct CVGraph {
  struct Node {
    double duration;
    std::set<int> parents;
  };
  std::vector<Node> nodes;
  std::map<Key, std::vector<int>> sets, waits;
  int add(double duration, std::vector<int> parents) {
    std::set<int> deps(parents.begin(), parents.end());
    deps.erase(-1);
    int id = nodes.size();
    nodes.push_back({duration, std::move(deps)});
    return id;
  }
  double finish() {
    for (const auto &[key, list] : waits)
      check(sets.count(key), "cross-core wait without set");
    for (const auto &[key, list] : sets) {
      check(list.size() == waits[key].size(), "unmatched cross-core flag");
      for (unsigned i = 0; i < list.size(); ++i)
        nodes[waits[key][i]].parents.insert(list[i]);
    }
    std::vector<double> ends(nodes.size());
    std::vector<unsigned> remaining(nodes.size());
    std::vector<std::vector<int>> children(nodes.size());
    std::queue<int> ready;
    for (unsigned i = 0; i < nodes.size(); ++i) {
      remaining[i] = nodes[i].parents.size();
      for (int parent : nodes[i].parents)
        children.at(parent).push_back(i);
      if (!remaining[i])
        ready.push(i);
    }
    double result = 0;
    unsigned completed = 0;
    while (!ready.empty()) {
      int id = ready.front();
      ready.pop();
      ++completed;
      ends[id] += nodes[id].duration;
      result = std::max(result, ends[id]);
      for (int child : children[id]) {
        ends[child] = std::max(ends[child], ends[id]);
        if (!--remaining[child])
          ready.push(child);
      }
    }
    check(completed == nodes.size(), "cyclic CV dependencies");
    return result;
  }
};
struct CVPlan : Plan {
  CVGraph &graph;
  int role, scalar = -1, pingpong = 0;
  double read = 0, write = 0, flops = 0, mmas = 0;
  std::map<int, int> pipes{{0, -1}, {1, -1}, {2, -1}, {3, -1},
                           {4, -1}, {5, -1}, {10, -1}},
      unitMma, unitFix;
  std::map<Key, std::deque<int>> flags;
  CVPlan(CVGraph &graph, int role) : graph(graph), role(role) {}
  int op(int pipe, double duration, std::vector<int> parents = {}) {
    check(pipes.count(pipe), "unknown CV pipe");
    parents.insert(parents.end(), {scalar, pipes[pipe]});
    int id = graph.add(duration, std::move(parents));
    pipes[pipe] = id;
    if (pipe == 0)
      scalar = id;
    return id;
  }
  void step() override { op(0, 1); }
  void vf(double cycles) override { op(1, cycles + 8); }
  void dma(bool load, double size) override {
    op(load ? 4 : 5, 32 + size / 32);
    (load ? read : write) += size;
  }
  void flag(bool set, int src, int dst, int id) override {
    auto &queue = flags[{src, dst, id}];
    if (set)
      queue.push_back(op(src, 1));
    else {
      check(!queue.empty(), "CV local flag wait without set");
      op(dst, 1, {queue.front()});
      queue.pop_front();
    }
  }
  void barrier(int pipe) override {
    std::vector<int> parents;
    if (pipe == 6)
      for (auto [p, id] : pipes)
        parents.push_back(id);
    op(pipe == 6 ? 0 : pipe, 0, std::move(parents));
  }
  void mma(Args a, bool batch) {
    check(role == -1 && a.size() == (batch ? 52 : 46), "unsupported MMA ABI");
    int offset = batch ? 4 : 0;
    int64_t m = integer(a, 23 + offset), k = integer(a, 24 + offset),
            n = integer(a, 25 + offset);
    check(std::min({m, k, n}) > 0, "nonpositive MMA size");
    int flagOffset = batch ? 6 : 0;
    int wa = integer(a, 37 + flagOffset), wb = integer(a, 38 + flagOffset);
    int ra = integer(a, 39 + flagOffset), rb = integer(a, 40 + flagOffset);
    for (unsigned i = 41 + flagOffset; i < 44 + flagOffset; ++i)
      check(integer(a, i) == -1, "alternate MMA protocol");
    int unit = integer(a, 44 + flagOffset), group = integer(a, 45 + flagOffset);
    if (batch) {
      int64_t count = integer(a, 3);
      check(count >= 2 && count == integer(a, 16) && count == integer(a, 33),
            "unsupported batch dimensions");
      check(!(m % 16 || k % 8 || n % 16 || unit),
            "unsupported batch tile or unit flag");
      int64_t chunk = std::min({65536 / (4 * m * k), 65536 / (4 * k * n),
                                262144 / (4 * m * n), count});
      check(chunk > 0, "batch matrix exceeds L0");
      flag(false, 2, 3, 0);
      for (int event : {wa, wb})
        if (event >= 0)
          flag(false, 4, 3, event);
      for (int64_t start = 0; start < count; start += chunk) {
        int64_t size = std::min(chunk, count - start);
        for (double bytes :
             {double(size * m * k * 4), double(size * k * n * 4)})
          op(3, 8 + bytes / 64);
        flag(true, 3, 2, 0);
        flag(false, 3, 2, 0);
        for (int64_t i = 0; i < size; ++i) {
          op(2, 16 + double(2 * m * k * n) / 2048, {pipes[3]});
          ++mmas;
          flops += 2 * m * k * n;
        }
        flag(true, 2, 3, 0);
        if (start + size < count)
          flag(false, 2, 3, 0);
      }
      for (int event : {ra, rb})
        if (event >= 0)
          flag(true, 3, 4, event);
      return;
    }
    int64_t pm = (m + 15) / 16 * 16, pn = (n + 15) / 16 * 16;
    int64_t tile = std::min(8192 / pm / 8 * 8, 8192 / pn / 8 * 8) / 16 * 16;
    check(tile && (unit == 0 || unit == 3),
          "unsupported MMA buffer or unit mode");
    check(!unit || (extent(a.slice(30, 2)) == m &&
                    integer(a, 29) * integer(a, 32) == n),
          "partial L0C update");
    for (int64_t start = 0; start < k; start += tile) {
      int64_t actual = std::min(tile, k - start), pk = (actual + 7) / 8 * 8;
      int event = pingpong;
      pingpong ^= 1;
      flag(false, 2, 3, event);
      for (auto [bytes, wait, release] :
           {std::tuple{pm * pk * 4, wa, ra}, std::tuple{pk * pn * 4, wb, rb}}) {
        if (start == 0 && wait >= 0)
          flag(false, 4, 3, wait);
        op(3, 8 + double(bytes) / 64);
        if (start + actual == k && release >= 0)
          flag(true, 3, 4, release);
      }
      flag(true, 3, 2, 0);
      flag(false, 3, 2, 0);
      std::vector<int> deps{pipes[3]};
      if (unit && unitFix.count(group))
        deps.push_back(unitFix[group]);
      int id = op(2, 16 + double(2 * pm * pk * pn) / 2048, deps);
      if (unit && start + actual == k)
        unitMma[group] = id;
      flag(true, 2, 3, event);
      ++mmas;
      flops += 2 * m * actual * n;
    }
  }
  bool extra(StringRef name, Args a) override {
    if (name == "llvm.hivm.SET.INTRA.BLOCKI.mode" ||
        name == "llvm.hivm.WAIT.INTRA.BLOCKI.mode") {
      check(a.size() == 2, "cross-core flag ABI");
      bool set = name.contains(".SET.");
      int id = integer(a, 1), subblock = role < 0 ? id / 16 : role,
          local = role < 0 ? id % 16 : id;
      check(id >= 0 && id < (role < 0 ? 32 : 16), "cross-core flag range");
      Key key{int((role < 0) == set), subblock, local};
      (set ? graph.sets : graph.waits)[key].push_back(op(integer(a, 0), 4));
    } else if (name == "nd2nz_batch_float") {
      check(role == -1 && a.size() == 23, "batch ND2NZ ABI");
      int64_t batch = integer(a, 3), rows = integer(a, 4), cols = integer(a, 5);
      int64_t stride = integer(a, 6), row = integer(a, 7), dst = integer(a, 17);
      check(std::min({batch, rows, cols, stride, row, dst}) > 0 &&
                integer(a, 8) == 1 && integer(a, 12) == batch &&
                extent(a.slice(12, 5)) > 0,
            "batch ND2NZ layout");
      bool fits =
          std::max({batch, row, stride, stride * 4, dst, dst / 8}) < 65536;
      int64_t commands = fits ? 1 : batch * (row < 65536 ? 1 : rows);
      double source = double(batch) * rows * cols * 4,
             padded = extent(a.slice(12, 5)) * 4;
      for (int64_t i = 0; i < commands; ++i)
        op(4, 32 + std::max(source, padded) / commands / 32);
      read += source;
    } else if (name == "nd2nz_float" || name == "set_l1_2d_float") {
      bool fill = name == "set_l1_2d_float";
      check(role == -1 && a.size() == (fill ? 6 : 19), "L1 transfer ABI");
      double src = extent(fill ? a.slice(4, 1) : a.slice(3, 2)) * 4;
      op(4, 32 + (fill ? src : std::max(src, extent(a.slice(10, 4)) * 4)) / 32);
      if (!fill)
        read += src;
    } else if (name == "mma_tile_float_to_float" ||
               name == "mma_tile_float_to_float_ta" ||
               name == "mma_tile_float_to_float_tb" ||
               name == "batch_mma_tile_float_to_float" ||
               name == "batch_mma_tile_float_to_float_ta" ||
               name == "batch_mma_tile_float_to_float_tb" ||
               name == "batch_mma_tile_float_to_float_ta_tb") {
      mma(a, name.starts_with("batch_"));
    } else if (name == "fixpipe_nz2nd_float_to_float_5d_to_3d_gm" ||
               name == "fixpipe_nz2nd_float_to_float_5d_to_3d_ubuf") {
      bool ub = name.ends_with("_ubuf");
      check(role == -1 && a.size() == (ub ? 30 : 29) && integer(a, 27) == 0,
            "batch Fixpipe ABI or unit mode");
      int64_t batch = integer(a, 16), m = integer(a, 17), n = integer(a, 18),
              src = integer(a, 8), c0 = integer(a, 11), dst = integer(a, 19),
              row = integer(a, 20);
      check(std::min({batch, m, n, src, c0, dst, row}) > 0 && src % c0 == 0,
            "batch Fixpipe layout");
      bool fits = std::max({batch, src / c0, dst}) <= 65535;
      check(!ub || fits, "batch Fixpipe descriptor overflow");
      int64_t commands = fits ? 1 : batch;
      double size = double(batch) * m * (ub ? std::max(n, row) : n) * 4;
      for (int64_t i = 0; i < commands; ++i)
        op(10, 16 + size / commands / 64);
      if (!ub)
        write += size;
    } else if (name == "fixpipe_nz2nd_dual_float_to_float_4d_to_2d_ubuf" ||
               name == "fixpipe_nz2nd_float_to_float_4d_to_2d_gm") {
      bool dual = name.contains("_dual_");
      check(role == -1 && a.size() == (dual ? 26 : 25) &&
                (!dual || integer(a, 25) == 1),
            "Fixpipe ABI");
      int unit = integer(a, 23), group = integer(a, 24);
      check(unit == 0 || unit == 3, "partial Fixpipe unit flag");
      std::vector<int> deps;
      if (unit) {
        check(unitMma.count(group), "Fixpipe without preceding unit MMA");
        deps.push_back(unitMma[group]);
      }
      double size = extent(a.slice(14, 2)) * 4 * (dual ? 2 : 1);
      int id = op(10, 16 + size / 64, deps);
      if (unit)
        unitFix[group] = id;
      if (!dual)
        write += size;
    } else
      return false;
    return true;
  }
  void complete() {
    for (const auto &[key, queue] : flags)
      check(queue.empty(), "unconsumed local CV flags");
  }
};

double schedule(const std::vector<KernelPlan> &blocks, unsigned cores,
                double bandwidth) {
  struct State {
    std::map<int, std::vector<Ref>> children;
    std::vector<int> remaining;
    std::vector<double> earliest;
    int core = 0;
  };
  std::vector<State> states(blocks.size());
  for (unsigned b = 0; b < blocks.size(); ++b) {
    for (unsigned i = 0; i < blocks[b].events.size(); ++i) {
      const auto &event = blocks[b].events[i];
      states[b].remaining.push_back(event.deps.size());
      states[b].earliest.push_back(0);
      for (auto [parent, delay] : event.deps)
        states[b].children[parent].emplace_back(i, delay);
    }
  }
  using Ready = std::tuple<double, unsigned, unsigned>;
  std::priority_queue<Ready, std::vector<Ready>, std::greater<Ready>> ready;
  auto release = [&](unsigned block, int parent, double finish) {
    auto &state = states[block];
    for (auto [child, delay] : state.children[parent]) {
      state.earliest[child] = std::max(state.earliest[child], finish + delay);
      if (!--state.remaining[child])
        ready.emplace(state.earliest[child], block, child);
    }
  };
  unsigned next = 0;
  double dispatch = 0, issuer = 0, bus = 0, finish = 0;
  auto start = [&](unsigned core, double available) {
    if (next == blocks.size())
      return;
    unsigned block = next++;
    double time = std::max(available, dispatch);
    dispatch = time + 32;
    states[block].core = core;
    release(block, -1, time);
  };
  for (unsigned i = 0; i < std::min<size_t>(cores, blocks.size()); ++i)
    start(i, 0);
  while (!ready.empty()) {
    auto [time, block, i] = ready.top();
    ready.pop();
    const auto &event = blocks[block].events[i];
    double end = time + event.duration;
    if (event.kind == "dma") {
      double issue = std::max(time, issuer);
      issuer = issue + 8;
      bus = std::max(issue + event.startup, bus) + event.bytes / bandwidth;
      end = std::max(issue + event.duration, bus);
    } else if (event.kind == "simt_helper" && event.bytes > 0) {
      bus = std::max(time + event.startup, bus) + event.bytes / bandwidth;
      end = std::max(end, bus);
    }
    if (event.kind == "end") {
      finish = std::max(finish, end);
      start(states[block].core, end);
    }
    release(block, i, end);
  }
  return finish;
}

OwningOpRef<ModuleOp> parse(StringRef input, MLIRContext &context) {
  std::string source = input.str();
  for (auto [from, to] :
       {std::pair{" ptc_simdvf ", " cc 1023 "},
        std::pair{" simt_entry ", " cc 1022 "},
        std::pair{" simd_vf_outlined ", " \"simd_vf_outlined\" "}}) {
    size_t pos = 0;
    while ((pos = source.find(from, pos)) != std::string::npos) {
      source.replace(pos, strlen(from), to);
      pos += strlen(to);
    }
  }
  llvm::LLVMContext llvmContext;
  SMDiagnostic diagnostic;
  auto module = parseAssemblyString(source, diagnostic, llvmContext);
  check(bool(module),
        "LLVM analysis parse failed: " + diagnostic.getMessage().str());
  StripDebugInfo(*module);
  // Import materializes constant aggregates as SSA operations. Mark original
  // instructions through the standard debug-location importer, so those extra
  // operations never become issued work in the cost model.
  DIBuilder debug(*module);
  auto *file = debug.createFile("block-auto-instruction", "");
  debug.createCompileUnit(dwarf::DW_LANG_C, file, "block-auto", false, "", 0);
  auto *type = debug.createSubroutineType(debug.getOrCreateTypeArray({}));
  // Upstream MLIR does not register BiSheng's private intrinsics/conventions.
  // Keep them as ordinary calls plus an ABI attribute in this analysis copy.
  for (auto &fn : *module) {
    auto *scope = fn.isDeclaration()
                      ? nullptr
                      : debug.createFunction(file, fn.getName(), fn.getName(),
                                             file, 1, type, 1, DINode::FlagZero,
                                             DISubprogram::SPFlagDefinition);
    if (scope)
      fn.setSubprogram(scope);
    if (fn.getName().starts_with("llvm.") && !fn.getIntrinsicID())
      fn.setName("a5." + fn.getName());
    if (fn.getCallingConv() >= 1022) {
      fn.addFnAttr("block_auto_calling_conv",
                   std::to_string(fn.getCallingConv()));
      fn.setCallingConv(llvm::CallingConv::C);
    }
    for (auto &block : fn)
      for (auto &op : block) {
        SmallVector<std::pair<unsigned, MDNode *>> metadata;
        op.getAllMetadataOtherThanDebugLoc(metadata);
        for (auto [kind, value] : metadata)
          op.setMetadata(kind, nullptr);
        op.setDebugLoc(DILocation::get(llvmContext, 1, 0, scope));
        if (auto *call = dyn_cast<llvm::CallBase>(&op))
          if (call->getCallingConv() >= 1022)
            call->setCallingConv(llvm::CallingConv::C);
      }
  }
  debug.finalize();
  auto result = translateLLVMIRToModule(std::move(module), &context, false);
  check(bool(result), "cannot import candidate LLVM into MLIR");
  result->walk([](Operation *op) {
    auto loc = op->getLoc()->findInstanceOf<FileLineColLoc>();
    if (!loc || loc.getFilename() != "block-auto-instruction")
      op->setAttr("block_auto.materialized", UnitAttr::get(op->getContext()));
  });
  // CF implements successor selection from constant operands. Use it also for
  // imported LLVM branches so loop traversal shares the standard interface.
  result->walk([](Operation *op) {
    OpBuilder builder(op);
    if (auto branch = dyn_cast<LLVM::BrOp>(op)) {
      cf::BranchOp::create(builder, op->getLoc(), branch.getDest(),
                           branch.getDestOperands());
      op->erase();
    } else if (auto branch = dyn_cast<LLVM::CondBrOp>(op)) {
      cf::CondBranchOp::create(
          builder, op->getLoc(), branch.getCondition(), branch.getTrueDest(),
          branch.getTrueDestOperands(), branch.getFalseDest(),
          branch.getFalseDestOperands());
      op->erase();
    }
  });
  // LLVM dialect leaves integer folding to LLVM. Lift scalar operations to
  // Arith so its existing fold hooks can use per-invocation abstract values.
  static const StringMap<StringRef> arithmetic{
      {"llvm.add", "arith.addi"},        {"llvm.sub", "arith.subi"},
      {"llvm.mul", "arith.muli"},        {"llvm.sdiv", "arith.divsi"},
      {"llvm.udiv", "arith.divui"},      {"llvm.srem", "arith.remsi"},
      {"llvm.urem", "arith.remui"},      {"llvm.shl", "arith.shli"},
      {"llvm.lshr", "arith.shrui"},      {"llvm.ashr", "arith.shrsi"},
      {"llvm.and", "arith.andi"},        {"llvm.or", "arith.ori"},
      {"llvm.xor", "arith.xori"},        {"llvm.zext", "arith.extui"},
      {"llvm.sext", "arith.extsi"},      {"llvm.trunc", "arith.trunci"},
      {"llvm.select", "arith.select"},   {"llvm.icmp", "arith.cmpi"},
      {"llvm.intr.smin", "arith.minsi"}, {"llvm.intr.smax", "arith.maxsi"},
      {"llvm.intr.umin", "arith.minui"}, {"llvm.intr.umax", "arith.maxui"}};
  result->walk([](Operation *op) {
    auto at = arithmetic.find(op->getName().getStringRef());
    if (at == arithmetic.end() || op->getNumResults() != 1 ||
        (!isa<IntegerType>(op->getResult(0).getType()) &&
         !isa<LLVM::SelectOp>(op)))
      return;
    OpBuilder builder(op);
    OperationState state(op->getLoc(), at->second);
    state.addOperands(op->getOperands());
    state.addTypes(op->getResultTypes());
    if (auto cmp = dyn_cast<LLVM::ICmpOp>(op))
      state.addAttribute("predicate",
                         arith::CmpIPredicateAttr::get(
                             op->getContext(), *arith::symbolizeCmpIPredicate(
                                                   LLVM::stringifyICmpPredicate(
                                                       cmp.getPredicate()))));
    if (isa<LLVM::SMinOp, LLVM::SMaxOp, LLVM::UMinOp, LLVM::UMaxOp>(op))
      state.addAttribute("block_auto.intrinsic", builder.getUnitAttr());
    auto *replacement = builder.create(state);
    op->replaceAllUsesWith(replacement->getResults());
    op->erase();
  });
  return result;
}
struct Hook {
  std::function<void(Operation &, unsigned)> record;
  std::function<bool(StringRef, Args, Scalar &)> call;
  std::function<Scalar(LLVM::LoadOp)> load;
};
std::string scalarKey(const Scalar &value) {
  if (auto *i = std::get_if<APInt>(&value.data)) {
    SmallString<32> str;
    i->toString(str, 10, true);
    return "i" + str.str().str();
  }
  if (auto *p = std::get_if<Pointer>(&value.data))
    return "p" + std::to_string(reinterpret_cast<uintptr_t>(p->allocation)) +
           ":" + std::to_string(p->depth) + ":" + std::to_string(p->offset);
  if (auto *f = std::get_if<double>(&value.data))
    return "f" + std::to_string(*f);
  if (auto *values = std::get_if<std::vector<Scalar>>(&value.data)) {
    std::string result = "[";
    for (const auto &v : *values)
      result += scalarKey(v) + ",";
    return result + "]";
  }
  return "?";
}
struct TraceResult {
  double cycles = 0;
  Memory effects;
};
struct Analyzer {
  ModuleOp module;
  DataLayout layout;
  const Profile &profile;
  bool lowered;
  int subblock = -1, depth = 0;
  std::map<std::string, TraceResult> cache;
  std::function<bool(StringRef, Args, Plan &)> external;
  Analyzer(ModuleOp m, const Profile &p, bool lowered)
      : module(m), layout(m), profile(p), lowered(lowered) {}
  TraceResult trace(LLVM::LLVMFuncOp fn, Args arguments, int blockId,
                    Plan *kernel, Memory initial = {}, Hook *hook = nullptr) {
    check(!fn.isExternal() && fn.getNumArguments() == arguments.size(),
          "analysis entry ABI mismatch: " + fn.getSymName().str());
    bool isVector = callingConvention(fn) == 1023;
    std::string key = fn.getSymName().str();
    for (const auto &a : arguments)
      key += ";" + scalarKey(a);
    for (const auto &[p, v] : initial)
      key += ";" + scalarKey(Scalar(p)) + "=" + scalarKey(v);
    if (isVector && cache.count(key))
      return cache.at(key);
    VectorPlan vector(profile, lowered);
    Environment env;
    for (auto a : fn.getArguments())
      env[a] = arguments[a.getArgNumber()];
    Memory memory = initial;
    auto val = [&](mlir::Value v) { return env.lookup(v); };
    auto attributes = [&](Operation &op) {
      SmallVector<Attribute> result;
      for (auto v : op.getOperands())
        result.push_back(toAttribute(val(v), v.getType()));
      return result;
    };
    Block *block = &fn.getBody().front();
    DenseMap<Block *, unsigned> visits;
    unsigned steps = 0;
    while (true) {
      unsigned visit = ++visits[block];
      Block *next = nullptr;
      for (Operation &i : *block) {
        if (constant(i)) {
          Scalar value;
          if (auto op = dyn_cast<LLVM::ConstantOp>(i))
            value = fromAttribute(op.getValue());
          else if (isa<LLVM::UndefOp, LLVM::PoisonOp>(i))
            value = emptyAggregate(i.getResult(0).getType());
          else if (auto zero = dyn_cast<LLVM::ZeroOp>(i))
            if (auto type = dyn_cast<LLVM::LLVMPointerType>(zero.getType()))
              if (type.getAddressSpace() == 6)
                value = Scalar(Pointer{});
          env[i.getResult(0)] = value;
          continue;
        }
        check(++steps <= (hook ? 100000 : 1000000),
              "bounded scalar tracing limit");
        bool materialized = i.hasAttr("block_auto.materialized");
        if (hook && hook->record && !materialized)
          hook->record(i, visit);
        if (isa<LLVM::ReturnOp>(i)) {
          TraceResult result;
          if (isVector)
            result.cycles = vector.complete();
          if (lowered)
            for (const auto &[p, v] : initial)
              result.effects[p] = memory[p];
          if (isVector)
            cache[key] = result;
          return result;
        }
        if (auto branch = dyn_cast<BranchOpInterface>(i)) {
          if (isVector)
            vector.scalar(i);
          next = branch.getSuccessorForOperands(attributes(i));
          check(next, "unknown scalar branch condition");
          auto index =
              llvm::find(i.getSuccessors(), next) - i.getSuccessors().begin();
          auto operands =
              branch.getSuccessorOperands(index).getForwardedOperands();
          SmallVector<std::tuple<Scalar, std::set<int>, double>> incoming;
          for (auto operand : operands)
            incoming.emplace_back(val(operand), vector.dependencies[operand],
                                  vector.times[operand]);
          for (auto [arg, state] : llvm::zip(next->getArguments(), incoming)) {
            env[arg] = std::move(std::get<0>(state));
            vector.dependencies[arg] = std::move(std::get<1>(state));
            vector.times[arg] = std::get<2>(state);
          }
          break;
        }
        if (!isVector && kernel && !materialized)
          kernel->step();
        Scalar result;
        bool vectorCall = false;
        if (auto call = dyn_cast<LLVM::CallOp>(&i)) {
          auto callee =
              call.getCallee()
                  ? module.lookupSymbol<LLVM::LLVMFuncOp>(*call.getCallee())
                  : LLVM::LLVMFuncOp{};
          check(bool(callee), "indirect call is not modelled");
          StringRef name = callName(call);
          std::vector<Scalar> args;
          for (mlir::Value a : call.getCalleeOperands())
            args.push_back(val(a));
          auto helper = [&](Plan *plan) {
            check(depth < 8, "lowered helper recursion limit");
            Memory inherited;
            if (lowered)
              for (const auto &a : args)
                if (auto *p = std::get_if<Pointer>(&a.data))
                  if (memory.count(*p))
                    inherited[*p] = memory[*p];
            ++depth;
            auto work = trace(callee, args, blockId, plan, inherited);
            --depth;
            for (const auto &[p, v] : work.effects)
              memory[p] = v;
            return work;
          };
          if (hook && hook->call && hook->call(name, args, result)) {
          } else if (!callee.isExternal() &&
                     callingConvention(callee) == 1023) {
            check(!isVector && kernel, "nested vector call");
            kernel->vf(helper(nullptr).cycles);
          } else if (name.starts_with("load_gm_to_ubuf_") ||
                     name.starts_with("store_ubuf_to_gm_") ||
                     name.starts_with("copy_ubuf_to_ubuf_")) {
            check(!isVector && kernel, "vector DMA helper");
            if (lowered && !callee.isExternal())
              helper(kernel);
            else {
              SmallVector<StringRef> match;
              static const Regex transfer(
                  R"(^(load_gm_to_ubuf|store_ubuf_to_gm|copy_ubuf_to_ubuf)_([0-9]+)d_(float|half|bfloat16_t|int[0-9]+_t|uint[0-9]+_t)$)");
              check(transfer.match(name, &match), "unmodelled transfer helper");
              bool copy = match[1] == "copy_ubuf_to_ubuf";
              check(!copy || !lowered, "lowered UB copy definition missing");
              unsigned rank = std::stoul(match[2].str());
              check(args.size() >= 3 + rank, "transfer helper ABI");
              std::string type = match[3].str();
              unsigned bits = type == "float" ? 32
                              : (type == "half" || type == "bfloat16_t")
                                  ? 16
                                  : std::stoul(type.substr(
                                        type.find_first_of("0123456789")));
              double bytes = extent(Args(args).slice(3, rank)) * bits / 8;
              if (copy)
                kernel->vf(std::ceil(bytes / 256));
              else
                kernel->dma(name.starts_with("load_"), bytes);
            }
          } else if (name == "llvm.hivm.GET.BLOCK.IDX")
            result = Scalar(int64_t(blockId));
          else if (name == "llvm.hivm.GET.SUBBLOCKID" && subblock >= 0)
            result = Scalar(int64_t(subblock));
          else if (name == "llvm.hivm.GET.CTRL") {
            if (!lowered)
              result = Scalar(int64_t(0));
          } else if (name == "llvm.hivm.GET.SYS.VA.BASE" && lowered) {
          } else if (name.starts_with("llvm.hivm.SBITSET")) {
            if (args.size() == 2 && args[0].known() && args[1].known()) {
              int64_t shift = args[1].integer();
              check(shift >= 0 && shift < 64, "invalid control bit");
              uint64_t mask = uint64_t(1) << shift, value = args[0].bits();
              result =
                  Scalar(APInt(64, name.contains("SBITSET1") ? value | mask
                                                             : value & ~mask));
            }
          } else if (name == "llvm.hivm.SET.CTRL") {
          } else if (name.contains(".FLAG.") && !isVector && kernel) {
            check(args.size() == 3, "flag ABI");
            kernel->flag(name.contains(".SET."), integer(args, 0),
                         integer(args, 1), integer(args, 2));
          } else if (name == "llvm.hivm.BARRIER" && !isVector && kernel)
            kernel->barrier(integer(args, 0));
          else if (name.starts_with("llvm.hivm.") && isVector) {
            vector.add(call);
            vectorCall = true;
          } else if (!isVector && kernel &&
                     (kernel->extra(name, args) ||
                      (external && external(name, args, *kernel)))) {
          } else {
            check(lowered && !isVector && kernel && !callee.isExternal() &&
                      callingConvention(callee) == 0 &&
                      isa<LLVM::LLVMVoidType>(
                          callee.getFunctionType().getReturnType()),
                  "unmodelled call: " + name.str());
            helper(kernel);
          }

        } else if (auto freeze = dyn_cast<LLVM::FreezeOp>(i)) {
          result = val(i.getOperand(0));
        } else if (auto alloc = dyn_cast<LLVM::AllocaOp>(i)) {
          result = Scalar(Pointer{&i, lowered ? depth : 0, 0});
        } else if (auto store = dyn_cast<LLVM::StoreOp>(i)) {
          Scalar pointer = val(store.getAddr());
          auto *p = std::get_if<Pointer>(&pointer.data);
          if (p && p->allocation)
            memory[*p] = val(store.getValue());
          else
            check(!isVector &&
                      cast<LLVM::LLVMPointerType>(store.getAddr().getType())
                              .getAddressSpace() == 6,
                  "unmodelled scalar write");
        } else if (auto load = dyn_cast<LLVM::LoadOp>(i)) {
          Scalar pointer = val(load.getAddr());
          if (auto *p = std::get_if<Pointer>(&pointer.data))
            if (memory.count(*p))
              result = memory[*p];
          if (hook && hook->load)
            result = hook->load(load);
        } else if (auto extract = dyn_cast<LLVM::ExtractValueOp>(i)) {
          result = val(extract.getContainer());
          for (auto index : extract.getPosition()) {
            auto *values = std::get_if<std::vector<Scalar>>(&result.data);
            Scalar selected =
                values && index >= 0 && index < int64_t(values->size())
                    ? (*values)[index]
                    : Scalar();
            result = std::move(selected);
          }
        } else if (auto insertOp = dyn_cast<LLVM::InsertValueOp>(i)) {
          result = insert(val(insertOp.getContainer()), insertOp.getPosition(),
                          val(insertOp.getValue()));
        } else if (auto castOp = dyn_cast<LLVM::PtrToIntOp>(i)) {
          auto pointer = val(castOp.getArg());
          if (auto *p = std::get_if<Pointer>(&pointer.data))
            if (lowered && !p->allocation)
              result = Scalar(p->offset);
        } else if (auto castOp = dyn_cast<LLVM::IntToPtrOp>(i)) {
          auto source = val(castOp.getArg());
          if (cast<LLVM::LLVMPointerType>(castOp.getType()).getAddressSpace() ==
                  6 &&
              source.known())
            result = Scalar(Pointer{nullptr, 0, source.integer()});
        } else if (auto gep = dyn_cast<LLVM::GEPOp>(i)) {
          auto pointer = val(gep.getBase());
          if (auto *p = std::get_if<Pointer>(&pointer.data))
            if (lowered && !p->allocation) {
              auto indices = gep.getRawConstantIndices();
              check(indices.size() == 1 &&
                        !isa<LLVM::LLVMStructType, LLVM::LLVMArrayType>(
                            gep.getElemType()),
                    "unsupported UB GEP layout");
              Scalar index = indices[0] == LLVM::GEPOp::kDynamicIndex
                                 ? val(gep.getDynamicIndices()[0])
                                 : Scalar(int64_t(indices[0]));
              if (index.known())
                result = Scalar(Pointer{
                    nullptr, 0,
                    p->offset +
                        index.integer() *
                            int64_t(layout.getTypeSize(gep.getElemType())
                                        .getFixedValue())});
            }
        } else {
          // Delegate arithmetic, comparisons, casts and selects to the
          // dialect's fold hooks instead of maintaining an opcode evaluator.
          check(isMemoryEffectFree(&i) ||
                    isa<LLVM::LifetimeStartOp, LLVM::LifetimeEndOp>(i),
                "unmodelled instruction effects");
          SmallVector<OpFoldResult> folded;
          SmallVector<mlir::Value> operands(i.getOperands());
          auto attrs = i.getAttrDictionary();
          if (succeeded(i.fold(attributes(i), folded))) {
            // As in SparseConstantPropagation, invocation-specific values
            // must not cause an in-place rewrite of the shared analysis IR.
            if (folded.empty()) {
              i.setOperands(operands);
              i.setAttrs(attrs);
            } else if (folded.size() == 1) {
              if (auto attr = dyn_cast<Attribute>(folded[0]))
                result = fromAttribute(attr);
              else
                result = val(cast<mlir::Value>(folded[0]));
            }
          }
        }
        if (i.getNumResults())
          env[i.getResult(0)] = result;
        if (isVector && !vectorCall && !materialized)
          vector.scalar(i);
      }
      check(next, "missing scalar terminator");
      block = next;
    }
  }
};

struct Work {
  double instructions = 0, gmLoads = 0, gmStores = 0, ubLoads = 0, ubStores = 0,
         ubBytes = 0, barriers = 0;
  void record(Operation &i) {
    if (!(constant(i) || isa<LLVM::ReturnOp, LLVM::BitcastOp>(i)))
      ++instructions;
    if (isa<LLVM::LoadOp, LLVM::StoreOp>(i)) {
      bool load = isa<LLVM::LoadOp>(i);
      mlir::Type type =
          load ? i.getResult(0).getType() : i.getOperand(0).getType();
      auto pointer = load ? i.getOperand(0) : i.getOperand(1);
      unsigned space = cast<LLVM::LLVMPointerType>(pointer.getType())
                           .getAddressSpace(),
               bits = type.getIntOrFloatBitWidth();
      check((space == 1 || space == 6) && !isa<VectorType>(type) && bits &&
                bits <= 64,
            "unsupported SIMT memory access");
      if (space == 1)
        (load ? gmLoads : gmStores)++;
      else {
        (load ? ubLoads : ubStores)++;
        ubBytes += (bits + 7) / 8;
      }
    }
    if (auto call = dyn_cast<LLVM::CallOp>(&i))
      if (callName(call) == "llvm.hivm.sync.workitems")
        ++barriers;
  }
  void combine(const Work &w, bool maximum = false) {
    for (auto field :
         {&Work::instructions, &Work::gmLoads, &Work::gmStores, &Work::ubLoads,
          &Work::ubStores, &Work::ubBytes, &Work::barriers})
      this->*field =
          maximum ? std::max(this->*field, w.*field) : this->*field + w.*field;
  }
};
bool simtIntrinsic(StringRef name) {
  return name == "llvm.hivm.get.TID.X" || name == "llvm.hivm.get.TID.Y" ||
         name == "llvm.hivm.get.TID.Z" || name == "llvm.hivm.get.BLOCK.DIM.X" ||
         name == "llvm.hivm.get.BLOCK.DIM.Y" ||
         name == "llvm.hivm.get.BLOCK.DIM.Z" ||
         name == "llvm.hivm.shfl.idx.i32" || name == "llvm.hivm.sync.workitems";
}
struct SimtModel {
  Analyzer &analyzer;
  const Object &parameters;
  struct Cost {
    double cycles, bytes, startup;
  };
  std::map<std::string, Cost> cache;
  Cost trace(LLVM::LLVMFuncOp wrapper, Args input) {
    std::vector<Scalar> bound;
    for (const auto &a : wrapper.getArguments()) {
      check(isa<LLVM::LLVMPointerType>(a.getType()) ||
                std::holds_alternative<APInt>(input[a.getArgNumber()].data),
            "dynamic SIMT helper scalar");
      bound.push_back(isa<LLVM::LLVMPointerType>(a.getType())
                          ? Scalar()
                          : input[a.getArgNumber()]);
    }
    std::string key = wrapper.getSymName().str();
    for (const auto &a : bound)
      key += ";" + scalarKey(a);
    if (cache.count(key))
      return cache.at(key);
    LLVM::LLVMFuncOp target;
    uint64_t info = 0;
    unsigned infos = 0, launches = 0, steps = 0;
    std::vector<Scalar> arguments;
    Hook capture;
    capture.record = [&](Operation &op, unsigned) {
      check(!isa<BranchOpInterface, LLVM::LoadOp, LLVM::StoreOp>(op),
            "SIMT wrapper has unresolved control or memory");
      steps += !isa<LLVM::ReturnOp>(op);
      if (auto call = dyn_cast<LLVM::CallOp>(op))
        if (callName(call) == "llvm.hivm.store.vfsimt.info") {
          APInt dimensions;
          check(call.getCalleeOperands().size() == 1 &&
                    matchPattern(call.getCalleeOperands()[0],
                                 m_ConstantInt(&dimensions)),
                "dynamic SIMT dimensions");
          info = dimensions.getZExtValue();
          ++infos;
        }
    };
    capture.call = [&](StringRef name, Args args, Scalar &) {
      auto fn = analyzer.module.lookupSymbol<LLVM::LLVMFuncOp>(name);
      if (fn && callingConvention(fn) == 1022) {
        target = fn;
        ++launches;
        arguments.assign(args.begin(), args.end());
      } else
        check(name == "llvm.hivm.store.vfsimt.info" ||
                  name == "llvm.hivm.BARRIER" || simtIntrinsic(name),
              "unmodelled SIMT wrapper call");
      return true;
    };
    KernelPlan dummy(analyzer.profile);
    analyzer.trace(wrapper, bound, 0, &dummy, {}, &capture);
    check(infos == 1 && launches == 1 && !target.isExternal() && !(info >> 48),
          "expected one static SIMT launch");
    std::array<int64_t, 3> dims{int64_t(info & 65535),
                                int64_t((info >> 16) & 65535),
                                int64_t((info >> 32) & 65535)};
    check(*std::min_element(dims.begin(), dims.end()) > 0 &&
              dims[0] * dims[1] * dims[2] <= 4096,
          "SIMT launch exceeds bounded domain");
    check(arguments.size() == target.getNumArguments(), "SIMT launch ABI");
    for (const auto &a : target.getArguments())
      check(isa<LLVM::LLVMPointerType>(a.getType()) ||
                std::holds_alternative<APInt>(arguments[a.getArgNumber()].data),
            "dynamic SIMT scalar");
    std::map<int, std::set<std::pair<Operation *, unsigned>>> warps;
    Work total, issued;
    for (int64_t x = 0; x < dims[0]; ++x)
      for (int64_t y = 0; y < dims[1]; ++y)
        for (int64_t z = 0; z < dims[2]; ++z) {
          std::array<int64_t, 3> coord{x, y, z};
          auto &events = warps[(x + dims[0] * (y + dims[1] * z)) / 32];
          Work upper;
          bool indirect = wrapper.getSymName().starts_with("indirect_load_");
          for (int mask = indirect ? 0 : -1; mask < (indirect ? 2 : 0);
               ++mask) {
            Work work;
            Hook hook;
            hook.record = [&](Operation &i, unsigned visit) {
              work.record(i);
              events.emplace(&i, visit);
            };
            hook.call = [&](StringRef name, Args, Scalar &value) {
              if (!simtIntrinsic(name))
                return false;
              auto axis = StringRef("XYZ").find(name.back());
              if (name.starts_with("llvm.hivm.get.TID."))
                value = Scalar(coord[axis]);
              if (name.starts_with("llvm.hivm.get.BLOCK.DIM."))
                value = Scalar(dims[axis]);
              return true;
            };
            hook.load = [&](LLVM::LoadOp i) {
              return mask >= 0 && i.getType().isInteger(8) &&
                             cast<LLVM::LLVMPointerType>(i.getAddr().getType())
                                     .getAddressSpace() == 6
                         ? Scalar(int64_t(mask))
                         : Scalar();
            };
            KernelPlan plan(analyzer.profile);
            analyzer.trace(target, arguments, 0, &plan, {}, &hook);
            upper.combine(work, true);
          }
          total.combine(upper);
        }
    for (const auto &[warp, events] : warps)
      for (auto [i, visit] : events)
        issued.record(*i);
    check(total.gmStores == 0, "unexpected SIMT GM store");
    double transaction = number(parameters, "simt_transaction_bytes"),
           transition = number(parameters, "simt_transition_cycles"),
           instruction = number(parameters, "simt_instruction_cycles");
    check(transaction > 0 && transition > 0 && instruction > 0,
          "invalid SIMT cost parameters");
    double bytes = total.gmLoads * transaction,
           startup = 2 * transition + steps;
    double cycles = startup + issued.instructions * instruction +
                    (issued.ubLoads + issued.ubStores) * 4 +
                    total.ubBytes / analyzer.profile.localBandwidth +
                    issued.barriers + bytes / analyzer.profile.localBandwidth;
    return cache[key] = {cycles, bytes, startup};
  }
  bool call(StringRef name, Args args, Plan &plan) {
    if (name != "indirect_load_nonvolatile_2d_float_int64_t" &&
        name != "cumsum_1d_float_dim0" &&
        name != "indirect_load_nonvolatile_1d_bfloat16_t_int64_t")
      return false;
    auto fn = analyzer.module.lookupSymbol<LLVM::LLVMFuncOp>(name);
    check(fn && !fn.isExternal() && fn.getNumArguments() == args.size(),
          "SIMT helper ABI mismatch");
    auto cost = trace(fn, args);
    auto &kernel = static_cast<KernelPlan &>(plan);
    Ref ref = kernel.add("simt_helper", kernel.all(), cost.cycles, cost.bytes,
                         cost.startup);
    kernel.scalar = ref;
    for (auto &[pipe, value] : kernel.pipes)
      value = ref;
    return true;
  }
};

} // namespace
json::Object analyzeBlockCandidate(const json::Object &config) {
  Profile profile(config);
  DialectRegistry registry;
  registry.insert<LLVM::LLVMDialect, DLTIDialect, cf::ControlFlowDialect,
                  arith::ArithDialect>();
  registerLLVMDialectImport(registry);
  MLIRContext context(registry, MLIRContext::Threading::DISABLED);
  bool vv = text(config, "mode") == "vv",
       autoBlock = config.getBoolean("auto_blockify").value_or(false);
  auto coreCount = config.getInteger("cores"),
       blockCount = config.getInteger("blocks");
  check(coreCount && blockCount && *coreCount > 0 && *coreCount <= 4096 &&
            *blockCount > 0 &&
            *blockCount <= std::numeric_limits<unsigned>::max() &&
            (vv || (*blockCount <= 10000 &&
                    (!autoBlock || *blockCount <= *coreCount))),
        "launch outside bounded analysis");
  unsigned cores = *coreCount, blocks = *blockCount;
  std::vector<Scalar> args;
  for (const auto &v : array(config, "arguments")) {
    Scalar s;
    if (auto i = v.getAsInteger())
      s = Scalar(*i);
    else if (auto b = v.getAsBoolean())
      s = Scalar(int64_t(*b));
    else if (auto f = v.getAsNumber())
      s.data = *f;
    args.push_back(s);
  }
  std::map<std::string, OwningOpRef<ModuleOp>> modules;
  for (const auto &[role, source] : object(*config.get("modules"))) {
    check(bool(source.getAsString()), "missing LLVM module text");
    modules[role.str()] = parse(*source.getAsString(), context);
  }
  std::string name = text(config, "name");
  auto entry = [&](StringRef role) {
    check(modules.count(role.str()), "LLVM core role missing");
    std::string suffix = vv || modules.size() == 1 ? "" : "_mix_" + role.str();
    auto fn =
        modules.at(role.str())->lookupSymbol<LLVM::LLVMFuncOp>(name + suffix);
    check(fn && !fn.isExternal() && callingConvention(fn) != 1023,
          "scalar entry missing");
    if (!vv && modules.size() == 1)
      check(!functionAttribute(fn, "cube-only-kernel").empty(),
            "single CV entry is not Cube");
    return fn;
  };
  Object result{{"status", "STATIC_COST_ESTIMATED"},
                {"auto_blockify", autoBlock}};
  result["logical_grid"] = Array(array(config, "grid"));
  if (vv) {
    auto fn = entry("aiv");
    Analyzer analyzer(*modules.at("aiv"), profile, true);
    SimtModel simt{analyzer, blockauto::optional(config, "mixed"), {}};
    analyzer.external = [&](StringRef callee, Args args, Plan &plan) {
      return config.getObject("mixed") && simt.call(callee, args, plan);
    };
    std::vector<KernelPlan> traces;
    for (unsigned block = 0; block < blocks; ++block) {
      traces.emplace_back(profile);
      analyzer.trace(fn, args, block, &traces.back());
      traces.back().complete();
    }
    result["score"] = schedule(traces, cores, profile.globalBandwidth);
    result["core_count"] = int64_t(cores);
    result["model"] = "vv-lowered-entry-v1";
  } else {
    check(modules.count("aic") && modules.size() <= 2, "unsupported CV roles");
    std::vector<double> ends(std::min(cores, blocks));
    double bytes = 0, flops = 0, mmas = 0;
    for (unsigned block = 0; block < blocks; ++block) {
      CVGraph graph;
      for (int role = -1; role < (modules.size() == 2 ? 2 : 0); ++role) {
        StringRef core = role < 0 ? "aic" : "aiv";
        CVPlan plan(graph, role);
        Analyzer analyzer(*modules.at(core.str()), profile, false);
        analyzer.subblock = role;
        analyzer.trace(entry(core), args, block, &plan);
        plan.complete();
        bytes += plan.read + plan.write;
        flops += plan.flops;
        mmas += plan.mmas;
      }
      auto slot = autoBlock ? ends.begin() + block
                            : std::min_element(ends.begin(), ends.end());
      *slot += graph.finish();
    }
    result["score"] =
        std::max(*std::max_element(ends.begin(), ends.end()), bytes / 512);
    result["mode"] = modules.size() == 2 ? "Cube+Vector" : "Cube";
    result["cube_core_count"] = int64_t(cores);
    result["launched_groups"] = int64_t(blocks);
    result["counts"] = Object{{"useful_flops", flops}, {"mma_calls", mmas}};
  }
  check(number(result, "score") >= 0, "invalid static cost");
  return result;
}
} // namespace mlir::triton::ascend

namespace mlir::triton::ascend::blockauto {
void registerCandidateDialects(DialectRegistry &registry) {
  registry.insert<arith::ArithDialect, cf::ControlFlowDialect, DLTIDialect,
                  func::FuncDialect, LLVM::LLVMDialect, memref::MemRefDialect,
                  scf::SCFDialect, hacc::HACCDialect, hivm::HIVMDialect>();
}
OwningOpRef<ModuleOp> parseCandidateIR(StringRef input, MLIRContext &context) {
  std::string source = input.str();
  // BiSheng LLVM 20 emits ':' here; the host LLVM 22 DLTI syntax uses '='.
  StringRef old = "#dlti.target_system_spec<\"NPU\" :";
  if (auto pos = source.find(old.str()); pos != std::string::npos)
    source[pos + old.size() - 1] = '=';
  auto module =
      parseSourceString<ModuleOp>(source, ParserConfig(&context, false));
  check(bool(module), "cannot parse candidate final MLIR");
  return module;
}
CandidateIRAnalysis::CandidateIRAnalysis(Operation *op,
                                         AnalysisManager &manager)
    : module(cast<ModuleOp>(op)),
      layout(manager.getAnalysis<DataLayoutAnalysis>().getAtOrAbove(op)) {}
int64_t CandidateIRAnalysis::spec(StringRef key) const {
  auto *ctx = module->getContext();
  auto value = layout.getDevicePropertyValue(StringAttr::get(ctx, "NPU"),
                                             StringAttr::get(ctx, key));
  auto number = value ? dyn_cast<IntegerAttr>(*value) : IntegerAttr{};
  check(number && number.getInt() > 0,
        "missing target specification: " + key.str());
  return number.getInt();
}
std::string CandidateIRAnalysis::metaLibrary() const {
  auto bitcode =
      module->getAttrOfType<hivm::AIV_BITCODEAttr>("hivm.aiv_bitcode");
  return bitcode ? bitcode.getPath().getValue().str() : "";
}
void CandidateIRAnalysis::checkResources(const O &item) const {
  const O &metadata = sub(item, "metadata");
  std::string mode = str(metadata, "parallel_mode");
  check(str(sub(metadata, "target"), "arch") == "Ascend950PR_9579",
        "cost model target is outside the validated A5 profile");
  bool vv = str(metadata, "mix_mode") == "aiv",
       mixed = vv && mode == "mix_simd_simt", bound = false, hasUB = false;
  int64_t capacity = spec("UB_SIZE"),
          alignment = vv ? spec("UB_ALIGN_SIZE") : 8, end = -1;
  auto isUB = [](Type type) {
    auto memref = dyn_cast<MemRefType>(type);
    auto space =
        memref
            ? dyn_cast_or_null<hivm::AddressSpaceAttr>(memref.getMemorySpace())
            : hivm::AddressSpaceAttr{};
    return space && space.getAddressSpace() == hivm::AddressSpace::UB;
  };
  module->walk([&](Operation *op) {
    hasUB |= llvm::any_of(op->getResultTypes(), isUB) ||
             llvm::any_of(op->getOperandTypes(), isUB);
    if (isa<memref::AllocOp, memref::AllocaOp>(op))
      check(!isUB(op->getResult(0).getType()),
            "unresolved UB allocation remains in final IR");
    auto cast = dyn_cast<hivm::PointerCastOp>(op);
    if (!cast || !isUB(cast.getType()))
      return;
    auto type = mlir::cast<MemRefType>(cast.getType());
    check(type.hasStaticShape() && type.getLayout().isIdentity(),
          "UB pointer has an unproved shape or layout");
    auto width = layout.getTypeSizeInBits(type.getElementType());
    check(!width.isScalable(), "scalable UB element size");
    int64_t bits = width.getFixedValue();
    bool conservative = vv && bits == 1;
    check(conservative || (bits >= 8 && bits % 8 == 0),
          "unsupported UB element width");
    A factors{conservative ? 8 : bits};
    for (int64_t dim : type.getShape())
      factors.push_back(dim);
    bits = product(factors);
    for (auto addr : cast.getAddrs()) {
      APInt offset;
      check(matchPattern(addr, m_ConstantInt(&offset)) &&
                offset.isNonNegative() && offset.isSignedIntN(64),
            "UB pointer has an unproved offset");
      int64_t start = offset.getSExtValue();
      check(bits <= INT64_MAX - alignment + 1 &&
                start <= (INT64_MAX - bits - alignment + 1) / 8,
            "UB extent overflow");
      end = std::max(end, start * 8 + int64_t(llvm::alignTo(bits, alignment)));
    }
    bound |= conservative;
  });
  if (end < 0) {
    check(!module->getRegion(0).getOps<func::FuncOp>().empty() && !vv &&
              !item.get("llvm_aiv") && !hasUB,
          "no verifiable static UB allocation in final IR");
    end = 0;
  }
  if (vv) {
    check(mode == "simd" || mixed,
          "only AIV SIMD and MIX resource proofs are supported");
    if (mixed) {
      int64_t dynamic = integer(metadata, "shared_mem_dynamic_size"),
              full = capacity;
      check(dynamic > 0 && dynamic <= INT64_MAX / 8,
            "missing runtime dynamic UB size");
      capacity = dynamic * 8;
      check(full - spec("MAXIMUM_D_CACHE_SIZE") <= capacity &&
                capacity <= full - spec("MINIMAL_D_CACHE_SIZE"),
            "MIX dynamic UB conflicts with D-cache reservation");
    }
  }
  check(end <= capacity, bound ? "i1 conservative UB upper bound exceeds "
                                 "capacity; actual overflow unconfirmed"
                               : "final UB address exceeds capacity");
}
} // namespace mlir::triton::ascend::blockauto

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

namespace mlir::triton::ascend::blockauto {
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
O rankCV(const O &records, const O &contract) {
  A incomplete;
  for (const auto &[key, value] : records) {
    const O &row = object(value);
    if (!flag(row, "resource_checked") &&
        (integer(row, "factor") == 1 || !excluded(str(row, "status"), false)))
      incomplete.push_back(key.str());
  }
  if (!incomplete.empty())
    return O{{"reason", "CV_MODEL_INCOMPLETE"},
             {"incomplete_candidates", std::move(incomplete)}};
  const O &baseline = sub(records, "unmerged");
  const A &original = array(contract, "grid");
  std::vector<std::string> eligible;
  for (const auto &[key, value] : records) {
    const O &row = object(value);
    int64_t factor = integer(row, "factor"), axis = integer(row, "axis");
    if (factor == 1 || !flag(row, "resource_checked") ||
        integer(original[axis]) % factor)
      continue;
    if (get(row, "mode") == get(baseline, "mode") &&
        get(row, "cube_core_count") == get(baseline, "cube_core_count") &&
        get(row, "useful_flops") == get(baseline, "useful_flops"))
      eligible.push_back(key.str());
  }
  auto order = [&](const std::string &key) {
    const O &r = sub(records, key);
    return std::make_tuple(number(r, "estimated_cycles"), integer(r, "factor"),
                           integer(r, "axis"));
  };
  std::string selected = "unmerged";
  for (const auto &key : eligible)
    if (order(key) < order(selected))
      selected = key;
  std::string reason = eligible.empty()         ? "MERGED_RESOURCE_UNAVAILABLE"
                       : selected == "unmerged" ? "ORIGINAL_LOWEST_LATENCY"
                                                : "LOWER_ESTIMATED_LATENCY";
  bool spatial = std::any_of(records.begin(), records.end(), [](const auto &v) {
    auto kind = str(object(v.second), "merge_kind");
    return kind == "M" || kind == "N";
  });
  if (contract.getBoolean("cv_empirical_short_batch").value_or(true) &&
      product(original) <= integer(baseline, "cube_core_count") && !spatial) {
    std::vector<std::pair<int64_t, std::string>> preferred;
    for (const auto &key : eligible) {
      const O &row = sub(records, key);
      int64_t axis = integer(row, "axis");
      bool single = true;
      for (int64_t i = 0; i < 3; ++i)
        if (i != axis && integer(original[i]) != 1)
          single = false;
      if (str(row, "merge_kind") == "independent_batch" &&
          integer(row, "factor") == 2 && single)
        preferred.emplace_back(axis, key);
    }
    if (!preferred.empty()) {
      selected = std::min_element(preferred.begin(), preferred.end())->second;
      reason = "EMPIRICAL_SINGLE_TILE_SHORT_BATCH_X2";
    }
  }
  return O{
      {"selected_key", selected}, {"reason", reason}, {"model_complete", true}};
}
} // namespace mlir::triton::ascend::blockauto

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
