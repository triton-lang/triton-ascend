// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "Selection.h"
#include "llvm/AsmParser/Parser.h"
#include "llvm/IR/ConstantFold.h"
#include "llvm/IR/Constants.h"
#include "llvm/IR/Instructions.h"
#include "llvm/IR/LLVMContext.h"
#include "llvm/IR/Module.h"
#include "llvm/Support/JSON.h"
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
#include <numeric>
#include <queue>
#include <set>
#include <stdexcept>
#include <tuple>
#include <variant>

namespace mlir::triton::ascend {
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
std::string printValue(const Value &v) {
  std::string s;
  raw_string_ostream out(s);
  v.print(out);
  return s;
}

struct Pointer {
  const Value *allocation = nullptr;
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
using Environment = std::map<const Value *, Scalar>;
Scalar emptyAggregate(Type *type) {
  std::vector<Scalar> values;
  if (auto *s = dyn_cast<StructType>(type))
    for (Type *element : s->elements())
      values.push_back(emptyAggregate(element));
  else if (auto *a = dyn_cast<ArrayType>(type))
    values.assign(a->getNumElements(), emptyAggregate(a->getElementType()));
  else
    return {};
  return Scalar(std::move(values));
}
Scalar evaluate(const Value *value, const Environment &env) {
  if (auto found = env.find(value); found != env.end())
    return found->second;
  if (auto *i = dyn_cast<ConstantInt>(value))
    return Scalar(i->getValue());
  if (auto *f = dyn_cast<ConstantFP>(value)) {
    Scalar result;
    // Keep the previous abstract domain: hexadecimal payloads are unknown.
    if (printValue(*f).find("0x") != std::string::npos)
      return result;
    double number = f->getValueAPF().convertToDouble();
    if (std::isfinite(number))
      result.data = number;
    return result;
  }
  if (isa<UndefValue, PoisonValue>(value))
    return emptyAggregate(value->getType());
  if (auto *a = dyn_cast<ConstantDataArray>(value)) {
    std::vector<Scalar> result;
    for (unsigned i = 0; i < a->getNumElements(); ++i)
      result.push_back(evaluate(a->getElementAsConstant(i), env));
    return Scalar(std::move(result));
  }
  if (auto *p = dyn_cast<ConstantPointerNull>(value))
    if (p->getType()->getAddressSpace() == 6)
      return Scalar(Pointer{});
  if (auto *c = dyn_cast<ConstantExpr>(value))
    if (c->getOpcode() == Instruction::IntToPtr &&
        c->getType()->getPointerAddressSpace() == 6)
      return Scalar(
          Pointer{nullptr, 0, evaluate(c->getOperand(0), env).integer()});
  if (isa<ConstantStruct, ConstantArray>(value)) {
    std::vector<Scalar> result;
    for (const Use &operand : cast<Constant>(value)->operands())
      result.push_back(evaluate(operand, env));
    return Scalar(std::move(result));
  }
  return {};
}
Scalar insert(Scalar value, ArrayRef<unsigned> indices, const Scalar &element) {
  if (indices.empty())
    return element;
  auto *values = std::get_if<std::vector<Scalar>>(&value.data);
  if (!values)
    return {};
  check(indices.front() < values->size(), "invalid aggregate index");
  (*values)[indices.front()] =
      insert((*values)[indices.front()], indices.drop_front(), element);
  return value;
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
  std::map<const Value *, std::set<int>> dependencies;
  std::map<const Value *, double> times;
  std::map<std::string, std::vector<double>> legacySlots{
      {"exec", {0, 0}}, {"load", {0, 0}}, {"store", {0}}};
  std::vector<Node> nodes;
  std::vector<int> retirement;
  std::set<int> control;
  std::vector<double> pending;
  double issue = 0, finish = 0;
  VectorPlan(const Profile &p, bool ready) : profile(p), readyMode(ready) {}
  std::set<int> deps(const Instruction &i) const {
    std::set<int> result;
    for (const Use &operand : i.operands())
      if (auto at = dependencies.find(operand); at != dependencies.end())
        result.insert(at->second.begin(), at->second.end());
    return result;
  }
  double dependencyTime(const Instruction &i) {
    double result = 0;
    for (const Use &operand : i.operands())
      result = std::max(result, times[operand]);
    return result;
  }
  int node(const Instruction &i, Rate rate) {
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
    if (!i.getType()->isVoidTy())
      dependencies[&i] = {id};
    return id;
  }
  void add(const CallBase &call) {
    auto rate = profile.cost(call.getCalledFunction()->getName(), readyMode);
    if (readyMode) {
      node(call, rate);
      return;
    }
    std::string pipe =
        rate.unit == "load" || rate.unit == "store" ? rate.unit : "exec";
    auto &slots = legacySlots[pipe];
    check(rate.slots <= slots.size(), "unsupported vector issue width");
    auto slot = std::min_element(slots.begin(), slots.begin() + rate.slots);
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [&](double t) { return t <= issue; }),
                  pending.end());
    std::sort(pending.begin(), pending.end());
    double queued = pending.size() >= profile.window ? pending.front() : 0;
    double start = std::max({issue, *slot, dependencyTime(call), queued});
    *slot = start + rate.interval;
    issue = start + 0.5;
    finish = std::max(finish, start + rate.latency);
    pending.erase(std::remove_if(pending.begin(), pending.end(),
                                 [&](double t) { return t <= start; }),
                  pending.end());
    pending.push_back(start + rate.latency);
    times[&call] = start + rate.latency;
  }
  void scalar(const Instruction &i) {
    if (readyMode)
      dependencies[&i] = deps(i);
    else
      times[&i] = dependencyTime(i);
    if (!readyMode || isa<CallBase, AllocaInst, LoadInst, StoreInst,
                          BitCastInst, ExtractValueInst, InsertValueInst>(i))
      return;
    int id = node(i, {1, 1, 1, "vf_control", "vf_control"});
    if (isa<BranchInst>(i))
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
      for (int parent : nodes[i].deps) {
        check(parent >= 0 && parent < int(i), "invalid vector dependency");
        children[parent].push_back(i);
      }
      if (!remaining[i])
        queue.emplace(earliest[i], i);
    }
    std::map<std::string, double> units;
    std::map<std::pair<std::string, std::string>, std::vector<double>> pipes;
    unsigned completed = 0;
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
      ++completed;
      for (int child : children[i]) {
        earliest[child] = std::max(earliest[child], end);
        if (!--remaining[child])
          queue.emplace(earliest[child], child);
      }
    }
    check(completed == nodes.size() && std::isfinite(finish),
          "vector schedule did not complete");
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
    check(size >= 0, "negative DMA extent");
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
  check(cores && !blocks.empty(), "empty launch or invalid core count");
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
  unsigned next = 0, completed = 0;
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
      ++completed;
      finish = std::max(finish, end);
      start(states[block].core, end);
    }
    release(block, i, end);
  }
  check(completed == blocks.size(), "dependency graph did not complete");
  return finish;
}

std::unique_ptr<Module> parse(StringRef input, LLVMContext &context) {
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
  SMDiagnostic diagnostic;
  auto module = parseAssemblyString(source, diagnostic, context);
  check(bool(module),
        "LLVM analysis parse failed: " + diagnostic.getMessage().str() + ": " +
            diagnostic.getLineContents().str());
  return module;
}
struct Hook {
  std::function<void(const Instruction &, unsigned)> record;
  std::function<bool(StringRef, Args, Scalar &)> call;
  std::function<Scalar(const LoadInst &)> load;
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
  Module &module;
  const Profile &profile;
  bool lowered;
  int subblock = -1, depth = 0;
  std::map<std::string, TraceResult> cache;
  std::function<bool(StringRef, Args, Plan &)> external;
  Analyzer(Module &m, const Profile &p, bool lowered)
      : module(m), profile(p), lowered(lowered) {}
  TraceResult trace(Function &fn, Args arguments, int blockId, Plan *kernel,
                    Memory initial = {}, Hook *hook = nullptr) {
    check(!fn.isDeclaration() && fn.arg_size() == arguments.size(),
          "analysis entry ABI mismatch: " + fn.getName().str());
    bool isVector = fn.getCallingConv() == 1023;
    std::string key = fn.getName().str();
    for (const auto &a : arguments)
      key += ";" + scalarKey(a);
    for (const auto &[p, v] : initial)
      key += ";" + scalarKey(Scalar(p)) + "=" + scalarKey(v);
    if (isVector && cache.count(key))
      return cache.at(key);
    VectorPlan vector(profile, lowered);
    Environment env;
    for (const auto &a : fn.args())
      env[&a] = arguments[a.getArgNo()];
    Memory memory = initial;
    auto val = [&](const Value *v) { return evaluate(v, env); };
    const BasicBlock *block = &fn.getEntryBlock(), *previous = nullptr;
    std::map<const BasicBlock *, unsigned> visits;
    unsigned steps = 0;
    while (true) {
      unsigned visit = ++visits[block];
      Environment phi;
      std::map<const Value *, std::set<int>> phiDeps;
      std::map<const Value *, double> phiTimes;
      for (const auto &i : block->phis()) {
        check(previous && i.getBasicBlockIndex(previous) >= 0,
              "missing phi predecessor");
        Value *incoming = i.getIncomingValueForBlock(previous);
        phi[&i] = val(incoming);
        phiDeps[&i] = vector.dependencies[incoming];
        phiTimes[&i] = vector.times[incoming];
      }
      for (const auto &[v, x] : phi) {
        env[v] = x;
        vector.dependencies[v] = phiDeps[v];
        vector.times[v] = phiTimes[v];
      }
      const BasicBlock *next = nullptr;
      for (const Instruction &i : *block) {
        try {
          check(++steps <= (hook ? 100000 : 1000000),
                "bounded scalar tracing limit");
          if (hook && hook->record)
            hook->record(i, visit);
          if (isa<PHINode>(i))
            continue;
          if (isa<ReturnInst>(i)) {
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
          if (auto *br = dyn_cast<BranchInst>(&i)) {
            if (isVector)
              vector.scalar(i);
            next = br->getSuccessor(
                br->isUnconditional() || val(br->getCondition()).integer() ? 0
                                                                           : 1);
            break;
          }
          if (!isVector && kernel)
            kernel->step();
          Scalar result;
          bool vectorCall = false;
          if (auto *call = dyn_cast<CallBase>(&i)) {
            Function *callee = call->getCalledFunction();
            check(callee, "indirect call is not modelled");
            StringRef name = callee->getName();
            std::vector<Scalar> args;
            for (const Use &a : call->args())
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
              auto work = trace(*callee, args, blockId, plan, inherited);
              --depth;
              for (const auto &[p, v] : work.effects)
                memory[p] = v;
              return work;
            };
            if (hook && hook->call && hook->call(name, args, result)) {
            } else if (!callee->isDeclaration() &&
                       callee->getCallingConv() == 1023) {
              check(!isVector && kernel, "nested vector call");
              kernel->vf(helper(nullptr).cycles);
            } else if (name.starts_with("load_gm_to_ubuf_") ||
                       name.starts_with("store_ubuf_to_gm_") ||
                       name.starts_with("copy_ubuf_to_ubuf_")) {
              check(!isVector && kernel, "vector DMA helper");
              if (lowered && !callee->isDeclaration())
                helper(kernel);
              else {
                SmallVector<StringRef> match;
                static const Regex transfer(
                    R"(^(load_gm_to_ubuf|store_ubuf_to_gm|copy_ubuf_to_ubuf)_([0-9]+)d_(float|half|bfloat16_t|int[0-9]+_t|uint[0-9]+_t)$)");
                check(transfer.match(name, &match),
                      "unmodelled transfer helper");
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
            } else if (name.starts_with("llvm.smin.") ||
                       name.starts_with("llvm.umin.") ||
                       name.starts_with("llvm.smax.") ||
                       name.starts_with("llvm.umax.")) {
              if (args.size() == 2 && args[0].known() && args[1].known())
                result = (name.contains("min")
                              ? args[0].integer() < args[1].integer()
                              : args[0].integer() > args[1].integer())
                             ? args[0]
                             : args[1];
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
                result = Scalar(APInt(64, name.contains("SBITSET1")
                                              ? value | mask
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
              vector.add(*call);
              vectorCall = true;
            } else if (!isVector && kernel &&
                       (kernel->extra(name, args) ||
                        (external && external(name, args, *kernel)))) {
            } else {
              check(lowered && !isVector && kernel &&
                        !callee->isDeclaration() &&
                        callee->getCallingConv() == CallingConv::C &&
                        callee->getReturnType()->isVoidTy(),
                    "unmodelled call: " + name.str());
              helper(kernel);
            }
          } else if (isa<FreezeInst>(i))
            result = val(i.getOperand(0));
          else if (isa<AllocaInst>(i))
            result = Scalar(Pointer{&i, lowered ? depth : 0, 0});
          else if (auto *store = dyn_cast<StoreInst>(&i)) {
            Scalar pointer = val(store->getPointerOperand());
            auto *p = std::get_if<Pointer>(&pointer.data);
            if (p && p->allocation)
              memory[*p] = val(store->getValueOperand());
            else
              check(!isVector && store->getPointerAddressSpace() == 6,
                    "unmodelled scalar write");
          } else if (auto *load = dyn_cast<LoadInst>(&i)) {
            Scalar pointer = val(load->getPointerOperand());
            if (auto *p = std::get_if<Pointer>(&pointer.data))
              if (memory.count(*p))
                result = memory[*p];
            if (hook && hook->load)
              result = hook->load(*load);
          } else if (auto *select = dyn_cast<SelectInst>(&i)) {
            auto condition = val(select->getCondition());
            if (condition.known())
              result = val(condition.integer() ? select->getTrueValue()
                                               : select->getFalseValue());
          } else if (auto *extract = dyn_cast<ExtractValueInst>(&i)) {
            result = val(extract->getAggregateOperand());
            for (unsigned index : extract->getIndices()) {
              auto *values = std::get_if<std::vector<Scalar>>(&result.data);
              Scalar selected = values && index < values->size()
                                    ? (*values)[index]
                                    : Scalar();
              result = std::move(selected);
            }
          } else if (auto *ins = dyn_cast<InsertValueInst>(&i))
            result = insert(val(ins->getAggregateOperand()), ins->getIndices(),
                            val(ins->getInsertedValueOperand()));
          else if (auto *cmp = dyn_cast<ICmpInst>(&i)) {
            auto a = val(cmp->getOperand(0)), b = val(cmp->getOperand(1));
            check(cmp->getOperand(0)->getType()->isIntegerTy(),
                  "unmodelled compare");
            if (a.known() && b.known()) {
              unsigned bits =
                  cmp->getOperand(0)->getType()->getIntegerBitWidth();
              result = Scalar(int64_t(ICmpInst::compare(APInt(bits, a.bits()),
                                                        APInt(bits, b.bits()),
                                                        cmp->getPredicate())));
            }
          } else if (auto *binary = dyn_cast<BinaryOperator>(&i)) {
            auto a = val(binary->getOperand(0)), b = val(binary->getOperand(1));
            if (i.getType()->isIntegerTy() && a.known() && b.known()) {
              auto *left = ConstantInt::get(i.getType(), a.bits());
              auto *right = ConstantInt::get(i.getType(), b.bits());
              auto *folded = dyn_cast_or_null<ConstantInt>(
                  ConstantFoldBinaryInstruction(i.getOpcode(), left, right));
              check(folded, "undefined scalar arithmetic");
              result = Scalar(folded->getValue());
            }
          } else if (isa<PtrToIntInst>(i) && lowered) {
            auto pointer = val(i.getOperand(0));
            if (auto *p = std::get_if<Pointer>(&pointer.data))
              if (!p->allocation)
                result = Scalar(p->offset);
          } else if (auto *cast = dyn_cast<CastInst>(&i)) {
            auto source = val(cast->getOperand(0));
            if (auto *a = std::get_if<APInt>(&source.data)) {
              if (i.getType()->isIntegerTy())
                result = Scalar(
                    i.getOpcode() == Instruction::ZExt
                        ? a->zextOrTrunc(i.getType()->getIntegerBitWidth())
                        : a->sextOrTrunc(i.getType()->getIntegerBitWidth()));
              else if (i.getType()->isFloatingPointTy())
                result = source;
            } else if (auto *a = std::get_if<double>(&source.data)) {
              if (i.getType()->isFloatingPointTy())
                result = source;
              else if (i.getType()->isIntegerTy() && *a >= double(INT64_MIN) &&
                       *a < -double(INT64_MIN))
                result = Scalar(APInt(i.getType()->getIntegerBitWidth(),
                                      uint64_t(int64_t(*a))));
            }
          } else if (auto *gep = dyn_cast<GetElementPtrInst>(&i)) {
            auto pointer = val(gep->getPointerOperand());
            if (auto *p = std::get_if<Pointer>(&pointer.data))
              if (lowered && !p->allocation) {
                check(gep->getNumIndices() == 1 &&
                          gep->getSourceElementType()->isSingleValueType(),
                      "unsupported UB GEP layout");
                auto index = val(gep->getOperand(1));
                if (index.known())
                  result = Scalar(Pointer{
                      nullptr, 0,
                      p->offset +
                          index.integer() *
                              int64_t(module.getDataLayout().getTypeAllocSize(
                                  gep->getSourceElementType()))});
              }
          } else
            check(isa<FCmpInst, ExtractElementInst, InsertElementInst,
                      ShuffleVectorInst>(i),
                  "unmodelled instruction: " + printValue(i));
          if (!i.getType()->isVoidTy())
            env[&i] = result;
          if (isVector && !vectorCall)
            vector.scalar(i);
        } catch (const std::runtime_error &error) {
          throw std::runtime_error(std::string(error.what()) + " in " +
                                   fn.getName().str() + ": " + printValue(i));
        }
      }
      check(next, "missing scalar terminator");
      previous = block;
      block = next;
    }
  }
};

struct Work {
  double instructions = 0, gmLoads = 0, gmStores = 0, ubLoads = 0, ubStores = 0,
         ubBytes = 0, barriers = 0;
  void record(const Instruction &i) {
    if (!isa<PHINode, ReturnInst, BitCastInst>(i))
      ++instructions;
    if (isa<LoadInst, StoreInst>(i)) {
      bool load = isa<LoadInst>(i);
      Type *type = load ? i.getType() : i.getOperand(0)->getType();
      auto *pointer = load ? i.getOperand(0) : i.getOperand(1);
      unsigned space = pointer->getType()->getPointerAddressSpace(),
               bits = type->getScalarSizeInBits();
      check((space == 1 || space == 6) && !type->isVectorTy() && bits &&
                bits <= 64,
            "unsupported SIMT memory access");
      if (space == 1)
        (load ? gmLoads : gmStores)++;
      else {
        (load ? ubLoads : ubStores)++;
        ubBytes += (bits + 7) / 8;
      }
    }
    if (auto *call = dyn_cast<CallBase>(&i))
      if (call->getCalledFunction() &&
          call->getCalledFunction()->getName() == "llvm.hivm.sync.workitems")
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
  Cost trace(Function &wrapper, Args input) {
    std::vector<Scalar> bound;
    for (const auto &a : wrapper.args()) {
      check(a.getType()->isPointerTy() ||
                std::holds_alternative<APInt>(input[a.getArgNo()].data),
            "dynamic SIMT helper scalar");
      bound.push_back(a.getType()->isPointerTy() ? Scalar()
                                                 : input[a.getArgNo()]);
    }
    std::string key = wrapper.getName().str();
    for (const auto &a : bound)
      key += ";" + scalarKey(a);
    if (cache.count(key))
      return cache.at(key);
    Function *target = nullptr;
    uint64_t info = 0;
    unsigned infos = 0, launches = 0, steps = 0;
    for (auto &block : wrapper)
      for (auto &i : block) {
        check(!isa<BranchInst, SwitchInst, LoadInst, StoreInst>(i),
              "SIMT wrapper has unresolved control or memory");
        steps += !isa<ReturnInst>(i);
        if (auto *call = dyn_cast<CallBase>(&i)) {
          auto *fn = call->getCalledFunction();
          check(fn, "indirect SIMT wrapper call");
          if (fn->getCallingConv() == 1022) {
            target = fn;
            ++launches;
          } else if (fn->getName() == "llvm.hivm.store.vfsimt.info") {
            check(call->arg_size() == 1 &&
                      isa<ConstantInt>(call->getArgOperand(0)),
                  "dynamic SIMT dimensions");
            info = cast<ConstantInt>(call->getArgOperand(0))->getZExtValue();
            ++infos;
          } else
            check(simtIntrinsic(fn->getName()) ||
                      fn->getName() == "llvm.hivm.BARRIER",
                  "unmodelled SIMT wrapper call");
        }
      }
    check(infos == 1 && launches == 1 && target && !target->isDeclaration() &&
              !(info >> 48),
          "expected one static SIMT launch");
    std::array<int64_t, 3> dims{int64_t(info & 65535),
                                int64_t((info >> 16) & 65535),
                                int64_t((info >> 32) & 65535)};
    check(*std::min_element(dims.begin(), dims.end()) > 0 &&
              dims[0] * dims[1] * dims[2] <= 4096,
          "SIMT launch exceeds bounded domain");
    std::vector<Scalar> arguments;
    Hook capture;
    capture.call = [&](StringRef name, Args args, Scalar &) {
      if (name == target->getName()) {
        check(arguments.empty(), "multiple SIMT launches");
        arguments.assign(args.begin(), args.end());
      }
      return name == target->getName() ||
             name == "llvm.hivm.store.vfsimt.info" ||
             name == "llvm.hivm.BARRIER";
    };
    KernelPlan dummy(analyzer.profile);
    analyzer.trace(wrapper, bound, 0, &dummy, {}, &capture);
    check(arguments.size() == target->arg_size(), "SIMT launch ABI");
    for (const auto &a : target->args())
      check(a.getType()->isPointerTy() ||
                std::holds_alternative<APInt>(arguments[a.getArgNo()].data),
            "dynamic SIMT scalar");
    std::map<int, std::set<std::pair<const Instruction *, unsigned>>> warps;
    Work total, issued;
    for (int64_t x = 0; x < dims[0]; ++x)
      for (int64_t y = 0; y < dims[1]; ++y)
        for (int64_t z = 0; z < dims[2]; ++z) {
          std::array<int64_t, 3> coord{x, y, z};
          auto &events = warps[(x + dims[0] * (y + dims[1] * z)) / 32];
          Work upper;
          bool indirect = wrapper.getName().starts_with("indirect_load_");
          for (int mask = indirect ? 0 : -1; mask < (indirect ? 2 : 0);
               ++mask) {
            Work work;
            Hook hook;
            hook.record = [&](const Instruction &i, unsigned visit) {
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
            hook.load = [&](const LoadInst &i) {
              return mask >= 0 && i.getType()->isIntegerTy(8) &&
                             i.getPointerAddressSpace() == 6
                         ? Scalar(int64_t(mask))
                         : Scalar();
            };
            KernelPlan plan(analyzer.profile);
            analyzer.trace(*target, arguments, 0, &plan, {}, &hook);
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
    auto *fn = analyzer.module.getFunction(name);
    check(fn && !fn->isDeclaration() && fn->arg_size() == args.size(),
          "SIMT helper ABI mismatch");
    auto cost = trace(*fn, args);
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
  LLVMContext context;
  bool vv = text(config, "mode") == "vv",
       autoBlock = config.getBoolean("auto_blockify").value_or(false);
  auto coreCount = config.getInteger("cores"),
       blockCount = config.getInteger("blocks");
  check(coreCount && blockCount && *coreCount > 0 && *coreCount <= 4096 &&
            *blockCount > 0 &&
            *blockCount <= std::numeric_limits<unsigned>::max() &&
            (vv || *blockCount <= 10000),
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
  std::map<std::string, std::unique_ptr<Module>> modules;
  for (const auto &[role, source] : object(*config.get("modules"))) {
    check(bool(source.getAsString()), "missing LLVM module text");
    modules[role.str()] = parse(*source.getAsString(), context);
  }
  std::string name = text(config, "name");
  auto entry = [&](StringRef role) -> Function & {
    check(modules.count(role.str()), "LLVM core role missing");
    std::string suffix = vv || modules.size() == 1 ? "" : "_mix_" + role.str();
    auto *fn = modules.at(role.str())->getFunction(name + suffix);
    check(fn && !fn->isDeclaration() && fn->getCallingConv() != 1023,
          "scalar entry missing");
    if (!vv && modules.size() == 1)
      check(fn->hasFnAttribute("cube-only-kernel"),
            "single CV entry is not Cube");
    return *fn;
  };
  Object result{{"status", "STATIC_COST_ESTIMATED"},
                {"auto_blockify", autoBlock}};
  result["logical_grid"] = Array(array(config, "grid"));
  if (vv) {
    Function &fn = entry("aiv");
    Analyzer analyzer(*modules.at("aiv"), profile, true);
    std::unique_ptr<SimtModel> simt;
    if (const auto *mixed = config.getObject("mixed")) {
      simt = std::make_unique<SimtModel>(SimtModel{analyzer, *mixed, {}});
    }
    analyzer.external = [&](StringRef callee, Args args, Plan &plan) {
      return simt && simt->call(callee, args, plan);
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
      check(slot < ends.end(), "invalid automatic core dispatch");
      *slot += graph.finish();
    }
    result["score"] =
        std::max(*std::max_element(ends.begin(), ends.end()), bytes / 512);
    result["mode"] = modules.size() == 2 ? "Cube+Vector" : "Cube";
    result["cube_core_count"] = int64_t(cores);
    result["launched_groups"] = int64_t(blocks);
    result["counts"] = Object{{"useful_flops", flops}, {"mma_calls", mmas}};
  }
  return result;
}
std::string analyzeBlockCandidate(StringRef request) {
  auto parsed = json::parse(request);
  if (!parsed)
    throw std::runtime_error(toString(parsed.takeError()));
  std::string output;
  raw_string_ostream stream(output);
  stream << json::Value(analyzeBlockCandidate(object(*parsed)));
  return output;
}
} // namespace mlir::triton::ascend
