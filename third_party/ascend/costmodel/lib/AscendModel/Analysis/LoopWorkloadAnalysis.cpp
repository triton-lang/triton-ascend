//===- LoopWorkloadAnalysis.cpp - Describe runtime loop work
//----------------===//
#include "AscendModel/Analysis/LoopWorkloadAnalysis.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/ScopeExit.h"
#include "llvm/ADT/StringSet.h"
#include "llvm/Support/raw_ostream.h"

#include <limits>

using namespace mlir;
using namespace mlir::ascend;
namespace {
using Json = llvm::json::Value;
using Object = llvm::json::Object;
using Array = llvm::json::Array;

static std::string location(Operation *op) {
  std::string text;
  llvm::raw_string_ostream os(text);
  op->getLoc().print(os);
  return text;
}

static unsigned bits(Type type) {
  if (auto integer = dyn_cast<IntegerType>(type))
    return integer.getWidth();
  if (isa<IndexType, triton::PointerType>(type))
    return 64;
  return 0;
}

static Json constant(int64_t value, unsigned width = 1) {
  return Object{{"op", "const"}, {"bits", width}, {"value", value}};
}

static Json node(llvm::StringRef op, unsigned width, Array args) {
  return Object{{"op", op.str()}, {"bits", width}, {"args", std::move(args)}};
}

static std::optional<int64_t> constantValue(const Json &expr) {
  auto *object = expr.getAsObject();
  if (object && object->getString("op") == "const")
    return object->getInteger("value");
  return std::nullopt;
}

static std::optional<int64_t> launchBoundInteger(Value value) {
  auto argument = dyn_cast<BlockArgument>(value);
  if (!argument || !isa<IntegerType, IndexType>(value.getType()))
    return std::nullopt;
  auto function =
      dyn_cast_or_null<FunctionOpInterface>(argument.getOwner()->getParentOp());
  if (!function || argument.getOwner() != &function.getFunctionBody().front())
    return std::nullopt;
  auto attr = function.getArgAttrOfType<IntegerAttr>(
      argument.getArgNumber(), "ascend.costmodel.runtime_value");
  const unsigned width = bits(value.getType());
  if (!attr || width == 0 || width > 64 || !attr.getValue().isSignedIntN(width))
    return std::nullopt;
  return attr.getValue().sextOrTrunc(64).getSExtValue();
}

static std::optional<int64_t> integerConstant(Value value) {
  if (auto bound = launchBoundInteger(value))
    return bound;
  Operation *definition = value.getDefiningOp();
  if (!definition || definition->getName().getStringRef() != "arith.constant")
    return std::nullopt;
  auto attr = definition->getAttrOfType<IntegerAttr>("value");
  if (!attr || attr.getValue().getBitWidth() > 64)
    return std::nullopt;
  return attr.getValue().getSExtValue();
}

// A signed minimum supplies a proven upper bound even when its other operand
// is runtime-dependent.
static std::optional<int64_t> signedUpperBound(Value value,
                                               unsigned depth = 0) {
  if (auto constant = integerConstant(value))
    return constant;
  Operation *definition = value.getDefiningOp();
  if (!definition || depth >= 8 ||
      definition->getName().getStringRef() != "arith.minsi" ||
      definition->getNumOperands() != 2)
    return std::nullopt;
  auto left = signedUpperBound(definition->getOperand(0), depth + 1);
  auto right = signedUpperBound(definition->getOperand(1), depth + 1);
  if (left && right)
    return std::min(*left, *right);
  return left ? left : right;
}

static Operation *stripIntegerCastDefinition(Value value) {
  Operation *definition = value.getDefiningOp();
  for (unsigned depth = 0; definition && depth < 8; ++depth) {
    const llvm::StringRef name = definition->getName().getStringRef();
    if ((name == "arith.index_cast" || name == "arith.extsi" ||
         name == "arith.extui" || name == "arith.trunci") &&
        definition->getNumOperands() == 1) {
      definition = definition->getOperand(0).getDefiningOp();
      continue;
    }
    break;
  }
  return definition;
}

static bool dependsOnProgramQuery(Value value, llvm::StringRef query,
                                  unsigned depth = 0) {
  if (auto argument = dyn_cast<BlockArgument>(value)) {
    Operation *parent = argument.getOwner()->getParentOp();
    if (parent && parent->getName().getStringRef() == "scf.for" &&
        argument.getArgNumber() == 0 && parent->getNumOperands() >= 1 &&
        depth < 8)
      return dependsOnProgramQuery(parent->getOperand(0), query, depth + 1);
    return false;
  }
  Operation *definition = stripIntegerCastDefinition(value);
  if (!definition || depth >= 8)
    return false;
  if (definition->getName().getStringRef() == query)
    return true;
  const llvm::StringRef name = definition->getName().getStringRef();
  if (name != "arith.remsi" && name != "arith.remui" && name != "arith.addi" &&
      name != "arith.subi" && name != "arith.muli" && name != "arith.divsi" &&
      name != "arith.divui" && name != "arith.ceildivsi" &&
      name != "arith.ceildivui" && name != "arith.minsi" &&
      name != "arith.minui" && name != "arith.maxsi" && name != "arith.maxui")
    return false;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return dependsOnProgramQuery(operand, query, depth + 1);
  });
}

static std::optional<int64_t>
gridStrideTripCount(Operation *loop, int64_t logicalProgramCountHint) {
  if (!loop || loop->getName().getStringRef() != "scf.for" ||
      loop->getNumOperands() < 3 || logicalProgramCountHint <= 0)
    return std::nullopt;
  const auto upper = integerConstant(loop->getOperand(1));
  if (!upper)
    return std::nullopt;
  if (*upper <= 0)
    return int64_t{0};
  if (!dependsOnProgramQuery(loop->getOperand(0), "tt.get_program_id") ||
      !dependsOnProgramQuery(loop->getOperand(2), "tt.get_num_programs"))
    return std::nullopt;
  const uint64_t extent = static_cast<uint64_t>(*upper);
  const uint64_t programs = static_cast<uint64_t>(logicalProgramCountHint);
  return static_cast<int64_t>(extent / programs + (extent % programs != 0));
}

static std::optional<int64_t> tripCountFromBounds(int64_t lower, int64_t upper,
                                                  int64_t step) {
  if (step <= 0)
    return std::nullopt;
  const __int128 distance = static_cast<__int128>(upper) - lower;
  const __int128 count = distance <= 0 ? 0 : (distance + step - 1) / step;
  if (count > std::numeric_limits<int64_t>::max())
    return std::nullopt;
  return static_cast<int64_t>(count);
}

// Short circuiting is essential: conditions can themselves load metadata.
static Json conjunction(Json lhs, Json rhs) {
  if (auto value = constantValue(lhs))
    return *value ? std::move(rhs) : constant(0);
  return node("select", 1, Array{std::move(lhs), std::move(rhs), constant(0)});
}

class Extractor {
public:
  bool supported = true;
  bool readsDevice = false;
  bool usesLaunchBounds = false;
  Array limitations;
  Array memoryReads;
  llvm::DenseSet<unsigned> arguments;

  Json unknown(llvm::StringRef reason, unsigned width = 0) {
    supported = false;
    if (seenLimitations.insert(reason).second)
      limitations.push_back(reason.str());
    return Object{{"op", "unknown"}, {"bits", width}, {"reason", reason.str()}};
  }

  Json expression(Value value) {
    unsigned width = bits(value.getType());
    if (remainingValueVisits == 0 || valueDepth >= 64)
      return unknown("expression expansion exceeds visit/depth limit", width);
    --remainingValueVisits;
    ++valueDepth;
    auto restoreDepth = llvm::make_scope_exit([&] { --valueDepth; });
    if (!width || width > 64)
      return unknown("non-scalar or unsupported integer width", width);
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      Operation *owner = arg.getOwner()->getParentOp();
      auto name = owner->getName().getStringRef();
      if ((name != "tt.func" && name != "func.func") ||
          arg.getOwner() != &arg.getOwner()->getParent()->front())
        return unknown(
            "non-entry block argument or loop-carried/induction value", width);
      arguments.insert(arg.getArgNumber());
      if (auto bound = launchBoundInteger(value)) {
        usesLaunchBounds = true;
        return Object{{"op", "const"},
                      {"bits", width},
                      {"value", *bound},
                      {"source", "launch_scalar_binding"},
                      {"argument_index", arg.getArgNumber()}};
      }
      Object result{{"op", "arg"},
                    {"bits", width},
                    {"index", arg.getArgNumber()},
                    {"arg_type", "integer"}};
      if (auto ptr = dyn_cast<triton::PointerType>(value.getType())) {
        result["arg_type"] = "pointer";
        result["element_bits"] = bits(ptr.getPointeeType());
      }
      return result;
    }
    Operation *op = value.getDefiningOp();
    if (!op || !activeValues.insert(value).second)
      return unknown("cyclic value expression", width);
    Json result = operationExpression(op, width);
    activeValues.erase(value);
    return result;
  }

  Json entryGuard(Operation *target) {
    SmallVector<Operation *> ancestors;
    for (Operation *op = target; op; op = op->getParentOp()) {
      ancestors.push_back(op);
      auto name = op->getName().getStringRef();
      if (name == "tt.func" || name == "func.func")
        break;
    }
    Json guard = constant(1);
    for (Operation *op : llvm::reverse(ancestors)) {
      auto name = op->getName().getStringRef();
      if (name == "tt.func" || name == "func.func")
        continue;
      Operation *parent = op->getParentOp();
      if (parent && parent->getName().getStringRef() == "scf.if") {
        Json cond = expression(parent->getOperand(0));
        if (op->getParentRegion() == &parent->getRegion(1))
          cond = node("select", 1,
                      Array{std::move(cond), constant(0), constant(1)});
        guard = conjunction(std::move(guard), std::move(cond));
      } else if (parent) {
        auto parentName = parent->getName().getStringRef();
        if (parentName != "tt.func" && parentName != "func.func" &&
            parentName != "scf.for")
          guard = conjunction(std::move(guard),
                              unknown("unsupported enclosing region control"));
      }
      llvm::DenseSet<Block *> activeBlocks;
      unsigned budget = 256;
      guard = conjunction(std::move(guard),
                          blockGuard(op->getBlock(), activeBlocks, budget));
    }
    return guard;
  }

private:
  llvm::StringSet<> seenLimitations;
  llvm::DenseSet<Value> activeValues;
  llvm::DenseSet<Operation *> recordedLoads;
  unsigned remainingValueVisits = 512;
  unsigned valueDepth = 0;

  Json operationExpression(Operation *op, unsigned width) {
    auto name = op->getName().getStringRef();
    if (name == "arith.constant") {
      auto attr = op->getAttrOfType<IntegerAttr>("value");
      if (attr && attr.getValue().getBitWidth() <= 64)
        return constant(attr.getValue().getSExtValue(), width);
      return unknown("unsupported constant", width);
    }
    if (name == "tt.get_program_id") {
      auto axis = op->getAttrOfType<IntegerAttr>("axis");
      if (axis && axis.getInt() >= 0 && axis.getInt() <= 2)
        return Object{{"op", "pid"}, {"bits", width}, {"axis", axis.getInt()}};
      return unknown("unsupported program-id axis", width);
    }
    if (auto flags = op->getAttr("overflowFlags")) {
      std::string text;
      llvm::raw_string_ostream os(text);
      os << flags;
      if (!llvm::StringRef(text).contains("none"))
        return unknown("overflow flags require poison proof", width);
    }
    if (name == "tt.load") {
      auto isVolatile = op->getAttrOfType<BoolAttr>("isVolatile");
      if ((isVolatile && isVolatile.getValue()) || op->hasAttr("volatile") ||
          op->getNumOperands() != 1 ||
          !isa<IntegerType, IndexType>(op->getResult(0).getType()))
        return unknown("masked, volatile or non-integer load", width);
      readsDevice = true;
      Json address = expression(op->getOperand(0));
      if (recordedLoads.insert(op).second)
        memoryReads.push_back(Object{{"source_location", location(op)},
                                     {"address", address},
                                     {"element_bits", width},
                                     {"requires_readonly_snapshot", true}});
      return node("load", width, Array{std::move(address)});
    }
    if (name == "tt.addptr") {
      Array args;
      for (Value operand : op->getOperands())
        args.push_back(expression(operand));
      auto ptr = dyn_cast<triton::PointerType>(op->getResult(0).getType());
      if (!ptr || args.size() != 2)
        return unknown("unsupported pointer arithmetic", width);
      return Object{{"op", "addptr"},
                    {"bits", width},
                    {"args", std::move(args)},
                    {"element_bits", bits(ptr.getPointeeType())}};
    }
    if (name == "arith.cmpi") {
      auto cmp = dyn_cast<arith::CmpIOp>(op);
      if (!cmp)
        return unknown("invalid integer comparison", width);
      return Object{
          {"op", "cmpi"},
          {"bits", width},
          {"predicate",
           arith::stringifyCmpIPredicate(cmp.getPredicate()).str()},
          {"args", Array{expression(cmp.getLhs()), expression(cmp.getRhs())}}};
    }
    static const llvm::StringSet<> supportedOps{
        "arith.addi",      "arith.subi",       "arith.muli",
        "arith.minsi",     "arith.maxsi",      "arith.minui",
        "arith.maxui",     "arith.divsi",      "arith.divui",
        "arith.ceildivsi", "arith.ceildivui",  "arith.remsi",
        "arith.remui",     "arith.extsi",      "arith.extui",
        "arith.trunci",    "arith.index_cast", "arith.index_castui",
        "arith.andi",      "arith.ori",        "arith.xori",
        "arith.select"};
    if (!supportedOps.contains(name))
      return unknown((llvm::Twine("unsupported operation: ") + name).str(),
                     width);
    Array args;
    for (Value operand : op->getOperands())
      args.push_back(expression(operand));
    return node(name == "arith.select" ? "select" : name, width,
                std::move(args));
  }

  Json blockGuard(Block *block, llvm::DenseSet<Block *> &active,
                  unsigned &budget) {
    if (block == &block->getParent()->front())
      return constant(1);
    if (budget == 0 || !active.insert(block).second)
      return unknown("cyclic or excessive control-flow paths");
    --budget;
    Json result = constant(0);
    for (Block *pred : block->getPredecessors()) {
      Operation *term = pred->getTerminator();
      Json path = blockGuard(pred, active, budget);
      auto name = term->getName().getStringRef();
      if (name == "cf.cond_br" && term->getNumSuccessors() == 2) {
        // Both successors may name the same block: the condition is irrelevant.
        if (term->getSuccessor(0) != term->getSuccessor(1)) {
          Json cond = expression(term->getOperand(0));
          if (term->getSuccessor(1) == block)
            cond = node("select", 1,
                        Array{std::move(cond), constant(0), constant(1)});
          path = conjunction(std::move(path), std::move(cond));
        }
      } else if (name != "cf.br") {
        path =
            conjunction(std::move(path), unknown("unsupported CFG terminator"));
      }
      result = node("select", 1,
                    Array{std::move(result), constant(1), std::move(path)});
    }
    active.erase(block);
    return result;
  }
};

} // namespace

LoopTripCountInfo mlir::ascend::analyzeLoopTripCount(
    Operation *loop, int64_t logicalProgramCountHint, llvm::StringRef loopId) {
  LoopTripCountInfo info;
  info.loopId = loopId.str();
  info.operation = loop;
  if (!loop) {
    info.workloadUsesFallback = true;
    info.description = Object{{"loop_id", info.loopId},
                              {"operation", "invalid"},
                              {"supported", false},
                              {"workload_count_kind", "fallback"}};
    return info;
  }

  const llvm::StringRef name = loop->getName().getStringRef();
  Extractor extractor;
  Json guard = extractor.entryGuard(loop);
  Json lower = nullptr, upper = nullptr, step = nullptr, trip = nullptr;
  if (name == "scf.for" && !loop->hasAttr("unsignedCmp") &&
      loop->getNumOperands() >= 3) {
    const auto lowerConstant = integerConstant(loop->getOperand(0));
    const auto upperConstant = integerConstant(loop->getOperand(1));
    const auto stepConstant = integerConstant(loop->getOperand(2));
    lower = extractor.expression(loop->getOperand(0));
    upper = extractor.expression(loop->getOperand(1));
    step = extractor.expression(loop->getOperand(2));
    if (lowerConstant && upperConstant && stepConstant)
      info.constantTripCount =
          tripCountFromBounds(*lowerConstant, *upperConstant, *stepConstant);
    if (!info.constantTripCount) {
      const auto upperBound = signedUpperBound(loop->getOperand(1));
      if (lowerConstant && upperBound && stepConstant)
        info.upperBoundTripCount =
            tripCountFromBounds(*lowerConstant, *upperBound, *stepConstant);
    }
    info.gridStrideTripCount =
        gridStrideTripCount(loop, logicalProgramCountHint);
    trip = node("trip_count", 64, Array{lower, upper, step});
    if (stepConstant && *stepConstant <= 0)
      trip = extractor.unknown("non-positive loop step", 64);
  } else {
    if (name == "scf.for" && loop->hasAttr("unsignedCmp") &&
        loop->getNumOperands() >= 3) {
      lower = extractor.expression(loop->getOperand(0));
      upper = extractor.expression(loop->getOperand(1));
      step = extractor.expression(loop->getOperand(2));
      const auto lowerConstant = integerConstant(loop->getOperand(0));
      const auto upperConstant = integerConstant(loop->getOperand(1));
      const auto stepConstant = integerConstant(loop->getOperand(2));
      if (lowerConstant && upperConstant && stepConstant &&
          *lowerConstant >= 0 && *upperConstant >= 0)
        info.constantTripCount =
            tripCountFromBounds(*lowerConstant, *upperConstant, *stepConstant);
    }
    trip =
        extractor.unknown("while or unsigned loop trip count unsupported", 64);
  }

  // Keep exact launch facts distinct from compile-time constants. The AST
  // retains their provenance and the original argument dependency.
  if (extractor.usesLaunchBounds) {
    info.launchBoundTripCount = info.constantTripCount;
    info.constantTripCount.reset();
  }
  llvm::StringRef workloadKind = "fallback";
  if (info.constantTripCount) {
    info.workloadTripCount = *info.constantTripCount;
    workloadKind = "exact_constant";
  } else if (info.launchBoundTripCount) {
    info.workloadTripCount = *info.launchBoundTripCount;
    workloadKind = "exact_launch_bound";
  } else if (info.upperBoundTripCount) {
    info.workloadTripCount = *info.upperBoundTripCount;
    info.workloadUsesUpperBound = info.workloadTripCount != 0;
    workloadKind = "proven_upper_bound";
  } else if (info.gridStrideTripCount) {
    info.workloadTripCount = *info.gridStrideTripCount;
    workloadKind = "grid_stride_hint";
  } else {
    info.workloadUsesFallback = true;
  }

  Array enclosing;
  for (Operation *parent = loop->getParentOp(); parent;
       parent = parent->getParentOp()) {
    auto parentName = parent->getName().getStringRef();
    if (parentName == "scf.for" || parentName == "scf.while")
      enclosing.push_back(Object{{"operation", parentName.str()},
                                 {"source_location", location(parent)}});
  }
  Array arguments;
  SmallVector<unsigned> ordered(extractor.arguments.begin(),
                                extractor.arguments.end());
  llvm::sort(ordered);
  for (unsigned arg : ordered)
    arguments.push_back(arg);
  info.supported = extractor.supported;
  Json effectiveTrip = node("select", 64, Array{guard, trip, constant(0, 64)});
  info.description = Object{
      {"loop_id", info.loopId},
      {"operation", name.str()},
      {"source_location", location(loop)},
      {"lower", std::move(lower)},
      {"upper", std::move(upper)},
      {"step", std::move(step)},
      {"entry_guard", std::move(guard)},
      {"trip_count", std::move(trip)},
      {"effective_trip_count", std::move(effectiveTrip)},
      {"supported", info.supported},
      {"requires_device_values", extractor.readsDevice},
      {"requires_readonly_snapshot", extractor.readsDevice},
      {"argument_dependencies", std::move(arguments)},
      {"memory_reads", std::move(extractor.memoryReads)},
      {"enclosing_loops", std::move(enclosing)},
      {"count_scope", "per_loop_invocation_per_program"},
      {"constant_trip_count",
       info.constantTripCount ? Json(*info.constantTripCount) : Json(nullptr)},
      {"launch_bound_trip_count", info.launchBoundTripCount
                                      ? Json(*info.launchBoundTripCount)
                                      : Json(nullptr)},
      {"upper_bound_trip_count", info.upperBoundTripCount
                                     ? Json(*info.upperBoundTripCount)
                                     : Json(nullptr)},
      {"grid_stride_trip_count", info.gridStrideTripCount
                                     ? Json(*info.gridStrideTripCount)
                                     : Json(nullptr)},
      {"workload_trip_count", info.workloadTripCount},
      {"workload_count_kind", workloadKind.str()},
      {"limitations", std::move(extractor.limitations)}};
  return info;
}

std::vector<LoopTripCountInfo>
mlir::ascend::analyzeLoopTripCountInfos(ModuleOp module,
                                        int64_t logicalProgramCountHint) {
  std::vector<LoopTripCountInfo> result;
  module.walk<WalkOrder::PreOrder>([&](Operation *function) {
    auto functionName = function->getName().getStringRef();
    if (functionName != "tt.func" && functionName != "func.func")
      return;
    auto symbol = function->getAttrOfType<StringAttr>("sym_name");
    if (!symbol)
      return;
    unsigned index = 0;
    function->walk<WalkOrder::PreOrder>([&](Operation *loop) {
      auto name = loop->getName().getStringRef();
      if ((name != "scf.for" && name != "scf.while") ||
          loop->hasAttr("ta.auto_blockify_v1.loop"))
        return;
      const std::string loopId =
          (llvm::Twine(symbol.getValue()) + "/loop" + llvm::Twine(index++))
              .str();
      result.push_back(
          analyzeLoopTripCount(loop, logicalProgramCountHint, loopId));
    });
  });
  return result;
}
