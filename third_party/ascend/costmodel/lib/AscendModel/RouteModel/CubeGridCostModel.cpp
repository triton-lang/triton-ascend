#include "AscendModel/RouteModel/CubeGridCostModel.h"

#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/Interfaces/FunctionInterfaces.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/ADT/SmallPtrSet.h"
#include <algorithm>
#include <array>
#include <cmath>
#include <limits>

using namespace mlir;
using namespace mlir::ascend;

bool CubeGridParameters::isValid() const {
  const double values[] = {inputBytesPerCycle,
                           transactionStartup,
                           cyclesPerQuantum,
                           fillBase,
                           fillInput,
                           fillMmad,
                           fillFix,
                           occupancyBase,
                           occupancyInput,
                           occupancyFix,
                           serializedInput,
                           serializedMmad,
                           serializedFix,
                           waveBase,
                           waveInput,
                           waveMmad,
                           waveFix};
  return coreCount == 28 && inputBytesPerCycle > 0 &&
         std::all_of(std::begin(values), std::end(values),
                     [](double v) { return std::isfinite(v) && v >= 0; });
}

bool mlir::ascend::isCubeGridCalibrationDomain(const CubeGridGeometry &g) {
  // Measured development geometries (641 points), not an arbitrary box.
  // Eight extreme geometries also passed independent fresh count validation.
  constexpr int64_t shapes[][3] = {
      {16, 16, 16},    {16, 32, 2048},  {16, 64, 256},   {16, 64, 1024},
      {16, 128, 512},  {16, 256, 256},  {32, 16, 2048},  {32, 32, 32},
      {32, 128, 32},   {32, 128, 256},  {32, 128, 1024}, {32, 256, 128},
      {32, 256, 256},  {32, 512, 128},  {64, 16, 256},   {64, 16, 1024},
      {64, 64, 16},    {64, 64, 64},    {64, 128, 64},   {64, 128, 128},
      {64, 128, 512},  {64, 256, 128},  {64, 256, 256},  {64, 256, 512},
      {64, 512, 64},   {64, 512, 128},  {64, 512, 256},  {128, 16, 512},
      {128, 32, 64},   {128, 32, 128},  {128, 32, 256},  {128, 32, 1024},
      {128, 64, 512},  {128, 128, 32},  {128, 128, 64},  {128, 256, 64},
      {128, 256, 128}, {128, 256, 256}, {256, 16, 256},  {256, 32, 128},
      {256, 32, 256},  {256, 64, 128},  {256, 64, 256},  {256, 64, 512},
      {256, 128, 64},  {256, 128, 128}, {256, 128, 256}, {512, 32, 128},
      {512, 64, 64},   {512, 64, 128},  {512, 64, 256}};
  if (g.programCount < 1 || g.programCount > 140)
    return false;
  return std::any_of(std::begin(shapes), std::end(shapes), [&](const auto &s) {
    return g.m == s[0] && g.n == s[1] && g.k == s[2];
  });
}

double mlir::ascend::evaluateCubeGridCycles(const CubeGridGeometry &g,
                                            const CubeGridParameters &p) {
  if (!p.isValid() || g.m <= 0 || g.n <= 0 || g.k <= 0 || g.programCount <= 0)
    return std::numeric_limits<double>::quiet_NaN();
  double m = g.m, n = g.n, k = g.k;
  double q = std::ceil(m / 16) * std::ceil(n / 16) * std::ceil(k / 16);
  double input = (2 * m * k + 2 * k * n) / p.inputBytesPerCycle / 1000;
  double x = (p.transactionStartup + p.cyclesPerQuantum * q) / 1000;
  double fix = 4 * m * n / p.inputBytesPerCycle / 1000;
  double fill =
      p.fillBase + p.fillInput * input + p.fillMmad * x + p.fillFix * fix;
  double occupancy =
      p.occupancyBase / (1 + p.occupancyInput * input + p.occupancyFix * fix);
  double serial =
      p.serializedInput * input + p.serializedMmad * x + p.serializedFix * fix;
  double wave =
      p.waveBase + p.waveInput * input + p.waveMmad * x + p.waveFix * fix;
  double total = fill + occupancy * std::min(g.programCount, p.coreCount) +
                 serial * g.programCount +
                 wave * ((g.programCount - 1) / p.coreCount);
  return std::isfinite(total) && total > 0
             ? total
             : std::numeric_limits<double>::quiet_NaN();
}

namespace {
// Affine element address in program id and the two local tensor coordinates.
// All coefficients are exact within the admitted, bounded geometry.
struct Address {
  Value base;
  int64_t constant = 0, pid = 0;
  std::array<int64_t, 2> axis = {0, 0};
};

class DirectGemmMatcher {
public:
  llvm::SmallPtrSet<Operation *, 32> consumed;

  std::optional<Address> address(Value value, unsigned depth = 0) {
    if (depth > 32)
      return std::nullopt;
    Type element = value.getType();
    if (auto tensor = dyn_cast<RankedTensorType>(element))
      element = tensor.getElementType();
    auto integerType = dyn_cast<IntegerType>(element);
    if (integerType && integerType.getWidth() != 32 &&
        integerType.getWidth() != 64)
      return std::nullopt;
    auto bounded = [&](Address result) -> std::optional<Address> {
      // Conservative bounds cover every supported geometry and P <= 140.
      // Reject wraparound even if a later operation would cancel it.
      double largest = double(result.constant) + 139.0 * result.pid +
                       2047.0 * (double(result.axis[0]) + result.axis[1]);
      if (integerType && integerType.getWidth() == 32 &&
          largest > std::numeric_limits<int32_t>::max())
        return std::nullopt;
      return result;
    };
    if (auto arg = dyn_cast<BlockArgument>(value)) {
      if (!isa<triton::PointerType>(arg.getType()))
        return std::nullopt;
      return Address{arg};
    }
    Operation *op = value.getDefiningOp();
    if (!op || op->getNumResults() != 1 || op->getNumRegions())
      return std::nullopt;
    consumed.insert(op);
    StringRef name = op->getName().getStringRef();
    if (name == "arith.constant") {
      Attribute attr = op->getAttr("value");
      if (auto dense = dyn_cast_or_null<DenseIntElementsAttr>(attr)) {
        if (!dense.isSplat())
          return std::nullopt;
        attr = IntegerAttr::get(dense.getElementType(),
                                dense.getSplatValue<APInt>());
      }
      auto integer = dyn_cast_or_null<IntegerAttr>(attr);
      if (!integer || integer.getValue().getBitWidth() > 64)
        return std::nullopt;
      int64_t v = integer.getInt();
      if (v < 0 || v > (1 << 24))
        return std::nullopt;
      return Address{{}, v};
    }
    if (auto pid = dyn_cast<triton::GetProgramIdOp>(op)) {
      if (static_cast<int>(pid.getAxis()) != 0)
        return std::nullopt;
      return Address{{}, 0, 1};
    }
    if (name == "tt.make_range") {
      auto start = op->getAttrOfType<IntegerAttr>("start");
      auto end = op->getAttrOfType<IntegerAttr>("end");
      auto type = dyn_cast<RankedTensorType>(value.getType());
      if (!start || !end || start.getInt() != 0 || !type ||
          type.getRank() != 1 || end.getInt() != type.getDimSize(0))
        return std::nullopt;
      return Address{{}, 0, 0, {1, 0}};
    }
    if (op->getNumOperands() == 1 &&
        (name == "tt.splat" || name == "tt.broadcast" ||
         name == "tt.expand_dims")) {
      auto result = address(op->getOperand(0), depth + 1);
      if (!result)
        return std::nullopt;
      auto source = dyn_cast<RankedTensorType>(op->getOperand(0).getType());
      auto target = dyn_cast<RankedTensorType>(value.getType());
      if (!target || target.getRank() > 2)
        return std::nullopt;
      if (name == "tt.splat" && source)
        return std::nullopt;
      if (name == "tt.broadcast") {
        if (!source || source.getRank() != target.getRank())
          return std::nullopt;
        for (unsigned i = 0; i < target.getRank(); ++i)
          if (source.getDimSize(i) != target.getDimSize(i) &&
              (source.getDimSize(i) != 1 || result->axis[i] != 0))
            return std::nullopt;
      }
      if (name == "tt.expand_dims") {
        auto axis = op->getAttrOfType<IntegerAttr>("axis");
        if (!axis || !source || source.getRank() != 1 ||
            (axis.getInt() != 0 && axis.getInt() != 1))
          return std::nullopt;
        if (axis.getInt() == 0)
          result->axis = {0, result->axis[0]};
      }
      return result;
    }
    if (op->getNumOperands() != 2 ||
        (name != "tt.addptr" && name != "arith.addi" && name != "arith.muli"))
      return std::nullopt;
    auto lhs = address(op->getOperand(0), depth + 1);
    auto rhs = address(op->getOperand(1), depth + 1);
    if (!lhs || !rhs || (lhs->base && rhs->base))
      return std::nullopt;
    for (int64_t v : {lhs->constant, lhs->pid, lhs->axis[0], lhs->axis[1],
                      rhs->constant, rhs->pid, rhs->axis[0], rhs->axis[1]})
      if (v > (1LL << 40))
        return std::nullopt;
    if (name == "arith.muli") {
      auto isConstant = [](const Address &a) {
        return !a.base && a.pid == 0 && a.axis[0] == 0 && a.axis[1] == 0;
      };
      if (!isConstant(*rhs))
        std::swap(lhs, rhs);
      if (!isConstant(*rhs) || lhs->base)
        return std::nullopt;
      // Avoid overflow from unsupported arithmetic before comparing offsets.
      int64_t scale = rhs->constant;
      for (int64_t v : {lhs->constant, lhs->pid, lhs->axis[0], lhs->axis[1]})
        if (scale && v > (1LL << 40) / scale)
          return std::nullopt;
      lhs->constant *= scale;
      lhs->pid *= scale;
      for (int64_t &v : lhs->axis)
        v *= scale;
      return bounded(*lhs);
    }
    lhs->base = lhs->base ? lhs->base : rhs->base;
    lhs->constant += rhs->constant;
    lhs->pid += rhs->pid;
    for (unsigned i = 0; i < 2; ++i)
      lhs->axis[i] += rhs->axis[i];
    return bounded(*lhs);
  }

  bool zero(Value value) {
    Operation *op = value.getDefiningOp();
    if (!op || op->getName().getStringRef() != "arith.constant")
      return false;
    auto attr = op->getAttrOfType<DenseFPElementsAttr>("value");
    if (!attr || !attr.isSplat() || !attr.getSplatValue<APFloat>().isZero())
      return false;
    consumed.insert(op);
    return true;
  }
};
} // namespace

std::optional<CubeGridCost> mlir::ascend::matchCubeGridCost(
    Operation *root, int64_t programCount, const CubeGridParameters &parameters,
    bool disjointPointerArguments, std::string &reason) {
  reason.clear();
  auto reject = [&](StringRef message) -> std::optional<CubeGridCost> {
    reason = message.str();
    return std::nullopt;
  };
  if (!disjointPointerArguments)
    return reject("cube_grid_requires_disjoint_pointer_contract");
  if (auto module = dyn_cast_or_null<ModuleOp>(root)) {
    if (!llvm::hasSingleElement(module.getBody()->getOperations()))
      return reject("cube_grid_requires_single_function_module");
    root = &module.getBody()->front();
  }
  auto function = dyn_cast_or_null<FunctionOpInterface>(root);
  if (!function || function.getFunctionBody().empty() ||
      !llvm::hasSingleElement(function.getFunctionBody()))
    return reject("cube_grid_requires_single_block_function");
  Block &block = function.getFunctionBody().front();
  Operation *dot = nullptr, *store = nullptr;
  for (Operation &op : block) {
    if (op.getNumRegions())
      return reject("cube_grid_nested_control_flow");
    if (op.getName().getStringRef() == "tt.dot") {
      if (dot)
        return reject("cube_grid_multiple_dots");
      dot = &op;
    }
    if (op.getName().getStringRef() == "tt.store") {
      if (store)
        return reject("cube_grid_multiple_stores");
      store = &op;
    }
  }
  if (!dot || !store || dot->getNumOperands() != 3 ||
      dot->getNumResults() != 1 || store->getNumOperands() != 2 ||
      store->getOperand(1) != dot->getResult(0))
    return reject("cube_grid_requires_direct_dot_store");
  auto a = dyn_cast<RankedTensorType>(dot->getOperand(0).getType());
  auto b = dyn_cast<RankedTensorType>(dot->getOperand(1).getType());
  auto c = dyn_cast<RankedTensorType>(dot->getResult(0).getType());
  if (!a || !b || !c || a.getRank() != 2 || b.getRank() != 2 ||
      c.getRank() != 2 || !a.getElementType().isF16() ||
      !b.getElementType().isF16() || !c.getElementType().isF32() ||
      a.getDimSize(1) != b.getDimSize(0) ||
      c.getDimSize(0) != a.getDimSize(0) || c.getDimSize(1) != b.getDimSize(1))
    return reject("cube_grid_requires_f16_f32_matrix_geometry");
  CubeGridGeometry g{a.getDimSize(0), b.getDimSize(1), a.getDimSize(1),
                     programCount};
  if (!isCubeGridCalibrationDomain(g) || !parameters.isValid())
    return reject("cube_grid_outside_calibration_domain");
  DirectGemmMatcher matcher;
  if (!matcher.zero(dot->getOperand(2)))
    return reject("cube_grid_nonzero_accumulator");
  Operation *loads[2] = {dot->getOperand(0).getDefiningOp(),
                         dot->getOperand(1).getDefiningOp()};
  for (Operation *load : loads) {
    if (!load || load->getName().getStringRef() != "tt.load" ||
        load->getNumOperands() != 1)
      return reject("cube_grid_requires_unmasked_direct_loads");
    if (auto volatileAttr = load->getAttrOfType<BoolAttr>("isVolatile"))
      if (volatileAttr.getValue())
        return reject("cube_grid_volatile_load");
    matcher.consumed.insert(load);
  }
  for (Operation *memory : {loads[0], loads[1], store}) {
    if (auto cache = memory->getAttrOfType<triton::CacheModifierAttr>("cache"))
      if (cache.getValue() != triton::CacheModifier::NONE)
        return reject("cube_grid_nondefault_memory_policy");
    if (auto evict = memory->getAttrOfType<triton::EvictionPolicyAttr>("evict"))
      if (evict.getValue() != triton::EvictionPolicy::NORMAL)
        return reject("cube_grid_nondefault_memory_policy");
    if (auto boundary =
            memory->getAttrOfType<DenseI32ArrayAttr>("boundaryCheck"))
      if (!boundary.empty())
        return reject("cube_grid_boundary_checked_memory");
    if (memory->hasAttr("padding"))
      return reject("cube_grid_padded_memory");
  }
  std::optional<Address> addresses[] = {
      matcher.address(loads[0]->getOperand(0)),
      matcher.address(loads[1]->getOperand(0)),
      matcher.address(store->getOperand(0))};
  int64_t rows[] = {g.m, g.k, g.m}, columns[] = {g.k, g.n, g.n};
  for (unsigned i = 0; i < 3; ++i) {
    const auto &addr = addresses[i];
    if (!addr || !addr->base || addr->constant != 0 ||
        addr->pid != rows[i] * columns[i] ||
        addr->axis != std::array<int64_t, 2>{columns[i], 1})
      return reject("cube_grid_requires_dense_per_program_addresses");
    for (unsigned j = 0; j < i; ++j)
      if (addr->base == addresses[j]->base)
        return reject("cube_grid_aliased_pointer_arguments");
  }
  matcher.consumed.insert(dot);
  matcher.consumed.insert(store);
  for (Operation &op : block)
    if (!matcher.consumed.contains(&op) &&
        !(op.hasTrait<OpTrait::IsTerminator>() && op.getNumOperands() == 0))
      return reject("cube_grid_unowned_operation");
  double cycles = evaluateCubeGridCycles(g, parameters);
  if (!std::isfinite(cycles) || cycles <= 0)
    return reject("cube_grid_invalid_prediction");
  return CubeGridCost{g, cycles};
}
