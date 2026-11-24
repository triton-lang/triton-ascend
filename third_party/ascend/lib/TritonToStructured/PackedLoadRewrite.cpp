#include "TritonToStructured/PackedLoadRewrite.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Matchers.h"
#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MathExtras.h"

#include <cstdint>
#include <limits>
#include <numeric>
#include <optional>
#include <utility>

namespace mlir::triton {
namespace {

static bool checkedAdd(int64_t a, int64_t b, int64_t &out) {
  return !__builtin_add_overflow(a, b, &out);
}

static bool checkedSub(int64_t a, int64_t b, int64_t &out) {
  return !__builtin_sub_overflow(a, b, &out);
}

static bool checkedMul(int64_t a, int64_t b, int64_t &out) {
  return !__builtin_mul_overflow(a, b, &out);
}

/// Euclidean floor division for arbitrary signed `a` and positive `d > 0`.
static int64_t floorDivSigned(int64_t a, int64_t d) {
  int64_t q = a / d;
  int64_t r = a % d;
  if (r < 0)
    --q;
  return q;
}

static std::optional<int64_t> getConstantScalarInt(Value v) {
  APInt apVal;
  if (matchPattern(v, m_ConstantInt(&apVal))) {
    if (apVal.getBitWidth() <= 64)
      return apVal.getSExtValue();
  }
  return std::nullopt;
}

static bool isKnownNonNegativeScalar(Value v, unsigned depth = 0) {
  if (!v || depth > 8)
    return false;
  if (auto c = getConstantScalarInt(v))
    return *c >= 0;
  if (isa_and_nonnull<GetProgramIdOp, GetNumProgramsOp>(v.getDefiningOp()))
    return true;
  if (auto blockArg = dyn_cast<BlockArgument>(v)) {
    if (auto forOp = dyn_cast_or_null<scf::ForOp>(
            blockArg.getOwner()->getParentOp())) {
      if (blockArg == forOp.getInductionVar()) {
        auto lb = getConstantScalarInt(forOp.getLowerBound());
        auto step = getConstantScalarInt(forOp.getStep());
        if (lb && *lb >= 0 && step && *step >= 0)
          return true;
      }
    }
    return false;
  }
  Operation *def = v.getDefiningOp();
  if (!def)
    return false;
  if (isa<arith::ExtUIOp, arith::IndexCastUIOp>(def))
    return true;
  if (isa<arith::ExtSIOp, arith::IndexCastOp>(def))
    return isKnownNonNegativeScalar(def->getOperand(0), depth + 1);
  if (auto shrui = dyn_cast<arith::ShRUIOp>(def)) {
    auto s = getConstantScalarInt(shrui.getRhs());
    int64_t bw = shrui.getType().getIntOrFloatBitWidth();
    if (s && *s > 0 && *s < bw)
      return true;
    return isKnownNonNegativeScalar(shrui.getLhs(), depth + 1);
  }
  if (auto shrsi = dyn_cast<arith::ShRSIOp>(def))
    return isKnownNonNegativeScalar(shrsi.getLhs(), depth + 1);
  if (auto andOp = dyn_cast<arith::AndIOp>(def)) {
    if (auto m = getConstantScalarInt(andOp.getRhs()); m && *m >= 0)
      return true;
    if (auto m = getConstantScalarInt(andOp.getLhs()); m && *m >= 0)
      return true;
    return isKnownNonNegativeScalar(andOp.getLhs(), depth + 1) &&
           isKnownNonNegativeScalar(andOp.getRhs(), depth + 1);
  }
  if (auto addOp = dyn_cast<arith::AddIOp>(def))
    return isKnownNonNegativeScalar(addOp.getLhs(), depth + 1) &&
           isKnownNonNegativeScalar(addOp.getRhs(), depth + 1);
  if (auto mulOp = dyn_cast<arith::MulIOp>(def))
    return isKnownNonNegativeScalar(mulOp.getLhs(), depth + 1) &&
           isKnownNonNegativeScalar(mulOp.getRhs(), depth + 1);
  if (auto divOp = dyn_cast<arith::DivSIOp>(def))
    return isKnownNonNegativeScalar(divOp.getLhs(), depth + 1) &&
           isKnownNonNegativeScalar(divOp.getRhs(), depth + 1);
  if (auto remOp = dyn_cast<arith::RemSIOp>(def))
    return isKnownNonNegativeScalar(remOp.getLhs(), depth + 1);
  if (isa<arith::DivUIOp, arith::RemUIOp>(def)) {
    int64_t bw = def->getResult(0).getType().getIntOrFloatBitWidth();
    return bw < 64 && isKnownNonNegativeScalar(def->getOperand(0), depth + 1);
  }
  return false;
}

/// Monomial `coeff * prod_j vars[j]` in Uniform SSA scalars (`vars` sorted).
struct ScalarMonomial {
  SmallVector<Value, 2> vars;
  int64_t coeff = 0;
};

/// Multivariate polynomial `constTerm + sum_k coeff_k * prod_j var_{k,j}` over
/// loop-uniform / tile-uniform SSA scalars.
struct AffineScalarExpr {
  static constexpr size_t kMaxMonomialDegree = 4;
  static constexpr size_t kMaxScalarTerms = 32;

  int64_t constTerm = 0;
  SmallVector<ScalarMonomial, 4> terms;

  bool isZero() const { return constTerm == 0 && terms.empty(); }
  bool isConstant() const { return terms.empty(); }

  bool addMonomial(SmallVector<Value, 2> vars, int64_t coeff) {
    if (coeff == 0)
      return true;
    if (vars.empty())
      return checkedAdd(constTerm, coeff, constTerm);
    if (vars.size() > kMaxMonomialDegree)
      return false;
    llvm::sort(vars, [](Value a, Value b) {
      return a.getAsOpaquePointer() < b.getAsOpaquePointer();
    });
    for (auto &existing : terms) {
      if (existing.vars == vars) {
        if (!checkedAdd(existing.coeff, coeff, existing.coeff))
          return false;
        if (existing.coeff == 0) {
          llvm::erase_if(terms,
                         [](const ScalarMonomial &m) { return m.coeff == 0; });
        }
        return true;
      }
    }
    if (terms.size() >= kMaxScalarTerms)
      return false;
    terms.push_back(ScalarMonomial{std::move(vars), coeff});
    return true;
  }

  bool addScaled(const AffineScalarExpr &other, int64_t scale) {
    if (scale == 0)
      return true;
    int64_t prod = 0;
    if (!checkedMul(other.constTerm, scale, prod) ||
        !checkedAdd(constTerm, prod, constTerm))
      return false;
    for (const ScalarMonomial &m : other.terms) {
      int64_t scaledCoeff = 0;
      if (!checkedMul(m.coeff, scale, scaledCoeff))
        return false;
      if (!addMonomial(m.vars, scaledCoeff))
        return false;
    }
    return true;
  }

  static bool multiply(const AffineScalarExpr &lhs, const AffineScalarExpr &rhs,
                       AffineScalarExpr &out) {
    AffineScalarExpr res;
    if (!checkedMul(lhs.constTerm, rhs.constTerm, res.constTerm))
      return false;
    if (lhs.constTerm != 0) {
      for (const ScalarMonomial &mr : rhs.terms) {
        int64_t c = 0;
        if (!checkedMul(lhs.constTerm, mr.coeff, c) ||
            !res.addMonomial(mr.vars, c))
          return false;
      }
    }
    if (rhs.constTerm != 0) {
      for (const ScalarMonomial &ml : lhs.terms) {
        int64_t c = 0;
        if (!checkedMul(ml.coeff, rhs.constTerm, c) ||
            !res.addMonomial(ml.vars, c))
          return false;
      }
    }
    for (const ScalarMonomial &ml : lhs.terms) {
      for (const ScalarMonomial &mr : rhs.terms) {
        if (ml.vars.size() + mr.vars.size() > kMaxMonomialDegree)
          return false;
        int64_t c = 0;
        if (!checkedMul(ml.coeff, mr.coeff, c))
          return false;
        SmallVector<Value, 2> combined;
        combined.reserve(ml.vars.size() + mr.vars.size());
        combined.append(ml.vars.begin(), ml.vars.end());
        combined.append(mr.vars.begin(), mr.vars.end());
        if (!res.addMonomial(std::move(combined), c))
          return false;
      }
    }
    out = std::move(res);
    return true;
  }

  bool scaleBy(int64_t factor) {
    if (factor == 0) {
      constTerm = 0;
      terms.clear();
      return true;
    }
    if (!checkedMul(constTerm, factor, constTerm))
      return false;
    for (ScalarMonomial &m : terms) {
      if (!checkedMul(m.coeff, factor, m.coeff))
        return false;
    }
    return true;
  }

  bool isDivisibleBy(int64_t d) const {
    if (d <= 0 || (constTerm % d) != 0)
      return false;
    for (const ScalarMonomial &m : terms) {
      if ((m.coeff % d) != 0)
        return false;
    }
    return true;
  }

  AffineScalarExpr divideExact(int64_t d) const {
    AffineScalarExpr res;
    res.constTerm = constTerm / d;
    for (const ScalarMonomial &m : terms)
      res.terms.push_back(ScalarMonomial{m.vars, m.coeff / d});
    return res;
  }

  bool isKnownNonNegative() const {
    if (constTerm < 0)
      return false;
    for (const ScalarMonomial &m : terms) {
      if (m.coeff < 0)
        return false;
      for (Value v : m.vars) {
        if (!isKnownNonNegativeScalar(v))
          return false;
      }
    }
    return true;
  }

  bool requiresI64() const {
    if (constTerm < std::numeric_limits<int32_t>::min() ||
        constTerm > std::numeric_limits<int32_t>::max())
      return true;
    for (const ScalarMonomial &m : terms) {
      if (m.coeff < std::numeric_limits<int32_t>::min() ||
          m.coeff > std::numeric_limits<int32_t>::max())
        return true;
      for (Value val : m.vars) {
        Type ty = val.getType();
        if (ty.isIndex() || ty.getIntOrFloatBitWidth() > 32)
          return true;
      }
    }
    return false;
  }
};

/// Symbolic integer expression over tensor coordinates `(d_0, ..., d_{r-1})`:
///   `scalarBase + sum_{i=0}^{r-2} outerStrides[i] * d_i
///               + sum_{(D, alpha)} alpha * floor(d_{r-1} / D)`
/// where `r = shape.size()` (`r == 0` represents a scalar).
struct SymExpr {
  SmallVector<int64_t, 4> shape;
  AffineScalarExpr scalarBase;
  SmallVector<AffineScalarExpr, 4> outerStrides;
  SmallVector<std::pair<int64_t, int64_t>, 8> innerTerms;

  size_t getRank() const { return shape.size(); }
  bool isScalar() const { return shape.empty(); }

  void canonicalizeInner() {
    if (shape.empty()) {
      innerTerms.clear();
      return;
    }
    int64_t innerExtent = shape.back();
    llvm::sort(innerTerms,
               [](const auto &a, const auto &b) { return a.first < b.first; });
    SmallVector<std::pair<int64_t, int64_t>, 8> merged;
    for (const auto &[div, coeff] : innerTerms) {
      if (coeff == 0 || div <= 0)
        continue;
      if (innerExtent <= 1 || (div >= innerExtent && innerExtent > 0))
        continue;
      if (!merged.empty() && merged.back().first == div) {
        int64_t sum = 0;
        if (!checkedAdd(merged.back().second, coeff, sum)) {
          // Leave unmerged on overflow; caller validates viachecked ops.
          merged.emplace_back(div, coeff);
        } else {
          merged.back().second = sum;
        }
      } else {
        merged.emplace_back(div, coeff);
      }
    }
    llvm::erase_if(merged, [](const auto &p) { return p.second == 0; });
    innerTerms = std::move(merged);
  }

  bool addInnerTerm(int64_t div, int64_t coeff) {
    if (coeff == 0 || shape.empty())
      return true;
    int64_t innerExtent = shape.back();
    if (innerExtent <= 1 || div >= innerExtent)
      return true;
    for (auto &p : innerTerms) {
      if (p.first == div)
        return checkedAdd(p.second, coeff, p.second);
    }
    innerTerms.emplace_back(div, coeff);
    canonicalizeInner();
    return true;
  }

  /// Exact bounds of `sum_{(D, alpha)} alpha * floor(c / D)` on `c in [0, K-1]`.
  std::optional<std::pair<int64_t, int64_t>> getInnerBounds() const {
    if (shape.empty() || innerTerms.empty())
      return std::pair<int64_t, int64_t>{0, 0};
    int64_t K = shape.back();
    if (K <= 1)
      return std::pair<int64_t, int64_t>{0, 0};

    if (K <= 65536) {
      int64_t lo = std::numeric_limits<int64_t>::max();
      int64_t hi = std::numeric_limits<int64_t>::min();
      for (int64_t c = 0; c < K; ++c) {
        int64_t val = 0;
        for (const auto &[div, coeff] : innerTerms) {
          int64_t term = 0;
          if (!checkedMul(coeff, c / div, term) || !checkedAdd(val, term, val))
            return std::nullopt;
        }
        lo = std::min(lo, val);
        hi = std::max(hi, val);
      }
      return std::pair<int64_t, int64_t>{lo, hi};
    }

    int64_t lo = 0, hi = 0;
    for (const auto &[div, coeff] : innerTerms) {
      int64_t maxQ = (K - 1) / div;
      int64_t prod = 0;
      if (!checkedMul(coeff, maxQ, prod))
        return std::nullopt;
      if (prod >= 0) {
        if (!checkedAdd(hi, prod, hi))
          return std::nullopt;
      } else {
        if (!checkedAdd(lo, prod, lo))
          return std::nullopt;
      }
    }
    return std::pair<int64_t, int64_t>{lo, hi};
  }

  /// Static `[min, max]` interval when no runtime SSA scalars are present.
  std::optional<std::pair<int64_t, int64_t>> getStaticBounds() const {
    if (!scalarBase.isConstant())
      return std::nullopt;
    for (const auto &s : outerStrides) {
      if (!s.isConstant())
        return std::nullopt;
    }
    int64_t lo = scalarBase.constTerm;
    int64_t hi = scalarBase.constTerm;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      int64_t maxCoord = std::max<int64_t>(0, shape[i] - 1);
      int64_t prod = 0;
      if (!checkedMul(outerStrides[i].constTerm, maxCoord, prod))
        return std::nullopt;
      if (prod >= 0) {
        if (!checkedAdd(hi, prod, hi))
          return std::nullopt;
      } else {
        if (!checkedAdd(lo, prod, lo))
          return std::nullopt;
      }
    }
    auto innerB = getInnerBounds();
    if (!innerB)
      return std::nullopt;
    if (!checkedAdd(lo, innerB->first, lo) ||
        !checkedAdd(hi, innerB->second, hi))
      return std::nullopt;
    return std::pair<int64_t, int64_t>{lo, hi};
  }

  bool fitsSigned(int64_t bw) const {
    if (bw >= 63)
      return true;
    if (bw <= 0)
      return false;
    auto b = getStaticBounds();
    int64_t minSigned = -(int64_t{1} << (bw - 1));
    int64_t maxSigned = (int64_t{1} << (bw - 1)) - 1;
    return b && b->first >= minSigned && b->second <= maxSigned;
  }

  bool fitsUnsigned(int64_t bw) const {
    if (bw >= 63)
      return isKnownNonNegative();
    if (bw <= 0)
      return false;
    auto b = getStaticBounds();
    int64_t maxUnsigned = (int64_t{1} << bw) - 1;
    return b && b->first >= 0 && b->second <= maxUnsigned;
  }

  bool fitsLosslessNarrow(int64_t bw) const {
    return bw >= 32 || fitsSigned(bw) || fitsUnsigned(bw);
  }

  bool isKnownNonNegative() const {
    if (auto b = getStaticBounds())
      return b->first >= 0;
    auto innerB = getInnerBounds();
    if (!innerB)
      return false;
    int64_t staticMin = 0;
    if (!checkedAdd(scalarBase.constTerm, innerB->first, staticMin))
      return false;
    for (size_t i = 0; i + 1 < shape.size(); ++i) {
      int64_t maxCoord = std::max<int64_t>(0, shape[i] - 1);
      if (outerStrides[i].constTerm < 0) {
        int64_t prod = 0;
        if (!checkedMul(outerStrides[i].constTerm, maxCoord, prod) ||
            !checkedAdd(staticMin, prod, staticMin))
          return false;
      }
      for (const ScalarMonomial &m : outerStrides[i].terms) {
        if (m.coeff < 0)
          return false;
        for (Value val : m.vars) {
          if (!isKnownNonNegativeScalar(val))
            return false;
        }
      }
    }
    if (staticMin < 0)
      return false;
    for (const ScalarMonomial &m : scalarBase.terms) {
      if (m.coeff < 0)
        return false;
      for (Value val : m.vars) {
        if (!isKnownNonNegativeScalar(val))
          return false;
      }
    }
    return true;
  }

  static SymExpr constant(ArrayRef<int64_t> tgtShape, int64_t val) {
    SymExpr e;
    e.shape.assign(tgtShape.begin(), tgtShape.end());
    e.scalarBase.constTerm = val;
    if (tgtShape.size() > 1)
      e.outerStrides.resize(tgtShape.size() - 1);
    return e;
  }

  FailureOr<SymExpr> splatTo(ArrayRef<int64_t> tgtShape) const {
    if (!isScalar())
      return failure();
    SymExpr res;
    res.shape.assign(tgtShape.begin(), tgtShape.end());
    res.scalarBase = scalarBase;
    if (tgtShape.size() > 1)
      res.outerStrides.resize(tgtShape.size() - 1);
    return res;
  }

  FailureOr<SymExpr> expandAxis(int64_t axis) const {
    if (axis < 0 || axis > static_cast<int64_t>(shape.size()))
      return failure();
    if (shape.empty()) {
      SymExpr res;
      res.shape = {1};
      res.scalarBase = scalarBase;
      return res;
    }
    SymExpr res;
    res.shape.append(shape.begin(), shape.begin() + axis);
    res.shape.push_back(1);
    res.shape.append(shape.begin() + axis, shape.end());
    res.scalarBase = scalarBase;

    int64_t oldRank = static_cast<int64_t>(shape.size());
    if (axis < oldRank) {
      res.outerStrides.append(outerStrides.begin(), outerStrides.begin() + axis);
      res.outerStrides.emplace_back();
      res.outerStrides.append(outerStrides.begin() + axis, outerStrides.end());
      res.innerTerms = innerTerms;
      res.canonicalizeInner();
      return res;
    }

    // Appending a size-1 innermost dimension: old innermost dim becomes an
    // outer dimension, which must be affine.
    res.outerStrides = outerStrides;
    AffineScalarExpr promotedOuter;
    if (innerTerms.size() == 1 && innerTerms[0].first == 1) {
      promotedOuter.constTerm = innerTerms[0].second;
    } else if (!innerTerms.empty()) {
      return failure();
    }
    res.outerStrides.push_back(std::move(promotedOuter));
    return res;
  }

  FailureOr<SymExpr> broadcastTo(ArrayRef<int64_t> tgtShape) const {
    if (shape.size() > tgtShape.size())
      return failure();
    SymExpr cur = *this;
    while (cur.shape.size() < tgtShape.size()) {
      auto expanded = cur.expandAxis(0);
      if (failed(expanded))
        return failure();
      cur = std::move(*expanded);
    }
    for (size_t i = 0; i < tgtShape.size(); ++i) {
      if (cur.shape[i] == tgtShape[i])
        continue;
      if (cur.shape[i] != 1)
        return failure();
      cur.shape[i] = tgtShape[i];
      if (i + 1 < tgtShape.size())
        cur.outerStrides[i] = AffineScalarExpr{};
      else
        cur.innerTerms.clear();
    }
    cur.canonicalizeInner();
    return cur;
  }
};

static FailureOr<SymExpr> combineAddSub(const SymExpr &lhs, const SymExpr &rhs,
                                        int64_t rhsSign) {
  if (lhs.shape != rhs.shape)
    return failure();
  SymExpr res = lhs;
  if (!res.scalarBase.addScaled(rhs.scalarBase, rhsSign))
    return failure();
  for (size_t i = 0; i < res.outerStrides.size(); ++i) {
    if (!res.outerStrides[i].addScaled(rhs.outerStrides[i], rhsSign))
      return failure();
  }
  for (const auto &[div, coeff] : rhs.innerTerms) {
    int64_t signedCoeff = 0;
    if (!checkedMul(coeff, rhsSign, signedCoeff) ||
        !res.addInnerTerm(div, signedCoeff))
      return failure();
  }
  res.canonicalizeInner();
  return res;
}

static FailureOr<SymExpr> multiplyExprs(const SymExpr &lhs,
                                        const SymExpr &rhs) {
  if (lhs.shape != rhs.shape)
    return failure();

  auto scaleByConst = [](const SymExpr &e, int64_t k) -> FailureOr<SymExpr> {
    SymExpr res = e;
    if (!res.scalarBase.scaleBy(k))
      return failure();
    for (auto &s : res.outerStrides) {
      if (!s.scaleBy(k))
        return failure();
    }
    if (k == 0) {
      res.innerTerms.clear();
    } else {
      for (auto &p : res.innerTerms) {
        if (!checkedMul(p.second, k, p.second))
          return failure();
      }
      res.canonicalizeInner();
    }
    return res;
  };

  auto isUniformScalarExpr = [](const SymExpr &e) {
    if (!e.innerTerms.empty())
      return false;
    return llvm::all_of(e.outerStrides,
                        [](const AffineScalarExpr &s) { return s.isZero(); });
  };

  if (isUniformScalarExpr(rhs) && rhs.scalarBase.isConstant())
    return scaleByConst(lhs, rhs.scalarBase.constTerm);
  if (isUniformScalarExpr(lhs) && lhs.scalarBase.isConstant())
    return scaleByConst(rhs, lhs.scalarBase.constTerm);

  // Multiply an outer-dimension affine expression by a uniform scalar
  // polynomial expression (e.g., `(pid_n * BLOCK_N + rows_n)[:, None] * ng`).
  auto mulOuterByUniformScalar =
      [&](const SymExpr &coordExpr,
          const AffineScalarExpr &scalarFactor) -> FailureOr<SymExpr> {
    if (!coordExpr.innerTerms.empty())
      return failure();
    SymExpr res;
    res.shape = coordExpr.shape;
    if (!AffineScalarExpr::multiply(coordExpr.scalarBase, scalarFactor,
                                    res.scalarBase))
      return failure();
    res.outerStrides.resize(coordExpr.outerStrides.size());
    for (size_t i = 0; i < coordExpr.outerStrides.size(); ++i) {
      if (!AffineScalarExpr::multiply(coordExpr.outerStrides[i], scalarFactor,
                                      res.outerStrides[i]))
        return failure();
    }
    return res;
  };

  if (isUniformScalarExpr(rhs))
    return mulOuterByUniformScalar(lhs, rhs.scalarBase);
  if (isUniformScalarExpr(lhs))
    return mulOuterByUniformScalar(rhs, lhs.scalarBase);

  return failure();
}

/// Computes `floor(expr / divisor)` for `divisor > 0`. When `isEuclideanFloor`
/// is true (e.g., two's-complement `arith.shrsi`), Euclidean floor division
/// holds regardless of runtime scalar sign; otherwise `expr >= 0` is required.
static FailureOr<SymExpr> floorDivExpr(const SymExpr &expr, int64_t divisor,
                                       bool isEuclideanFloor = false) {
  if (divisor <= 0 || (!isEuclideanFloor && !expr.isKnownNonNegative()))
    return failure();
  if (divisor == 1)
    return expr;

  SymExpr quotient;
  quotient.shape = expr.shape;
  quotient.outerStrides.resize(expr.outerStrides.size());

  // Extract runtime scalar and outer-dimension multiples of `divisor`.
  for (const ScalarMonomial &m : expr.scalarBase.terms) {
    if ((m.coeff % divisor) != 0)
      return failure();
    quotient.scalarBase.terms.push_back(ScalarMonomial{m.vars, m.coeff / divisor});
  }
  for (size_t i = 0; i < expr.outerStrides.size(); ++i) {
    if (!expr.outerStrides[i].isDivisibleBy(divisor))
      return failure();
    quotient.outerStrides[i] = expr.outerStrides[i].divideExact(divisor);
  }

  int64_t qConst = floorDivSigned(expr.scalarBase.constTerm, divisor);
  int64_t remConst = expr.scalarBase.constTerm - qConst * divisor;
  quotient.scalarBase.constTerm = qConst;

  // Remaining expression `remConst + expr.innerTerms` depends only on `d_{r-1}`.
  SymExpr rem;
  rem.shape = expr.shape;
  if (expr.shape.size() > 1)
    rem.outerStrides.resize(expr.shape.size() - 1);
  rem.scalarBase.constTerm = remConst;
  rem.innerTerms = expr.innerTerms;

  auto remBounds = rem.getStaticBounds();
  if (!remBounds || remBounds->first < 0)
    return failure();

  // Extract inner floor terms whose coefficients are divisible by `divisor`
  // whenever the residual remains provably non-negative on `[0, K - 1]`.
  SmallVector<std::pair<int64_t, int64_t>, 8> residualTerms;
  SmallVector<std::pair<int64_t, int64_t>, 8> extractedTerms;
  for (const auto &[d, coeff] : rem.innerTerms) {
    if ((coeff % divisor) == 0)
      extractedTerms.emplace_back(d, coeff / divisor);
    else
      residualTerms.emplace_back(d, coeff);
  }

  SymExpr candidateRem = rem;
  candidateRem.innerTerms = residualTerms;
  auto candBounds = candidateRem.getStaticBounds();
  if (candBounds && candBounds->first >= 0) {
    for (const auto &[d, qCoeff] : extractedTerms) {
      if (!quotient.addInnerTerm(d, qCoeff))
        return failure();
    }
    rem = std::move(candidateRem);
    remBounds = candBounds;
  }

  if (remBounds->second < divisor) {
    quotient.canonicalizeInner();
    return quotient;
  }

  if (rem.scalarBase.constTerm == 0 && rem.innerTerms.size() == 1 &&
      rem.innerTerms[0].second == 1) {
    int64_t combinedDiv = 0;
    if (!checkedMul(rem.innerTerms[0].first, divisor, combinedDiv) ||
        !quotient.addInnerTerm(combinedDiv, 1))
      return failure();
    quotient.canonicalizeInner();
    return quotient;
  }

  return failure();
}

static FailureOr<SymExpr> floorModExpr(const SymExpr &expr, int64_t modulus,
                                       bool isEuclideanFloor = false) {
  if (modulus <= 0 || (!isEuclideanFloor && !expr.isKnownNonNegative()))
    return failure();
  if (modulus == 1)
    return SymExpr::constant(expr.shape, 0);
  auto q = floorDivExpr(expr, modulus, isEuclideanFloor);
  if (failed(q))
    return failure();
  return combineAddSub(expr, *q, -modulus);
}

class SymbolicLoadAnalyzer {
public:
  FailureOr<SymExpr> analyzeInt(Value v) {
    auto it = memo.find(v);
    if (it != memo.end())
      return it->second;
    if (!active.insert(v).second)
      return failure();
    FailureOr<SymExpr> res = analyzeIntImpl(v);
    active.erase(v);
    memo.try_emplace(v, res);
    return res;
  }

  FailureOr<std::pair<Value, SymExpr>> analyzePtr(Value ptrVal) {
    if (isa<PointerType>(ptrVal.getType()))
      return std::make_pair(ptrVal, SymExpr::constant({}, 0));

    auto ptrTensorTy = dyn_cast<RankedTensorType>(ptrVal.getType());
    if (!ptrTensorTy || !ptrTensorTy.hasStaticShape() ||
        !isa<PointerType>(ptrTensorTy.getElementType()))
      return failure();

    if (auto splat = ptrVal.getDefiningOp<SplatOp>()) {
      auto base = analyzePtr(splat.getSrc());
      if (failed(base))
        return failure();
      auto splatted = base->second.splatTo(ptrTensorTy.getShape());
      if (failed(splatted))
        return failure();
      return std::make_pair(base->first, std::move(*splatted));
    }

    if (auto expand = ptrVal.getDefiningOp<ExpandDimsOp>()) {
      auto src = analyzePtr(expand.getSrc());
      if (failed(src))
        return failure();
      auto expanded = src->second.expandAxis(expand.getAxis());
      if (failed(expanded) || expanded->shape != ptrTensorTy.getShape())
        return failure();
      return std::make_pair(src->first, std::move(*expanded));
    }

    if (auto bcast = ptrVal.getDefiningOp<BroadcastOp>()) {
      auto src = analyzePtr(bcast.getSrc());
      if (failed(src))
        return failure();
      auto broadcasted = src->second.broadcastTo(ptrTensorTy.getShape());
      if (failed(broadcasted))
        return failure();
      return std::make_pair(src->first, std::move(*broadcasted));
    }

    if (auto addptr = ptrVal.getDefiningOp<AddPtrOp>()) {
      auto base = analyzePtr(addptr.getPtr());
      auto off = analyzeInt(addptr.getOffset());
      if (failed(base) || failed(off))
        return failure();
      int64_t offBits = getElemBitWidth(addptr.getOffset().getType());
      if (offBits < 32 && !off->fitsSigned(offBits))
        return failure();
      auto combined = combineAddSub(base->second, *off, 1);
      if (failed(combined))
        return failure();
      return std::make_pair(base->first, std::move(*combined));
    }

    return failure();
  }

  bool isProvablyAllTrueMask(Value mask, unsigned depth = 0) {
    if (!mask || depth > 12)
      return false;
    APInt splatBool;
    if (matchPattern(mask, m_ConstantInt(&splatBool)))
      return splatBool.isAllOnes();
    if (auto cst = mask.getDefiningOp<arith::ConstantOp>()) {
      if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue())) {
        if (dense.getElementType().isInteger(1)) {
          if (dense.isSplat())
            return dense.getSplatValue<bool>();
          return llvm::all_of(dense.getValues<bool>(), [](bool b) { return b; });
        }
      }
    }
    if (auto splat = mask.getDefiningOp<SplatOp>())
      return isProvablyAllTrueMask(splat.getSrc(), depth + 1);
    if (auto expand = mask.getDefiningOp<ExpandDimsOp>())
      return isProvablyAllTrueMask(expand.getSrc(), depth + 1);
    if (auto bcast = mask.getDefiningOp<BroadcastOp>())
      return isProvablyAllTrueMask(bcast.getSrc(), depth + 1);
    if (auto andOp = mask.getDefiningOp<arith::AndIOp>())
      return isProvablyAllTrueMask(andOp.getLhs(), depth + 1) &&
             isProvablyAllTrueMask(andOp.getRhs(), depth + 1);
    if (auto orOp = mask.getDefiningOp<arith::OrIOp>())
      return isProvablyAllTrueMask(orOp.getLhs(), depth + 1) ||
             isProvablyAllTrueMask(orOp.getRhs(), depth + 1);
    if (auto cmp = mask.getDefiningOp<arith::CmpIOp>()) {
      auto lhs = analyzeInt(cmp.getLhs());
      auto rhs = analyzeInt(cmp.getRhs());
      if (failed(lhs) || failed(rhs))
        return false;
      int64_t cmpBits = getElemBitWidth(cmp.getLhs().getType());
      auto pred = cmp.getPredicate();
      bool unsignedCmp =
          pred == arith::CmpIPredicate::ult ||
          pred == arith::CmpIPredicate::ule ||
          pred == arith::CmpIPredicate::ugt ||
          pred == arith::CmpIPredicate::uge;
      if (cmpBits < 32) {
        if (unsignedCmp) {
          if (!lhs->fitsUnsigned(cmpBits) || !rhs->fitsUnsigned(cmpBits))
            return false;
        } else if (pred == arith::CmpIPredicate::eq ||
                   pred == arith::CmpIPredicate::ne) {
          bool bothSigned =
              lhs->fitsSigned(cmpBits) && rhs->fitsSigned(cmpBits);
          bool bothUnsigned =
              lhs->fitsUnsigned(cmpBits) && rhs->fitsUnsigned(cmpBits);
          if (!bothSigned && !bothUnsigned)
            return false;
        } else {
          if (!lhs->fitsSigned(cmpBits) || !rhs->fitsSigned(cmpBits))
            return false;
        }
      }
      auto diff = combineAddSub(*lhs, *rhs, -1);
      if (failed(diff))
        return false;
      auto bounds = diff->getStaticBounds();
      if (!bounds)
        return false;
      if (unsignedCmp &&
          (!lhs->isKnownNonNegative() || !rhs->isKnownNonNegative()))
        return false;
      switch (pred) {
      case arith::CmpIPredicate::slt:
      case arith::CmpIPredicate::ult:
        return bounds->second < 0;
      case arith::CmpIPredicate::sle:
      case arith::CmpIPredicate::ule:
        return bounds->second <= 0;
      case arith::CmpIPredicate::sgt:
      case arith::CmpIPredicate::ugt:
        return bounds->first > 0;
      case arith::CmpIPredicate::sge:
      case arith::CmpIPredicate::uge:
        return bounds->first >= 0;
      case arith::CmpIPredicate::eq:
        return bounds->first == 0 && bounds->second == 0;
      case arith::CmpIPredicate::ne:
        return bounds->second < 0 || bounds->first > 0;
      }
    }
    return false;
  }

private:
  llvm::DenseMap<Value, FailureOr<SymExpr>> memo;
  llvm::SmallDenseSet<Value, 16> active;

  static std::optional<int64_t> getScalarOrSplatConst(const SymExpr &e) {
    if (!e.innerTerms.empty() || !e.scalarBase.isConstant())
      return std::nullopt;
    for (const auto &s : e.outerStrides) {
      if (!s.isZero())
        return std::nullopt;
    }
    return e.scalarBase.constTerm;
  }

  static int64_t getElemBitWidth(Type ty) {
    Type elemTy = getElementTypeOrSelf(ty);
    if (elemTy.isIndex())
      return 64;
    return elemTy.getIntOrFloatBitWidth();
  }

  FailureOr<SymExpr> analyzeScalarLeaf(Value v) {
    SymExpr e;
    e.scalarBase.addMonomial({v}, 1);
    return e;
  }

  FailureOr<SymExpr> analyzeIntImpl(Value v) {
    Type ty = v.getType();
    if (!isa<RankedTensorType>(ty)) {
      if (!ty.isIntOrIndex())
        return failure();
      if (auto c = getConstantScalarInt(v))
        return SymExpr::constant({}, *c);
      if (auto addOp = v.getDefiningOp<arith::AddIOp>()) {
        auto l = analyzeInt(addOp.getLhs());
        auto r = analyzeInt(addOp.getRhs());
        if (succeeded(l) && succeeded(r))
          return combineAddSub(*l, *r, 1);
      }
      if (auto subOp = v.getDefiningOp<arith::SubIOp>()) {
        auto l = analyzeInt(subOp.getLhs());
        auto r = analyzeInt(subOp.getRhs());
        if (succeeded(l) && succeeded(r))
          return combineAddSub(*l, *r, -1);
      }
      if (auto mulOp = v.getDefiningOp<arith::MulIOp>()) {
        auto l = analyzeInt(mulOp.getLhs());
        auto r = analyzeInt(mulOp.getRhs());
        if (succeeded(l) && succeeded(r)) {
          auto prod = multiplyExprs(*l, *r);
          if (succeeded(prod))
            return prod;
        }
      }
      if (auto extOp = v.getDefiningOp<arith::ExtSIOp>()) {
        int64_t inBits = getElemBitWidth(extOp.getIn().getType());
        if (inBits >= 32) {
          auto sub = analyzeInt(extOp.getIn());
          if (succeeded(sub))
            return sub;
        }
      }
      if (auto idxCast = v.getDefiningOp<arith::IndexCastOp>()) {
        int64_t inBits = getElemBitWidth(idxCast.getIn().getType());
        int64_t outBits = getElemBitWidth(idxCast.getType());
        if (inBits >= 32 && outBits >= 32) {
          auto sub = analyzeInt(idxCast.getIn());
          if (succeeded(sub))
            return sub;
        }
      }
      return analyzeScalarLeaf(v);
    }

    auto tensorTy = cast<RankedTensorType>(ty);
    if (!tensorTy.hasStaticShape() || !tensorTy.getElementType().isIntOrIndex())
      return failure();
    ArrayRef<int64_t> shape = tensorTy.getShape();
    for (int64_t dim : shape) {
      if (dim <= 0)
        return failure();
    }

    if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
      if (auto dense = dyn_cast<DenseIntElementsAttr>(cst.getValue())) {
        if (dense.isSplat()) {
          APInt val = dense.getSplatValue<APInt>();
          if (val.getBitWidth() <= 64)
            return SymExpr::constant(shape, val.getSExtValue());
        }
      }
      return failure();
    }

    if (auto range = v.getDefiningOp<MakeRangeOp>()) {
      if (shape.size() != 1)
        return failure();
      int64_t start = range.getStart();
      int64_t end = range.getEnd();
      if (end - start != shape[0])
        return failure();
      SymExpr e;
      e.shape = {shape[0]};
      e.scalarBase.constTerm = start;
      if (shape[0] > 1)
        e.innerTerms.emplace_back(1, 1);
      return e;
    }

    if (auto splat = v.getDefiningOp<SplatOp>()) {
      auto src = analyzeInt(splat.getSrc());
      if (failed(src))
        return failure();
      return src->splatTo(shape);
    }

    if (auto expand = v.getDefiningOp<ExpandDimsOp>()) {
      auto src = analyzeInt(expand.getSrc());
      if (failed(src))
        return failure();
      auto res = src->expandAxis(expand.getAxis());
      if (failed(res) || res->shape != shape)
        return failure();
      return res;
    }

    if (auto bcast = v.getDefiningOp<BroadcastOp>()) {
      auto src = analyzeInt(bcast.getSrc());
      if (failed(src))
        return failure();
      return src->broadcastTo(shape);
    }

    int64_t bw = getElemBitWidth(ty);

    if (auto addOp = v.getDefiningOp<arith::AddIOp>()) {
      auto l = analyzeInt(addOp.getLhs());
      auto r = analyzeInt(addOp.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto res = combineAddSub(*l, *r, 1);
      if (failed(res) || !res->fitsLosslessNarrow(bw))
        return failure();
      return res;
    }

    if (auto subOp = v.getDefiningOp<arith::SubIOp>()) {
      auto l = analyzeInt(subOp.getLhs());
      auto r = analyzeInt(subOp.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto res = combineAddSub(*l, *r, -1);
      if (failed(res) || !res->fitsLosslessNarrow(bw))
        return failure();
      return res;
    }

    if (auto mulOp = v.getDefiningOp<arith::MulIOp>()) {
      auto l = analyzeInt(mulOp.getLhs());
      auto r = analyzeInt(mulOp.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto res = multiplyExprs(*l, *r);
      if (failed(res) || !res->fitsLosslessNarrow(bw))
        return failure();
      return res;
    }

    if (auto shlOp = v.getDefiningOp<arith::ShLIOp>()) {
      auto l = analyzeInt(shlOp.getLhs());
      auto r = analyzeInt(shlOp.getRhs());
      if (failed(l) || failed(r))
        return failure();
      auto shift = getScalarOrSplatConst(*r);
      if (!shift || *shift < 0 || *shift >= std::min<int64_t>(bw, 62))
        return failure();
      int64_t factor = int64_t{1} << *shift;
      auto res = multiplyExprs(*l, SymExpr::constant(shape, factor));
      if (failed(res) || !res->fitsLosslessNarrow(bw))
        return failure();
      return res;
    }

    if (isa_and_nonnull<arith::ShRSIOp, arith::ShRUIOp>(v.getDefiningOp())) {
      Operation *op = v.getDefiningOp();
      auto l = analyzeInt(op->getOperand(0));
      auto r = analyzeInt(op->getOperand(1));
      if (failed(l) || failed(r))
        return failure();
      auto shift = getScalarOrSplatConst(*r);
      if (!shift || *shift < 0 || *shift >= std::min<int64_t>(bw, 62))
        return failure();
      bool isSigned = isa<arith::ShRSIOp>(op);
      if (bw < 32) {
        if (isSigned && !l->fitsSigned(bw))
          return failure();
        if (!isSigned && !l->fitsUnsigned(bw))
          return failure();
      }
      bool isSignedFloor = isSigned && bw >= 32;
      return floorDivExpr(*l, int64_t{1} << *shift, isSignedFloor);
    }

    if (isa_and_nonnull<arith::DivSIOp, arith::DivUIOp>(v.getDefiningOp())) {
      Operation *op = v.getDefiningOp();
      auto l = analyzeInt(op->getOperand(0));
      auto r = analyzeInt(op->getOperand(1));
      if (failed(l) || failed(r))
        return failure();
      auto rawDiv = getScalarOrSplatConst(*r);
      if (!rawDiv)
        return failure();
      bool isSigned = isa<arith::DivSIOp>(op);
      int64_t div = *rawDiv;
      if (bw < 32) {
        if (isSigned && !l->fitsSigned(bw))
          return failure();
        if (!isSigned) {
          if (!l->fitsUnsigned(bw))
            return failure();
          div = static_cast<int64_t>(static_cast<uint64_t>(div) &
                                     ((uint64_t{1} << bw) - 1));
        }
      }
      if (div <= 0)
        return failure();
      return floorDivExpr(*l, div, /*isEuclideanFloor=*/false);
    }

    if (isa_and_nonnull<arith::RemSIOp, arith::RemUIOp>(v.getDefiningOp())) {
      Operation *op = v.getDefiningOp();
      auto l = analyzeInt(op->getOperand(0));
      auto r = analyzeInt(op->getOperand(1));
      if (failed(l) || failed(r))
        return failure();
      auto rawMod = getScalarOrSplatConst(*r);
      if (!rawMod)
        return failure();
      bool isSigned = isa<arith::RemSIOp>(op);
      int64_t mod = *rawMod;
      if (bw < 32) {
        if (isSigned && !l->fitsSigned(bw))
          return failure();
        if (!isSigned) {
          if (!l->fitsUnsigned(bw))
            return failure();
          mod = static_cast<int64_t>(static_cast<uint64_t>(mod) &
                                     ((uint64_t{1} << bw) - 1));
        }
      }
      if (mod <= 0)
        return failure();
      return floorModExpr(*l, mod, /*isEuclideanFloor=*/false);
    }

    if (auto andOp = v.getDefiningOp<arith::AndIOp>()) {
      auto l = analyzeInt(andOp.getLhs());
      auto r = analyzeInt(andOp.getRhs());
      if (failed(l) || failed(r))
        return failure();
      bool isWide = bw >= 32;
      auto tryMask = [&](const SymExpr &valExpr,
                         int64_t mask) -> FailureOr<SymExpr> {
        uint64_t uMask = (bw < 64)
                             ? (static_cast<uint64_t>(mask) &
                                ((uint64_t{1} << bw) - 1))
                             : static_cast<uint64_t>(mask);
        if (uMask == 0)
          return SymExpr::constant(shape, 0);
        uint64_t allOnes =
            (bw < 64) ? ((uint64_t{1} << bw) - 1) : ~uint64_t{0};
        if (uMask == allOnes)
          return valExpr;
        if (uMask < (uint64_t{1} << 62) && ((uMask & (uMask + 1)) == 0)) {
          // For `bw < 32` with `uMask == 2^m - 1 < 2^bw`, `(H mod 2^bw) mod 2^m`
          // equals Euclidean `E mod 2^m` for any `E == H (mod 2^bw)`.
          bool euclid = isWide || valExpr.fitsLosslessNarrow(bw);
          return floorModExpr(valExpr, static_cast<int64_t>(uMask + 1), euclid);
        }
        return failure();
      };
      if (auto mr = getScalarOrSplatConst(*r))
        return tryMask(*l, *mr);
      if (auto ml = getScalarOrSplatConst(*l))
        return tryMask(*r, *ml);
      return failure();
    }

    if (auto truncOp = v.getDefiningOp<arith::TruncIOp>()) {
      auto sub = analyzeInt(truncOp.getIn());
      if (failed(sub))
        return failure();
      if (bw <= 0 || bw >= 63)
        return failure();
      if (!sub->fitsSigned(bw) && !sub->fitsUnsigned(bw))
        return failure();
      return sub;
    }

    if (auto extS = v.getDefiningOp<arith::ExtSIOp>()) {
      auto sub = analyzeInt(extS.getIn());
      if (failed(sub))
        return failure();
      int64_t inBits = getElemBitWidth(extS.getIn().getType());
      if (inBits < 32 && !sub->fitsSigned(inBits))
        return failure();
      return sub;
    }

    if (auto extU = v.getDefiningOp<arith::ExtUIOp>()) {
      auto sub = analyzeInt(extU.getIn());
      if (failed(sub))
        return failure();
      int64_t inBits = getElemBitWidth(extU.getIn().getType());
      if (inBits < 32) {
        if (auto c = getScalarOrSplatConst(*sub);
            c && *c < 0 && *c >= -(int64_t{1} << (inBits - 1))) {
          return SymExpr::constant(
              shape, (*c) & ((int64_t{1} << inBits) - 1));
        }
        if (!sub->fitsUnsigned(inBits))
          return failure();
        return sub;
      }
      if (!sub->isKnownNonNegative())
        return failure();
      return sub;
    }

    if (isa_and_nonnull<arith::IndexCastOp, arith::IndexCastUIOp>(
            v.getDefiningOp())) {
      Operation *castOp = v.getDefiningOp();
      auto sub = analyzeInt(castOp->getOperand(0));
      if (failed(sub))
        return failure();
      bool isUI = isa<arith::IndexCastUIOp>(castOp);
      if (isUI && !sub->isKnownNonNegative())
        return failure();
      int64_t minBits = std::min(getElemBitWidth(castOp->getOperand(0).getType()),
                                 bw);
      if (minBits < 32) {
        if (isUI ? !sub->fitsUnsigned(minBits) : !sub->fitsSigned(minBits))
          return failure();
      }
      return sub;
    }

    return failure();
  }
};

struct SubElementLayout {
  int64_t groups = 1;
  int64_t packWidth = 1;      // P
  int64_t stridedRepeat = 1;  // R_strided
  int64_t consecRepeat = 1;   // R_consec
  int64_t innerStride = 1;    // physical element stride along compact inner dim

  int64_t physicalCols() const { return groups * packWidth; }
};

static std::optional<SubElementLayout>
deduceSubElementLayout(const SymExpr &expr) {
  if (expr.shape.empty())
    return std::nullopt;
  int64_t K = expr.shape.back();
  if (K < 2)
    return std::nullopt;

  if (expr.innerTerms.empty()) {
    SubElementLayout layout;
    layout.groups = 1;
    layout.packWidth = 1;
    layout.stridedRepeat = 1;
    layout.consecRepeat = K;
    layout.innerStride = 1;
    return layout;
  }

  int64_t rConsec = 0;
  for (const auto &[div, _] : expr.innerTerms)
    rConsec = std::gcd(rConsec, div);
  if (rConsec <= 0 || (K % rConsec) != 0)
    return std::nullopt;

  int64_t Ku = K / rConsec;
  SmallVector<std::pair<int64_t, int64_t>, 8> uTerms;
  for (const auto &[div, coeff] : expr.innerTerms)
    uTerms.emplace_back(div / rConsec, coeff);

  // Family 1: Pure consecutive sub-element replication (`S_inner * floor(c / R)`).
  if (uTerms.size() == 1 && uTerms[0].first == 1 && uTerms[0].second > 0) {
    if (rConsec < 2)
      return std::nullopt;
    SubElementLayout layout;
    layout.groups = Ku;
    layout.packWidth = 1;
    layout.stridedRepeat = 1;
    layout.consecRepeat = rConsec;
    layout.innerStride = uTerms[0].second;
    return layout;
  }

  // Family 2: Strided periodic replication (optionally composed with rConsec):
  //   `S * u - (P * S) * floor(u / P) [+ (P * S) * floor(u / Gu)]`
  if (uTerms.size() == 2 || uTerms.size() == 3) {
    int64_t sInner = uTerms[0].second;
    if (uTerms[0].first != 1 || sInner <= 0)
      return std::nullopt;
    int64_t P = uTerms[1].first;
    int64_t expectedNeg = 0;
    if (P < 2 || !checkedMul(P, sInner, expectedNeg) ||
        uTerms[1].second != -expectedNeg)
      return std::nullopt;

    int64_t Gu = Ku;
    if (uTerms.size() == 3) {
      Gu = uTerms[2].first;
      if (uTerms[2].second != expectedNeg)
        return std::nullopt;
    }
    if (Gu <= P || (Gu % P) != 0 || (Ku % Gu) != 0)
      return std::nullopt;

    SubElementLayout layout;
    layout.groups = Ku / Gu;
    layout.packWidth = P;
    layout.stridedRepeat = Gu / P;
    layout.consecRepeat = rConsec;
    layout.innerStride = sInner;
    return layout;
  }

  return std::nullopt;
}

static bool canProjectInnerInvariant(Value v, ArrayRef<int64_t> fullShape,
                                     unsigned depth = 0) {
  if (!v || depth > 8)
    return false;
  auto ty = dyn_cast<RankedTensorType>(v.getType());
  if (!ty || !ty.hasStaticShape() || ty.getShape() != fullShape)
    return false;

  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    if (auto dense = dyn_cast<DenseElementsAttr>(cst.getValue()))
      return dense.isSplat();
  }
  if (v.getDefiningOp<SplatOp>())
    return true;
  if (auto expand = v.getDefiningOp<ExpandDimsOp>())
    return expand.getAxis() == static_cast<int64_t>(fullShape.size()) - 1;
  if (auto bcast = v.getDefiningOp<BroadcastOp>()) {
    auto srcTy = dyn_cast<RankedTensorType>(bcast.getSrc().getType());
    return srcTy && srcTy.hasStaticShape() &&
           srcTy.getRank() == static_cast<int64_t>(fullShape.size()) &&
           srcTy.getShape().back() == 1;
  }
  if (isa_and_nonnull<arith::AndIOp, arith::OrIOp>(v.getDefiningOp())) {
    Operation *bin = v.getDefiningOp();
    return canProjectInnerInvariant(bin->getOperand(0), fullShape, depth + 1) &&
           canProjectInnerInvariant(bin->getOperand(1), fullShape, depth + 1);
  }
  return false;
}

static Value projectInnerInvariant(Value v, ArrayRef<int64_t> fullShape,
                                   ArrayRef<int64_t> compactShape,
                                   Location loc, PatternRewriter &rewriter) {
  auto ty = cast<RankedTensorType>(v.getType());
  auto compactTy = RankedTensorType::get(compactShape, ty.getElementType());

  if (auto cst = v.getDefiningOp<arith::ConstantOp>()) {
    auto dense = cast<DenseElementsAttr>(cst.getValue());
    Attribute splatElem = dense.getSplatValue<Attribute>();
    return rewriter.create<arith::ConstantOp>(
        loc, DenseElementsAttr::get(compactTy, splatElem));
  }
  if (auto splat = v.getDefiningOp<SplatOp>())
    return rewriter.create<SplatOp>(loc, compactTy, splat.getSrc());
  if (auto expand = v.getDefiningOp<ExpandDimsOp>()) {
    if (compactShape.back() == 1)
      return expand.getResult();
    return rewriter.create<BroadcastOp>(loc, compactTy, expand.getResult());
  }
  if (auto bcast = v.getDefiningOp<BroadcastOp>()) {
    auto srcTy = cast<RankedTensorType>(bcast.getSrc().getType());
    if (srcTy.getShape() == compactShape)
      return bcast.getSrc();
    return rewriter.create<BroadcastOp>(loc, compactTy, bcast.getSrc());
  }
  if (auto andOp = v.getDefiningOp<arith::AndIOp>()) {
    Value l = projectInnerInvariant(andOp.getLhs(), fullShape, compactShape,
                                    loc, rewriter);
    Value r = projectInnerInvariant(andOp.getRhs(), fullShape, compactShape,
                                    loc, rewriter);
    return rewriter.create<arith::AndIOp>(loc, l, r);
  }
  auto orOp = cast<arith::OrIOp>(v.getDefiningOp());
  Value l = projectInnerInvariant(orOp.getLhs(), fullShape, compactShape, loc,
                                  rewriter);
  Value r = projectInnerInvariant(orOp.getRhs(), fullShape, compactShape, loc,
                                  rewriter);
  return rewriter.create<arith::OrIOp>(loc, l, r);
}

static Value castScalarToIntType(Location loc, Value val, IntegerType intTy,
                                 PatternRewriter &rewriter) {
  if (val.getType() == intTy)
    return val;
  if (val.getType().isIndex())
    return rewriter.create<arith::IndexCastOp>(loc, intTy, val);
  unsigned srcW = val.getType().getIntOrFloatBitWidth();
  unsigned dstW = intTy.getWidth();
  if (srcW < dstW)
    return rewriter.create<arith::ExtSIOp>(loc, intTy, val);
  return rewriter.create<arith::TruncIOp>(loc, intTy, val);
}

static Value materializeAffineScalar(Location loc, const AffineScalarExpr &expr,
                                     IntegerType intTy,
                                     PatternRewriter &rewriter) {
  Value acc = nullptr;
  for (const ScalarMonomial &m : expr.terms) {
    if (m.coeff == 0)
      continue;
    Value term = nullptr;
    for (Value scalarVal : m.vars) {
      Value casted = castScalarToIntType(loc, scalarVal, intTy, rewriter);
      term = term ? rewriter.create<arith::MulIOp>(loc, term, casted).getResult()
                  : casted;
    }
    if (!term) {
      term =
          rewriter.create<arith::ConstantIntOp>(loc, m.coeff, intTy.getWidth());
    } else if (m.coeff != 1) {
      Value mulC =
          rewriter.create<arith::ConstantIntOp>(loc, m.coeff, intTy.getWidth());
      term = rewriter.create<arith::MulIOp>(loc, term, mulC);
    }
    acc = acc ? rewriter.create<arith::AddIOp>(loc, acc, term).getResult()
              : term;
  }
  if (!acc || expr.constTerm != 0) {
    Value cst = rewriter.create<arith::ConstantIntOp>(loc, expr.constTerm,
                                                      intTy.getWidth());
    acc =
        acc ? rewriter.create<arith::AddIOp>(loc, acc, cst).getResult() : cst;
  }
  return acc;
}

static bool isFlatContiguousUnmasked(const SymExpr &expr,
                                     const SubElementLayout &layout,
                                     bool hasRuntimeMask) {
  if (hasRuntimeMask || layout.innerStride != 1 || !expr.scalarBase.isConstant())
    return false;
  if (expr.shape.size() < 2)
    return false;
  int64_t expectedStride = layout.physicalCols();
  for (int64_t d = static_cast<int64_t>(expr.shape.size()) - 2; d >= 0; --d) {
    if (!expr.outerStrides[d].isConstant())
      return false;
    if (expr.shape[d] > 1 && expr.outerStrides[d].constTerm != expectedStride)
      return false;
    if (!checkedMul(expectedStride, expr.shape[d], expectedStride))
      return false;
  }
  return expectedStride <= std::numeric_limits<int32_t>::max();
}

static Value createCompactLoad1D(Location loc, Value basePtr, int64_t elements,
                                 int64_t baseOffset, LoadOp origLoad,
                                 PatternRewriter &rewriter) {
  auto ptrElemTy = cast<PointerType>(basePtr.getType());
  auto ptrTensorTy = RankedTensorType::get({elements}, ptrElemTy);
  auto idxTensorTy = RankedTensorType::get({elements}, rewriter.getI32Type());

  Value offsets =
      rewriter.create<MakeRangeOp>(loc, idxTensorTy, 0, elements).getResult();
  if (baseOffset != 0) {
    Value cst = rewriter.create<arith::ConstantIntOp>(loc, baseOffset, 32);
    Value splatCst = rewriter.create<SplatOp>(loc, idxTensorTy, cst);
    offsets = rewriter.create<arith::AddIOp>(loc, offsets, splatCst);
  }
  Value ptrSplat = rewriter.create<SplatOp>(loc, ptrTensorTy, basePtr);
  Value ptr = rewriter.create<AddPtrOp>(loc, ptrTensorTy, ptrSplat, offsets);
  return rewriter
      .create<LoadOp>(loc, ptr, /*mask=*/nullptr, /*other=*/nullptr,
                      origLoad.getCache(), origLoad.getEvict(),
                      origLoad.getIsVolatile())
      .getResult();
}

static Value createCompactLoadND(Location loc, Value basePtr,
                                 ArrayRef<int64_t> compactShape,
                                 const SymExpr &expr,
                                 const SubElementLayout &layout,
                                 Value compactMask, Value compactOther,
                                 LoadOp origLoad, PatternRewriter &rewriter) {
  bool needI64 = expr.scalarBase.requiresI64();
  for (const auto &s : expr.outerStrides)
    needI64 |= s.requiresI64();
  IntegerType intTy = needI64 ? rewriter.getI64Type() : rewriter.getI32Type();
  unsigned intW = intTy.getWidth();

  // Fold dynamic scalar base terms (`k_idx * 64`, etc.) into scalar `basePtr`
  // while keeping constant byte/element offsets on the final tensor `AddPtrOp`
  // so downstream complementary-deinterleave matchers see static lane offsets.
  if (!expr.scalarBase.terms.empty()) {
    AffineScalarExpr dynOnly;
    dynOnly.terms = expr.scalarBase.terms;
    Value dynScalarOff = materializeAffineScalar(loc, dynOnly, intTy, rewriter);
    basePtr =
        rewriter.create<AddPtrOp>(loc, basePtr.getType(), basePtr, dynScalarOff);
  }

  auto fullIdxTy = RankedTensorType::get(compactShape, intTy);
  size_t rank = compactShape.size();
  Value totalOffsets = nullptr;

  for (size_t axis = 0; axis < rank; ++axis) {
    int64_t dimExtent = compactShape[axis];
    if (dimExtent <= 1)
      continue;

    bool isInner = (axis + 1 == rank);
    if (!isInner && expr.outerStrides[axis].isZero())
      continue;

    auto range1DTyI32 =
        RankedTensorType::get({dimExtent}, rewriter.getI32Type());
    Value axisRange =
        rewriter.create<MakeRangeOp>(loc, range1DTyI32, 0, dimExtent)
            .getResult();
    if (intW == 64) {
      axisRange = rewriter.create<arith::ExtSIOp>(
          loc, RankedTensorType::get({dimExtent}, intTy), axisRange);
    }

    Value oriented = axisRange;
    if (rank > 1) {
      for (size_t d = 0; d < axis; ++d)
        oriented = rewriter.create<ExpandDimsOp>(loc, oriented, 0);
      for (size_t d = axis + 1; d < rank; ++d) {
        int64_t curRank =
            cast<RankedTensorType>(oriented.getType()).getRank();
        oriented = rewriter.create<ExpandDimsOp>(loc, oriented, curRank);
      }
    }
    auto orientedTy = cast<RankedTensorType>(oriented.getType());

    if (isInner) {
      if (layout.innerStride != 1) {
        Value strideScalar = rewriter.create<arith::ConstantIntOp>(
            loc, layout.innerStride, intW);
        Value strideSplat =
            rewriter.create<SplatOp>(loc, orientedTy, strideScalar);
        oriented = rewriter.create<arith::MulIOp>(loc, oriented, strideSplat);
      }
    } else {
      Value strideScalar = materializeAffineScalar(
          loc, expr.outerStrides[axis], intTy, rewriter);
      Value strideSplat =
          rewriter.create<SplatOp>(loc, orientedTy, strideScalar);
      oriented = rewriter.create<arith::MulIOp>(loc, oriented, strideSplat);
    }

    Value bcastAxis =
        (orientedTy == fullIdxTy)
            ? oriented
            : rewriter.create<BroadcastOp>(loc, fullIdxTy, oriented).getResult();
    totalOffsets =
        totalOffsets
            ? rewriter.create<arith::AddIOp>(loc, totalOffsets, bcastAxis)
                  .getResult()
            : bcastAxis;
  }

  if (!totalOffsets) {
    Value zero = rewriter.create<arith::ConstantIntOp>(loc, 0, intW);
    totalOffsets = rewriter.create<SplatOp>(loc, fullIdxTy, zero);
  }

  auto ptrElemTy = cast<PointerType>(basePtr.getType());
  auto ptrTensorTy = RankedTensorType::get(compactShape, ptrElemTy);
  Value ptrSplat = rewriter.create<SplatOp>(loc, ptrTensorTy, basePtr);
  Value ptr =
      rewriter.create<AddPtrOp>(loc, ptrTensorTy, ptrSplat, totalOffsets);

  if (expr.scalarBase.constTerm != 0) {
    Value cst = rewriter.create<arith::ConstantIntOp>(
        loc, expr.scalarBase.constTerm, intW);
    Value cstSplat = rewriter.create<SplatOp>(loc, fullIdxTy, cst);
    ptr = rewriter.create<AddPtrOp>(loc, ptrTensorTy, ptr, cstSplat);
  }

  return rewriter
      .create<LoadOp>(loc, ptr, compactMask, compactOther, origLoad.getCache(),
                      origLoad.getEvict(), origLoad.getIsVolatile())
      .getResult();
}

} // namespace

LogicalResult SubElementLoadCoalesce::matchAndRewrite(
    LoadOp op, PatternRewriter &rewriter) const {
  if (!op.getBoundaryCheck().empty())
    return failure();
  auto resultType = dyn_cast<RankedTensorType>(op.getResult().getType());
  if (!resultType || !resultType.hasStaticShape() || resultType.getRank() < 1)
    return failure();
  if (op.getOther() && !op.getMask())
    return failure();

  ArrayRef<int64_t> fullShape = resultType.getShape();
  if (fullShape.back() < 2)
    return failure();

  SymbolicLoadAnalyzer analyzer;
  bool maskAllTrue = !op.getMask() || analyzer.isProvablyAllTrueMask(op.getMask());
  if (!maskAllTrue) {
    if (!canProjectInnerInvariant(op.getMask(), fullShape))
      return failure();
    if (op.getOther() && !canProjectInnerInvariant(op.getOther(), fullShape))
      return failure();
  }

  auto analyzedPtr = analyzer.analyzePtr(op.getPtr());
  if (failed(analyzedPtr) || analyzedPtr->second.shape != fullShape)
    return failure();

  Value basePtr = analyzedPtr->first;
  const SymExpr &offsetExpr = analyzedPtr->second;
  auto layout = deduceSubElementLayout(offsetExpr);
  if (!layout)
    return failure();

  Location loc = op.getLoc();
  SmallVector<int64_t, 4> compactShape(fullShape.drop_back(1));
  compactShape.push_back(layout->physicalCols());

  Value compact = nullptr;
  if (isFlatContiguousUnmasked(offsetExpr, *layout, !maskAllTrue)) {
    int64_t flatElements = 1;
    for (int64_t d : compactShape)
      flatElements *= d;
    compact = createCompactLoad1D(loc, basePtr, flatElements,
                                  offsetExpr.scalarBase.constTerm, op, rewriter);
  } else {
    Value compactMask = nullptr;
    Value compactOther = nullptr;
    if (!maskAllTrue) {
      compactMask = projectInnerInvariant(op.getMask(), fullShape, compactShape,
                                          loc, rewriter);
      if (op.getOther())
        compactOther = projectInnerInvariant(op.getOther(), fullShape,
                                             compactShape, loc, rewriter);
    }
    compact = createCompactLoadND(loc, basePtr, compactShape, offsetExpr,
                                  *layout, compactMask, compactOther, op,
                                  rewriter);
  }

  Type elemTy = resultType.getElementType();
  Value cur = compact;
  int64_t rank = resultType.getRank();

  if (layout->stridedRepeat > 1) {
    SmallVector<int64_t, 5> groupedShape(fullShape.drop_back(1));
    groupedShape.push_back(layout->groups);
    groupedShape.push_back(layout->packWidth);
    auto groupedTy = RankedTensorType::get(groupedShape, elemTy);
    if (cur.getType() != groupedTy)
      cur = rewriter.create<ReshapeOp>(loc, groupedTy, cur);

    int64_t repAxis = rank;
    Value expanded = rewriter.create<ExpandDimsOp>(loc, cur, repAxis);
    SmallVector<int64_t, 6> bcastShape(groupedShape.begin(),
                                       groupedShape.end());
    bcastShape.insert(bcastShape.begin() + repAxis, layout->stridedRepeat);
    Value broadcasted = rewriter.create<BroadcastOp>(
        loc, RankedTensorType::get(bcastShape, elemTy), expanded);

    SmallVector<int64_t, 4> postStridedShape(fullShape.drop_back(1));
    postStridedShape.push_back(layout->groups * layout->stridedRepeat *
                               layout->packWidth);
    cur = rewriter.create<ReshapeOp>(
        loc, RankedTensorType::get(postStridedShape, elemTy), broadcasted);
  }

  if (layout->consecRepeat > 1) {
    auto curTy = cast<RankedTensorType>(cur.getType());
    if (curTy.getRank() != rank) {
      cur = rewriter.create<ReshapeOp>(
          loc, RankedTensorType::get(compactShape, elemTy), cur);
      curTy = cast<RankedTensorType>(cur.getType());
    }
    Value expanded = rewriter.create<ExpandDimsOp>(loc, cur, rank);
    SmallVector<int64_t, 5> bcastShape(curTy.getShape().begin(),
                                       curTy.getShape().end());
    bcastShape.push_back(layout->consecRepeat);
    Value broadcasted = rewriter.create<BroadcastOp>(
        loc, RankedTensorType::get(bcastShape, elemTy), expanded);
    cur = rewriter.create<ReshapeOp>(loc, resultType, broadcasted);
  }

  rewriter.replaceOp(op, cur);
  return success();
}

} // namespace mlir::triton
