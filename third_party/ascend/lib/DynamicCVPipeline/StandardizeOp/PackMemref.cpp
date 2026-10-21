#include <memory>
#include <string>

#include "llvm/ADT/DenseSet.h"
#include "llvm/ADT/STLExtras.h"
#include "llvm/ADT/SetVector.h"
#include "llvm/ADT/SmallPtrSet.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Debug.h"
#include "llvm/Support/LogicalResult.h"

#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/IRMapping.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Support/WalkResult.h"

#include "ascend/include/DynamicCVPipeline/StandardizeOp/Passes.h"

#include "DynamicCVPipeline/Common/Analysis.h"
#include "DynamicCVPipeline/Common/ScalarClosure.h"
#include "DynamicCVPipeline/Common/ScopeOpUtils.h"
#include "DynamicCVPipeline/Common/Utils.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/IR/HIVMImpl.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

using namespace mlir;
using namespace CVPipeline;

static constexpr const char *DEBUG_TYPE = "pack-memref";
#define DBGS() (llvm::dbgs() << '[' << DEBUG_TYPE << "] ")
#define LOG_DEBUG(...)                                                         \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__)

// `ssbuffer.core_type` written onto packed scope units. Mirrors the literal
// read by OpClassifierPass.
static constexpr llvm::StringLiteral kCoreTypeAttr = "ssbuffer.core_type";

// Resolve the root memref behind `val`, following view-like ops
// (memref.subview / memref.reinterpret_cast / ...) and
// bufferization.to_tensor.
static Value getUnderlyingMemref(Value val) {
  while (val) {
    if (auto toTensor = val.getDefiningOp<bufferization::ToTensorOp>()) {
      val = toTensor.getBuffer();
      continue;
    }
    if (auto view = val.getDefiningOp<ViewLikeOpInterface>()) {
      val = view.getViewSource();
      continue;
    }
    break;
  }
  return val;
}

// Top-level ancestor of `op` that lives directly in `block`, or nullptr.
static Operation *topLevelOpInBlock(Operation *op, Block *block) {
  Operation *cur = op;
  while (cur && cur->getBlock() != block)
    cur = cur->getParentOp();
  return cur;
}

static bool usesMemref(Operation *op, Value base) {
  bool found = false;
  op->walk([&](Operation *nested) {
    for (Value operand : nested->getOperands()) {
      if (getUnderlyingMemref(operand) == base) {
        found = true;
        return WalkResult::interrupt();
      }
    }
    return WalkResult::advance();
  });
  return found;
}

// The block directly containing `op` holds nothing but `op` (terminators
// excepted).
static bool isTrivialSingleOpRegion(Operation *op) {
  Block *block = op->getBlock();
  if (!block)
    return false;
  for (Operation &nested : *block) {
    if (&nested == op || nested.hasTrait<OpTrait::IsTerminator>())
      continue;
    return false;
  }
  return true;
}

namespace {

class PackMemrefPass
    : public PassWrapper<PackMemrefPass, OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(PackMemrefPass);

  PackMemrefPass() = default;
  void runOnOperation() override;

  void getDependentDialects(mlir::DialectRegistry &registry) const override {
    registry
        .insert<scope::ScopeDialect, hivm::HIVMDialect,
                bufferization::BufferizationDialect, memref::MemRefDialect>();
  }

  [[nodiscard]] llvm::StringRef getArgument() const final {
    return "ssbuf-standardize-op-pack-memref";
  }
};

} // namespace

// Group hivm.hir.custom + its tightly-coupled buffers into a scope unit.
static void packCustomOps(ModuleOp module);
// Pack `memref.alloc -> <ops on it> -> bufferization.to_tensor` load units.
static void packLoadUnits(ModuleOp module);
static llvm::LogicalResult packLoadUnit(memref::AllocOp allocOp);

void PackMemrefPass::runOnOperation() {
  auto moduleOp = getOperation();
  packCustomOps(moduleOp);
  packLoadUnits(moduleOp);
}

// ============================================================================
// CustomOp grouping (migrated from OpClassifierPass::groupCustomOps).
// Each group is packed as a single scope::ScopeOp carrying the analyzed core
// type, so OpClassifierPass can honor it without re-deriving it.
// ============================================================================
void packCustomOps(ModuleOp module) {
  SmallVector<hivm::CustomOp> customOps;
  module.walk(
      [&customOps](hivm::CustomOp customOp) { customOps.push_back(customOp); });

  for (auto customOp : customOps) {
    auto anaRes = CustomOpAnalysis::get(customOp);
    if (llvm::failed(anaRes)) {
      LOG_DEBUG("Warning: failed to analyze " << customOp << "\n");
      continue;
    }
    auto &ana = anaRes.value();
    LOG_DEBUG("Packing: " << ana.customOp << "\n");

    auto scopeOp = packScopeOp(ana.relaventOps);
    if (!scopeOp) {
      LOG_DEBUG("Failed to pack ScopeOp: "
                << customOp << "; fallback to default behaviour.\n");
      continue;
    }
    scopeOp->setAttr(
        kCoreTypeAttr,
        StringAttr::get(scopeOp->getContext(), coreTypeToString(ana.coreType)));
  }
}

void packLoadUnits(ModuleOp module) {
  SmallVector<memref::AllocOp> allocs;
  module.walk([&allocs](memref::AllocOp alloc) { allocs.push_back(alloc); });
  for (auto allocOp : allocs) {
    // Skip buffers already swallowed by an earlier packing (e.g. a CustomOp
    // group).
    if (allocOp->getParentOfType<scope::ScopeOp>())
      continue;
    (void)packLoadUnit(allocOp);
  }
}

// ============================================================================
// Load unit packing.
//
// Pattern:
//   %a  = memref.alloc()
//        <ops operating on %a>       // ExtractedLoadOrStore loop, or a
//                                    // trivial single-op region
//   %b  = bufferization.to_tensor %a     (may go through a subview /
//                                        reinterpret_cast of %a)
//
// The unit is packed as a scope::ScopeOp tagged with `ssbuffer.load`. When the
// unit is driven by an ExtractedLoadOrStore loop the scope is stamped VECTOR
// right away (ExtractedLoadOrStore never runs on CUBE).
// ============================================================================
llvm::LogicalResult packLoadUnit(memref::AllocOp allocOp) {
  Value base = allocOp.getMemref();
  Block *block = allocOp->getBlock();
  Operation *parentOp = block ? block->getParentOp() : nullptr;
  if (!parentOp)
    return llvm::failure();

  // --- Step 3: locate the unique bufferization.to_tensor rooted at `base`. ---
  SmallVector<bufferization::ToTensorOp> candidates;
  parentOp->walk([&](bufferization::ToTensorOp toTensorOp) {
    if (toTensorOp->getBlock() == block &&
        getUnderlyingMemref(toTensorOp.getBuffer()) == base)
      candidates.push_back(toTensorOp);
  });
  if (candidates.size() != 1)
    return llvm::failure();
  bufferization::ToTensorOp toTensorOp = candidates.front();

  // --- Step 2: collect the ops operating on `base` and validate them. ---
  llvm::SetVector<Operation *> unit;
  unit.insert(allocOp);

  bool isExtractedLoadStore = false;
  bool reachedToTensor = false;
  for (Operation *op = allocOp->getNextNode(); op; op = op->getNextNode()) {
    if (op == toTensorOp) {
      reachedToTensor = true;
      break;
    }
    if (!usesMemref(op, base))
      continue;

    // Normalize a nested load op to its top-level ancestor in `block`, so the
    // packed unit stays within one block.
    Operation *top = topLevelOpInBlock(op, block);
    if (!top)
      return llvm::failure();

    if (isa<scf::ForOp>(top) && top->hasAttr(hivm::ExtractLoadStoreAttr)) {
      isExtractedLoadStore = true;
    } else if (top != op && !isTrivialSingleOpRegion(op)) {
      // Neither an ExtractedLoadOrStore loop nor a trivial single-op region:
      // abort the whole unit.
      return llvm::failure();
    }

    unit.insert(top);
  }
  if (!reachedToTensor)
    return llvm::failure();
  unit.insert(toTensorOp);

  // --- Step 4: honour ScalarClosure for subviews / reinterpret_cast / scalars.
  SmallVector<Operation *> unitOps(unit.begin(), unit.end());
  llvm::sort(unitOps,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  ScalarClosure closure(allocOp->getBlock(), unitOps, false);
  closure.collect();

  llvm::DenseSet<Operation *> inUnit(unit.begin(), unit.end());
  llvm::DenseSet<Operation *> inClosure;
  for (Operation *scalarOp : closure.scalarOps) {
    inClosure.insert(scalarOp);
  }

  SmallVector<Operation *> scalars(closure.scalarOps.getArrayRef());
  llvm::sort(scalars,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  OpBuilder builder(allocOp);
  builder.setInsertionPoint(allocOp);
  IRMapping mapper;

  SmallVector<Operation *> packList = unitOps;
  for (Operation *scalarOp : scalars) {
    bool hasExternalUse = false;
    for (Value result : scalarOp->getResults()) {
      for (OpOperand &use : result.getUses()) {
        Operation *user = use.getOwner();
        if (!inUnit.contains(user) && !inClosure.contains(user)) {
          hasExternalUse = true;
          break;
        }
      }
      if (hasExternalUse)
        break;
    }

    if (!hasExternalUse) {
      // No external consumer: move it straight into the scope.
      packList.push_back(scalarOp);
      continue;
    }

    // External consumers exist: clone the scalar for them (before the unit)
    // and keep the original inside the scope so internal consumers still work.
    Operation *cloned = builder.clone(*scalarOp, mapper);
    for (unsigned i = 0; i < scalarOp->getNumResults(); ++i) {
      Value original = scalarOp->getResult(i);
      Value replacement = cloned->getResult(i);
      for (OpOperand &use : llvm::make_early_inc_range(original.getUses())) {
        Operation *user = use.getOwner();
        if (!inUnit.contains(user) && !inClosure.contains(user))
          use.set(replacement);
      }
    }
    packList.push_back(scalarOp);
  }

  llvm::sort(packList,
             [](Operation *a, Operation *b) { return a->isBeforeInBlock(b); });

  auto scopeOp = packScopeOp(packList);
  if (!scopeOp)
    return llvm::failure();

  scopeOp->setAttr(CVPipeline::kScopeLoadAttr,
                   UnitAttr::get(scopeOp->getContext()));
  if (isExtractedLoadStore)
    scopeOp->setAttr(kCoreTypeAttr,
                     StringAttr::get(scopeOp->getContext(), "VECTOR"));
  return llvm::success();
}

std::unique_ptr<OperationPass<ModuleOp>> CVPipeline::createPackMemrefPass() {
  return std::make_unique<PackMemrefPass>();
}
