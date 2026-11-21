// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchRequest.h"
#include "TritonToGraph/BlockBatching/ReductionModels.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/FormatVariadic.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace mlir::triton::cfg {
std::string describeBatchComputation(Operation *root) {
  llvm::json::Array aggregations;
  llvm::DenseMap<Operation *, int64_t> ids;
  root->walk([&](Operation *op) {
    if (isa<triton::ReduceOp, triton::ScanOp>(op))
      ids[op] = ids.size();
  });
  root->walk([&](Operation *op) {
    if (!ids.count(op))
      return;
    llvm::json::Object record;
    record["id"] = ids.lookup(op);
    record["kind"] = isa<triton::ScanOp>(op) ? "scan" : "reduce";
    auto type = op->getNumOperands()
                    ? dyn_cast<RankedTensorType>(op->getOperand(0).getType())
                    : RankedTensorType{};
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    if (type && axis && axis.getInt() >= 0 && axis.getInt() < type.getRank()) {
      llvm::json::Array shape, effective;
      int64_t effectiveAxis = 0;
      for (auto [dim, extent] : llvm::enumerate(type.getShape())) {
        shape.push_back(extent);
        if (extent != 1) {
          effective.push_back(extent);
          if (static_cast<int64_t>(dim) < axis.getInt())
            ++effectiveAxis;
        }
      }
      record["original_shape"] = std::move(shape);
      record["effective_shape"] = std::move(effective);
      record["original_axis"] = axis.getInt();
      record["lifted_axis"] = axis.getInt() + 1;
      record["axis_extent"] = type.getDimSize(axis.getInt());
      record["effective_axis"] = type.getDimSize(axis.getInt()) == 1
                                     ? llvm::json::Value(nullptr)
                                     : llvm::json::Value(effectiveAxis);
      record["domain"] =
          isa<triton::ReduceOp>(op)
              ? "original_program_reduction"
              : (hasBatchIndependentScanDomain(type, axis.getInt())
                     ? "independent_row_scan"
                     : "unsupported_multidimensional_scan");
    } else {
      record["domain"] = "unproven";
    }
    record["arity"] = static_cast<int64_t>(op->getNumOperands());
    int64_t loops = 0, branches = 0;
    for (Operation *parent = op->getParentOp(); parent && parent != root;
         parent = parent->getParentOp()) {
      loops += isa<scf::ForOp>(parent);
      branches += isa<scf::IfOp>(parent);
    }
    record["loop_depth"] = loops;
    record["conditional_depth"] = branches;
    // Immediate SSA producers only; this is not a transitive dependence proof.
    llvm::json::Array producers;
    for (Value input : op->getOperands()) {
      auto found = ids.find(input.getDefiningOp());
      producers.push_back(found == ids.end()
                              ? llvm::json::Value(nullptr)
                              : llvm::json::Value(found->second));
    }
    record["direct_input_aggregations"] = std::move(producers);
    aggregations.push_back(std::move(record));
  });
  llvm::json::Object result;
  result["version"] = 1;
  result["scope"] = "normalized_original_program";
  result["lane_policy"] = "separate_original_domains";
  result["aggregations"] = std::move(aggregations);
  return llvm::formatv("{0}", llvm::json::Value(std::move(result))).str();
}

namespace {
constexpr llvm::StringLiteral vvAggregationId = "__vv_aggregation_id";
}

void markVvAggregations(Operation *root) {
  int64_t id = 0;
  Builder builder(root->getContext());
  root->walk([&](Operation *op) {
    if (isa<triton::ReduceOp, triton::ScanOp>(op))
      op->setAttr(vvAggregationId, builder.getI64IntegerAttr(id++));
  });
}

LogicalResult finalizeVvComputation(Operation *root, std::string &description,
                                    bool sequential, int64_t factor) {
  auto parsed = llvm::json::parse(description);
  if (!parsed) {
    llvm::consumeError(parsed.takeError());
    return failure();
  }
  auto *object = parsed->getAsObject();
  auto *sources = object ? object->getArray("aggregations") : nullptr;
  if (!sources)
    return failure();
  SmallVector<llvm::json::Array> targets(sources->size());
  bool valid = true;
  int64_t generatedId = 0;
  root->walk([&](Operation *op) {
    if (!isa<triton::ReduceOp, triton::ScanOp>(op))
      return;
    auto id = op->getAttrOfType<IntegerAttr>(vvAggregationId);
    op->removeAttr(vvAggregationId);
    if (!id || id.getInt() < 0 ||
        static_cast<size_t>(id.getInt()) >= sources->size()) {
      valid = false;
      return;
    }
    auto *source = (*sources)[id.getInt()].getAsObject();
    auto *shape = source ? source->getArray("original_shape") : nullptr;
    auto originalAxis =
        source ? source->getInteger("original_axis") : std::optional<int64_t>{};
    auto type = op->getNumOperands()
                    ? dyn_cast<RankedTensorType>(op->getOperand(0).getType())
                    : RankedTensorType{};
    auto axis = op->getAttrOfType<IntegerAttr>("axis");
    if (!source || !shape || !originalAxis || !type || !axis ||
        source->getString("kind") !=
            (isa<triton::ScanOp>(op) ? "scan" : "reduce") ||
        source->getInteger("arity") != op->getNumOperands()) {
      valid = false;
      return;
    }
    // Uniform computations retain their original domain even in ProgramLanes.
    // A varying computation must have exactly one new leading lane dimension.
    bool prefix = !sequential && type.getRank() == shape->size() + 1;
    if (type.getRank() != shape->size() + static_cast<size_t>(prefix) ||
        (prefix && type.getDimSize(0) != factor) ||
        axis.getInt() != *originalAxis + static_cast<int64_t>(prefix)) {
      valid = false;
      return;
    }
    for (auto [dim, extent] : llvm::enumerate(*shape))
      if (extent.getAsInteger() != type.getDimSize(dim + prefix)) {
        valid = false;
        return;
      }
    llvm::json::Array generatedShape;
    for (int64_t extent : type.getShape())
      generatedShape.push_back(extent);
    targets[id.getInt()].push_back(
        llvm::json::Object{{"id", generatedId++},
                           {"axis", axis.getInt()},
                           {"shape", std::move(generatedShape)},
                           {"batch_prefix", prefix}});
  });
  if (!valid)
    return root->emitError("invalid VV aggregation correspondence"), failure();
  for (auto [id, source] : llvm::enumerate(*sources)) {
    auto &mapped = targets[id];
    if (mapped.empty() || (!sequential && mapped.size() != 1))
      return root->emitError("missing VV aggregation correspondence"),
             failure();
    auto axis = mapped.front().getAsObject()->getInteger("axis");
    for (const auto &target : mapped)
      if (target.getAsObject()->getInteger("axis") != axis)
        return failure();
    auto &record = *source.getAsObject();
    record["lifted_axis"] = *axis;
    record["mapped_operations"] = std::move(mapped);
  }
  (*object)["version"] = 2;
  (*object)["stage"] = "semantic_ttir_before_memory_views";
  (*object)["schedule"] =
      sequential ? "sequential_iterations" : "program_lanes";
  description = llvm::formatv("{0}", *parsed).str();
  return success();
}

BatchDiagnostic snapshotBatchDiagnostic(Operation *op, llvm::StringRef stage,
                                        llvm::StringRef reason) {
  BatchDiagnostic report;
  report.stage = stage.str();
  report.reason = reason.str();
  report.operation = op->getName().getStringRef().str();
  report.context = "BlockBody";
  Operation *child = op;
  for (Operation *parent = op->getParentOp(); parent;
       child = parent, parent = parent->getParentOp()) {
    if (parent->getName().getStringRef() == "tt.reduce") {
      report.context = "ReduceCombiner";
      break;
    }
    if (parent->getName().getStringRef() == "tt.scan") {
      report.context = "ScanCombiner";
      break;
    }
    if (parent->getName().getStringRef() == "scf.for") {
      report.context = "ForBody";
      break;
    }
    if (parent->getName().getStringRef() == "scf.if") {
      report.context = child->getParentRegion() == &parent->getRegion(0)
                           ? "IfThen"
                           : "IfElse";
      break;
    }
  }
  report.location.clear();
  {
    llvm::raw_string_ostream stream(report.location);
    op->getLoc().print(stream);
  }
  return report;
}

void applyBatchDiagnostic(BatchAnalysisReport &report,
                          const BatchDiagnostic &detail) {
  report.reason = detail.reason;
  report.operation = detail.operation;
  report.context = detail.context;
  report.location = detail.location;
}

void setBatchDiagnostic(BatchAnalysisReport &report, Operation *op,
                        llvm::StringRef reason) {
  applyBatchDiagnostic(report, snapshotBatchDiagnostic(op, "analysis", reason));
}

FailureOr<BatchAnalysisReport> analyzeBlockBatching(ModuleOp module,
                                                    BatchRequest request) {
  if (!request.isValid())
    return failure();
  BatchAnalysisReport report;
  if (request.factor == 1) {
    report.status = "DISABLED";
    report.reason = "DISABLED";
    return report;
  }
  if (request.factor > std::numeric_limits<int32_t>::max()) {
    report.reason = "FACTOR_OUT_OF_RANGE";
    return report;
  }
  unsigned entries = 0;
  for (auto function : module.getOps<triton::FuncOp>())
    if (function.isPublic())
      ++entries;
  if (entries != 1) {
    report.reason = "EXPECTED_SINGLE_PUBLIC_ENTRY";
    return report;
  }
  module.walk([&](Operation *op) {
    const auto name = op->getName().getStringRef();
    ++report.operations[name.str()];
  });
  if (report.reason.empty()) {
    // Analyze with the same registry and verifier on a private clone. This
    // report never authorizes a launch: alias/resource/mode proofs remain.
    request.semanticOnly = true;
    auto rewrite = probeBlockBatching(module, request);
    if (failed(rewrite))
      return failure();
    report.status =
        rewrite->status == "REWRITTEN" ? "ANALYZED" : rewrite->status;
    report.reason = rewrite->reason;
    report.location = rewrite->location;
    report.operation = rewrite->operation;
    report.context = rewrite->context;
    report.layoutJSON = rewrite->layoutJSON;
    report.computationJSON = rewrite->computationJSON;
  }
  return report;
}
} // namespace mlir::triton::cfg
