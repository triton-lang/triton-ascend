// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchPlan.h"
#include "triton/Dialect/Triton/IR/Dialect.h"
#include "llvm/Support/raw_ostream.h"
#include <limits>

namespace mlir::triton::cfg {
void setBatchDiagnostic(BatchAnalysisReport &report, Operation *op,
                        llvm::StringRef reason) {
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
  llvm::raw_string_ostream stream(report.location);
  op->getLoc().print(stream);
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
  }
  return report;
}
} // namespace mlir::triton::cfg
