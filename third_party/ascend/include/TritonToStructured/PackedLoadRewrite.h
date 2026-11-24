#ifndef TRITON_ASCEND_PACKED_LOAD_REWRITE_H
#define TRITON_ASCEND_PACKED_LOAD_REWRITE_H

#include "mlir/IR/PatternMatch.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::triton {

/// Coalesces tensor loads whose innermost dimension exhibits statically
/// provable sub-element replication (e.g., sub-byte / packed formats using
/// floor-division, right-shifts, or power-of-two bitmasks) into a compact
/// physical `tt.load` followed by canonical `tt.expand_dims`, `tt.broadcast`,
/// and `tt.reshape` operations.
class SubElementLoadCoalesce : public OpRewritePattern<LoadOp> {
public:
  explicit SubElementLoadCoalesce(MLIRContext *context,
                                  PatternBenefit benefit = 1)
      : OpRewritePattern<LoadOp>(context, benefit) {}

  LogicalResult matchAndRewrite(LoadOp op,
                                PatternRewriter &rewriter) const override;
};

using PackedLoadRewrite = SubElementLoadCoalesce;

} // namespace mlir::triton

#endif // TRITON_ASCEND_PACKED_LOAD_REWRITE_H
