/*
 * Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 */

#include "DynamicCVPipeline/Common/Utils.h"
#include "DynamicCVPipeline/ComputeBlockOpt/CubePageLoaders.h"
#include "DynamicCVPipeline/ComputeBlockOpt/Passes.h"
#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVM/Utils/Utils.h"
#include "bishengir/Dialect/Utils/Util.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"
#include "mlir/Interfaces/ViewLikeInterface.h"
#include "llvm/ADT/SetVector.h"
#include <algorithm>

using namespace mlir;

namespace {

bool isZero(Value value) { return matchPattern(value, m_PosZeroFloat()); }

bool isGMSource(Value value) {
  while (true) {
    auto space = hivm::getOptionalHIVMAddressSpace(value.getType());
    if (space && *space != hivm::AddressSpace::GM)
      return false;
    auto view = value.getDefiningOp<ViewLikeOpInterface>();
    if (!view)
      break;
    value = view.getViewSource();
  }
  if (auto arg = dyn_cast<BlockArgument>(value))
    return isa<func::FuncOp>(arg.getOwner()->getParentOp());
  auto space = hivm::getOptionalHIVMAddressSpace(value.getType());
  return space && *space == hivm::AddressSpace::GM;
}

std::optional<CVPipeline::CubePageLoader>
getPageLoader(tensor::InsertSliceOp insert) {
  auto tensor = insert.getSource().getDefiningOp<bufferization::ToTensorOp>();
  if (!tensor || !tensor->hasOneUse() ||
      utils::getAnnotateOpWithAttr(tensor.getResult(),
                                   hivm::kMayImplicitTransposeWithLastAxis))
    return std::nullopt;
  auto alloc = tensor.getBuffer().getDefiningOp<memref::AllocOp>();
  if (!alloc ||
      alloc.getType().getShape() != insert.getSourceType().getShape() ||
      alloc->getBlock() != insert->getBlock() ||
      tensor->getBlock() != insert->getBlock())
    return std::nullopt;

  CVPipeline::CubePageLoader page{insert, tensor, {}, {}};
  SmallVector<Value> worklist{alloc.getResult()};
  llvm::SetVector<Operation *> bufferOps;
  bufferOps.insert(alloc);
  bool padded = false;
  while (!worklist.empty()) {
    Value buffer = worklist.pop_back_val();
    for (Operation *user : buffer.getUsers()) {
      if (user == tensor)
        continue;
      if (auto view = dyn_cast<memref::SubViewOp>(user)) {
        // Only the leading rectangular mask used by a contiguous page load.
        if (view.getSource() != alloc ||
            llvm::any_of(view.getStaticOffsets(),
                         [](int64_t x) { return x != 0; }) ||
            llvm::any_of(view.getStaticStrides(),
                         [](int64_t x) { return x != 1; }) ||
            view.getStaticSizes()[1] <= 0 ||
            view.getStaticSizes()[1] > alloc.getType().getDimSize(1))
          return std::nullopt;
        if (bufferOps.insert(view))
          worklist.push_back(view.getResult());
        continue;
      }
      if (auto copy = dyn_cast<memref::CopyOp>(user)) {
        if (copy.getTarget() != buffer || page.copy ||
            copy->getBlock() != insert->getBlock() ||
            !isGMSource(copy.getSource()))
          return std::nullopt;
        page.copy = copy;
        continue;
      }
      if (auto fill = dyn_cast<linalg::FillOp>(user)) {
        if (fill.getOutputs()[0] != alloc || !isZero(fill.getInputs()[0]))
          return std::nullopt;
        padded = true;
        bufferOps.insert(fill);
        continue;
      }
      if (isa<annotation::MarkOp>(user) &&
          !user->hasAttr(hivm::kMayImplicitTransposeWithLastAxis)) {
        bufferOps.insert(user);
        continue;
      }
      return std::nullopt;
    }
  }
  if (!page.copy || !page.copy->isBeforeInBlock(tensor))
    return std::nullopt;
  // Moving initialization past a writer would change the loaded values.
  for (Operation *op : bufferOps) {
    if (!isa<linalg::FillOp>(op))
      continue;
    while (op && op->getBlock() != insert->getBlock())
      op = op->getParentOp();
    if (!op || !op->isBeforeInBlock(page.copy))
      return std::nullopt;
  }
  auto srcType = dyn_cast<MemRefType>(page.copy.getSource().getType());
  SmallVector<int64_t> strides;
  int64_t offset;
  if (!srcType || srcType.getRank() != 2 ||
#if defined(__LLVM_MAJOR_VERSION_22_COMPATIBLE__)
      failed(srcType.getStridesAndOffset(strides, offset)) ||
#else
      failed(getStridesAndOffset(srcType, strides, offset)) ||
#endif
      strides[1] != 1 || srcType.getDimSize(1) <= 0 ||
      srcType.getDimSize(1) > alloc.getType().getDimSize(1))
    return std::nullopt;
  if (!padded && srcType.getShape() != alloc.getType().getShape())
    return std::nullopt;
  page.bufferOps.assign(bufferOps.begin(), bufferOps.end());
  return page;
}

} // namespace

std::optional<SmallVector<CVPipeline::CubePageLoader>>
CVPipeline::getCubePageLoaders(tensor::InsertSliceOp root) {
  if (!root)
    return std::nullopt;
  auto type = root.getType();
  if (type.getRank() != 2 || !type.hasStaticShape() ||
      (!type.getElementType().isF16() && !type.getElementType().isBF16()) ||
      type.getDimSize(0) <= 0 || type.getDimSize(1) <= 0 ||
      type.getDimSize(0) % 16 || type.getDimSize(1) % 16)
    return std::nullopt;

  int64_t end = type.getDimSize(0);
  SmallVector<CubePageLoader> pages;
  Value current = root.getResult();
  while (auto insert = current.getDefiningOp<tensor::InsertSliceOp>()) {
    auto sourceType = insert.getSourceType();
    if (insert != root && !insert->hasOneUse())
      return std::nullopt;
    if (insert->getBlock() != root->getBlock() || sourceType.getRank() != 2 ||
        !sourceType.hasStaticShape() || insert.getType() != type ||
        insert.getStaticOffsets()[1] != 0 ||
        insert.getStaticSizes()[1] != type.getDimSize(1) ||
        insert.getStaticSizes()[0] != sourceType.getDimSize(0) ||
        llvm::any_of(insert.getStaticStrides(),
                     [](int64_t x) { return x != 1; }))
      return std::nullopt;
    int64_t row = insert.getStaticOffsets()[0];
    int64_t rows = sourceType.getDimSize(0);
    // Each page occupies whole NZ rows, possibly part of one 16-row block.
    if (row < 0 || rows <= 0 || row + rows != end ||
        (rows < 16 ? row % 16 + rows > 16 : row % 16 || rows % 16))
      return std::nullopt;
    auto page = getPageLoader(insert);
    if (!page)
      return std::nullopt;
    pages.push_back(std::move(*page));
    end = row;
    current = insert.getDest();
  }
  if (end != 0 || !isa_and_nonnull<tensor::EmptyOp, linalg::FillOp>(
                      current.getDefiningOp()))
    return std::nullopt;
  std::reverse(pages.begin(), pages.end());
  return pages;
}

namespace {

void materializePages(ArrayRef<CVPipeline::CubePageLoader> pages) {
  auto root = pages.back().insert;
  auto type = root.getType();
  int64_t rows = type.getDimSize(0), cols = type.getDimSize(1);
  OpBuilder builder(root.getContext());
  auto cbuf = builder.getAttr<hivm::AddressSpaceAttr>(hivm::AddressSpace::L1);
  auto nzType = MemRefType::get({cols / 16, rows / 16, 16, 16},
                                type.getElementType(), nullptr, cbuf);
  // The whole aggregate belongs to the matmul's compute block. Keeping the
  // buffer inside the same iteration also prevents writes racing with MMAD.
  auto tag = [&](Operation *op) {
    op->setAttr(CVPipeline::kCoreType, builder.getStringAttr("CUBE"));
    if (auto id = root->getAttr(CVPipeline::kBlockId))
      op->setAttr(CVPipeline::kBlockId, id);
  };
  auto indexAttrs = [&](ArrayRef<int64_t> values) {
    return llvm::map_to_vector(values, [&](int64_t value) -> OpFoldResult {
      return builder.getIndexAttr(value);
    });
  };
  Location loc = root.getLoc();
  Operation *firstCopy = pages.front().copy;
  for (const auto &page : pages)
    if (page.copy->isBeforeInBlock(firstCopy))
      firstCopy = page.copy;
  builder.setInsertionPoint(firstCopy);
  auto aggregate = builder.create<memref::AllocOp>(loc, nzType);
  tag(aggregate);
  auto zero = builder.create<arith::ConstantOp>(
      loc, builder.getZeroAttr(type.getElementType()));
  tag(zero);
  auto fill = builder.create<linalg::FillOp>(loc, ValueRange{zero},
                                             ValueRange{aggregate});
  tag(fill);

  for (auto page : pages) {
    builder.setInsertionPoint(page.copy);
    int64_t row = page.insert.getStaticOffsets()[0];
    int64_t pageRows = page.insert.getSourceType().getDimSize(0);
    SmallVector<OpFoldResult> offsets = indexAttrs({0, row / 16, row % 16, 0});
    SmallVector<OpFoldResult> sizes = indexAttrs(
        {cols / 16, (pageRows + 15) / 16, std::min(pageRows, int64_t{16}), 16});
    SmallVector<OpFoldResult> strides = indexAttrs({1, 1, 1, 1});
    auto view = builder.create<memref::SubViewOp>(loc, aggregate, offsets,
                                                  sizes, strides);
    tag(view);
    Value source = page.copy.getSource();
    auto count = builder.create<memref::DimOp>(loc, source, 0);
    tag(count);
    auto c0 = builder.create<arith::ConstantIndexOp>(loc, 0);
    tag(c0);
    auto live = builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::sgt,
                                              count, c0);
    tag(live);
    auto guard = builder.create<scf::IfOp>(loc, live, false);
    tag(guard);
    builder.setInsertionPointToStart(&guard.getThenRegion().front());
    // ND2NZ uses the full aggregate's N-stride, so a short page writes only
    // its own rows. The initial zero fill covers masked and sentinel pages.
    auto load = builder.create<hivm::ND2NZOp>(loc, TypeRange{}, source, view,
                                              builder.getUnitAttr());
    tag(load);
    tag(guard.thenYield());
  }

  builder.setInsertionPoint(root);
  auto ndType =
      MemRefType::get(type.getShape(), type.getElementType(), nullptr, cbuf);
  auto layout = builder.create<hivm::ConvertLayoutOp>(
      loc, ndType, aggregate,
      builder.getAttr<hivm::DataLayoutAttr>(hivm::DataLayout::nZ),
      builder.getAttr<hivm::DataLayoutAttr>(hivm::DataLayout::ND));
  tag(layout);
  auto cast = builder.create<memref::MemorySpaceCastOp>(
      loc, MemRefType::get(type.getShape(), type.getElementType()),
      layout.getResult());
  tag(cast);
  auto tensor =
      builder.create<bufferization::ToTensorOp>(loc, type, cast, true, true);
  tag(tensor);
  root.replaceAllUsesWith(tensor.getResult());

  llvm::SetVector<Operation *> cleanup;
  for (const auto &page : llvm::reverse(pages)) {
    for (Operation *op : page.bufferOps)
      cleanup.insert(op->getParentOp());
    page.insert->erase();
    page.tensor->erase();
    page.copy->erase();
    for (Operation *op : llvm::reverse(page.bufferOps))
      op->erase();
  }
  for (Operation *op : cleanup)
    if (isa<scf::IfOp>(op) && isOpTriviallyDead(op))
      op->erase();
}

class MaterializeCubePageLoadersPass
    : public PassWrapper<MaterializeCubePageLoadersPass,
                         OperationPass<ModuleOp>> {
public:
  MLIR_DEFINE_EXPLICIT_INTERNAL_INLINE_TYPE_ID(MaterializeCubePageLoadersPass)
  void getDependentDialects(DialectRegistry &registry) const override {
    registry.insert<hivm::HIVMDialect, arith::ArithDialect,
                    memref::MemRefDialect, bufferization::BufferizationDialect,
                    linalg::LinalgDialect, scf::SCFDialect>();
  }
  StringRef getArgument() const override {
    return "materialize-cube-page-loaders";
  }
  StringRef getDescription() const override {
    return "Load unrolled matmul pages directly into one CUBE L1 buffer";
  }
  void runOnOperation() override {
    if (!CVPipeline::isCubeBlockMergeEnabled() ||
        CVPipeline::hasFallbackAttr(getOperation()))
      return;
    SmallVector<SmallVector<CVPipeline::CubePageLoader>> chains;
    getOperation().walk([&](tensor::InsertSliceOp root) {
      if (CVPipeline::getOpCoreType(root) != CVPipeline::CoreType::CUBE_ONLY)
        return;
      if (llvm::any_of(root->getUsers(), [](Operation *user) {
            return isa<tensor::InsertSliceOp>(user);
          }))
        return;
      auto pages = CVPipeline::getCubePageLoaders(root);
      if (!pages || llvm::any_of(*pages, [&](const auto &page) {
            return page.copy->getAttr(CVPipeline::kBlockId) !=
                       root->getAttr(CVPipeline::kBlockId) ||
                   page.insert->getAttr(CVPipeline::kBlockId) !=
                       root->getAttr(CVPipeline::kBlockId);
          })) {
        CVPipeline::setFallbackAttr(getOperation(), CVPipeline::ERRCODE_FAILED);
        return;
      }
      chains.push_back(std::move(*pages));
    });
    if (CVPipeline::hasFallbackAttr(getOperation()))
      return;
    for (const auto &pages : chains)
      materializePages(pages);
  }
};

} // namespace

std::unique_ptr<OperationPass<ModuleOp>>
mlir::triton::createMaterializeCubePageLoadersPass() {
  return std::make_unique<MaterializeCubePageLoadersPass>();
}
