// Copyright (c) Huawei Technologies Co., Ltd. 2025. All rights reserved.
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
// THE SOFTWARE.
// TLA-specific Ascend IR builder bindings.
#include "Utils/AscendNPUIROpBuilder.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

#include <optional>
#include <utility>
#include <vector>

using namespace mlir;
namespace py = pybind11;

struct TLABuilder : public AscendNPUIROpBuilder {
  explicit TLABuilder(MLIRContext *context, std::string target = "",
                      std::string compile_mode = "simd")
      : AscendNPUIROpBuilder(context, target, compile_mode) {}
};

namespace {

// Resolve the element type of an AVE vector register from the precise tag
// passed by the SIMD frontend (extension/simd.py::_elem_tag).  Each tag maps to
// exactly one MLIR type, keeping the frontend's declared dtype identical to the
// handle's IR type:
//   "i8"       -> i8
//   "bf16"     -> bf16
//   "f32"      -> f32      (decode output)
//   "fp8e4nv"  -> f8E4M3FN (E4M3FN profile)
Type resolveSimdElementType(MLIRContext *ctx, const py::object &elem) {
  if (!py::isinstance<py::str>(elem)) {
    throw std::invalid_argument(
        "simd element spec must be a tag string: i8 / bf16 / f32 / fp8e4nv");
  }
  const std::string tag = elem.cast<std::string>();
  if (tag == "i8")
    return IntegerType::get(ctx, 8);
  if (tag == "i16")
    return IntegerType::get(ctx, 16);
  if (tag == "i32")
    return IntegerType::get(ctx, 32);
  if (tag == "i64")
    return IntegerType::get(ctx, 64);
  if (tag == "bf16")
    return BFloat16Type::get(ctx);
  if (tag == "f16")
    return Float16Type::get(ctx);
  if (tag == "f32")
    return Float32Type::get(ctx);
  if (tag == "fp8e4nv")
    return Float8E4M3FNType::get(ctx);
  if (tag == "fp8e5")
    return Float8E5M2Type::get(ctx);
  throw std::invalid_argument("unknown simd element tag \"" + tag +
                              "\"; expected i8 / bf16 / f32 / fp8e4nv");
}

// memref.view needs a one-dimensional identity-layout byte source. Rebase a
// strided descriptor without losing its offset or the buffer's alias identity.
Value predicateByteView(TLABuilder &self, Value ptr, Value byteOffset) {
  auto &b = self.getBuilder();
  auto ty = cast<MemRefType>(ptr.getType());
  if (!ty.getLayout().isIdentity()) {
    auto metadata = self.create<memref::ExtractStridedMetadataOp>(ptr);
    Value offset = metadata.getOffset();
    byteOffset = self.create<arith::AddIOp>(offset, byteOffset);
    Value extent =
        self.create<arith::AddIOp>(offset, metadata.getSizes().front());
    auto baseTy =
        MemRefType::get({ShapedType::kDynamic}, b.getI8Type(),
                        MemRefLayoutAttrInterface{}, ty.getMemorySpace());
    // reinterpret_cast replaces the descriptor offset; view below applies it
    // exactly once, in bytes. Keep ptr as the source for memory-effect
    // analysis.
    ptr = self.create<memref::ReinterpretCastOp>(
        baseTy, ptr, b.getIndexAttr(0), ArrayRef<OpFoldResult>{extent},
        ArrayRef<OpFoldResult>{b.getIndexAttr(1)});
  }
  auto bitsTy = MemRefType::get(
      {256}, b.getI1Type(), MemRefLayoutAttrInterface{}, ty.getMemorySpace());
  return self.create<memref::ViewOp>(bitsTy, ptr, byteOffset, ValueRange{});
}

// Core type of the enclosing TLA core-mode scope, defaulting to the AIV
// (vector) core that plain TLA kernels run on.
hivm::TCoreType enclosingCoreType(OpBuilder &builder) {
  Block *block = builder.getInsertionBlock();
  for (Operation *op = block ? block->getParentOp() : nullptr; op;
       op = op->getParentOp()) {
    if (auto attr = op->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type"))
      return attr.getTcoretype();
  }
  return hivm::TCoreType::VECTOR;
}

// Pipes available to each core for explicit synchronization.
bool isSyncPipeSupported(hivm::PIPE pipe, hivm::TCoreType core) {
  switch (core) {
  case hivm::TCoreType::VECTOR:
    return pipe == hivm::PIPE::PIPE_S || pipe == hivm::PIPE::PIPE_V ||
           pipe == hivm::PIPE::PIPE_MTE2 || pipe == hivm::PIPE::PIPE_MTE3;
  case hivm::TCoreType::CUBE:
    return pipe == hivm::PIPE::PIPE_S || pipe == hivm::PIPE::PIPE_M ||
           pipe == hivm::PIPE::PIPE_MTE1 || pipe == hivm::PIPE::PIPE_MTE2 ||
           pipe == hivm::PIPE::PIPE_MTE3 || pipe == hivm::PIPE::PIPE_FIX;
  default:
    return false;
  }
}

// Resolve a frontend pipe name ("PIPE_MTE2") to its HIVM attribute and check
// that the enclosing core can use it. PIPE_ALL is only valid for barrier.
hivm::PipeAttr getSyncPipe(OpBuilder &builder, const std::string &pipe,
                           bool allowAll = false) {
  auto pipeEnum = hivm::symbolizePIPE(pipe);
  if (!pipeEnum)
    throw std::invalid_argument("unknown sync pipe: " + pipe);
  auto core = enclosingCoreType(builder);
  if (!(allowAll && *pipeEnum == hivm::PIPE::PIPE_ALL) &&
      !isSyncPipeSupported(*pipeEnum, core))
    throw std::invalid_argument(
        "sync pipe " + pipe + " is not supported by the enclosing " +
        (core == hivm::TCoreType::CUBE ? "cube" : "vector") + " core");
  return hivm::PipeAttr::get(builder.getContext(), *pipeEnum);
}

// Hardware ID limits and the frontend value that selects MODE.INTRA.
constexpr int64_t kMaxEventId = 7;
constexpr int64_t kMaxBlockFlagId = 15;
constexpr int64_t kMaxCubeIntraBlockId = 31;
constexpr int64_t kMaxVectorIntraBlockId = 15;
constexpr int64_t kMaxBufferId = 31;
constexpr int kSyncModeIntra = 3;

// Validate a frontend constant against its hardware range [0, max_id].
int64_t checkSyncRange(int64_t value, int64_t max_id, const char *what) {
  if (value < 0 || value > max_id)
    throw std::invalid_argument(std::string(what) + " must be in [0, " +
                                std::to_string(max_id) + "]");
  return value;
}

// Preserve a static ID or a runtime i64 scalar ID.
OpFoldResult getSyncId(TLABuilder &self, const py::object &id_value,
                       int64_t max_id, const char *what) {
  if (py::isinstance<py::int_>(id_value) &&
      !py::isinstance<py::bool_>(id_value))
    return self.getBuilder().getI64IntegerAttr(
        checkSyncRange(id_value.cast<int64_t>(), max_id, what));
  auto value = id_value.cast<Value>();
  if (!value.getType().isInteger(64))
    throw std::invalid_argument(std::string(what) + " must be an i64 scalar");
  return value;
}

template <typename Op>
void createEventSync(TLABuilder &self, const std::string &pipe,
                     const std::string &waitPipe, const py::object &event_id) {
  auto id = getSyncId(self, event_id, kMaxEventId, "event_id");
  hivm::EventAttr staticId;
  Value dynamicId;
  if (auto attr = dyn_cast<Attribute>(id))
    staticId = hivm::EventAttr::get(
        self.getBuilder().getContext(),
        static_cast<hivm::EVENT>(cast<IntegerAttr>(attr).getInt()));
  else
    dynamicId = cast<Value>(id);
  self.create<Op>(getSyncPipe(self.getBuilder(), pipe),
                  getSyncPipe(self.getBuilder(), waitPipe), staticId,
                  dynamicId);
}

// Cross-core set/wait are block-level synchronization: emit them with the
// block instruction mode requested by the frontend and the core type of the
// enclosing TLA scope. INTRA uses direct A5 intra-block counters, which keep
// their physical IDs instead of the automatic partner offset.
void createBlockSync(TLABuilder &self, const std::string &pipe,
                     const py::object &flag_id, int sync_mode, bool is_set) {
  auto &builder = self.getBuilder();
  auto *ctx = builder.getContext();
  auto core = enclosingCoreType(builder);
  auto coreAttr = hivm::TCoreTypeAttr::get(ctx, core);
  auto pipeAttr = getSyncPipe(builder, pipe);
  checkSyncRange(sync_mode, kSyncModeIntra, "sync mode");
  if (sync_mode == kSyncModeIntra) {
    const int64_t max_flag_id = core == hivm::TCoreType::CUBE
                                    ? kMaxCubeIntraBlockId
                                    : kMaxVectorIntraBlockId;
    auto flag_id_attr = getSyncId(self, flag_id, max_flag_id, "flagid");
    if (is_set)
      self.create<hivm::SyncIntraBlockSetOp>(coreAttr, pipeAttr, flag_id_attr);
    else
      self.create<hivm::SyncIntraBlockWaitOp>(coreAttr, pipeAttr, flag_id_attr);
    return;
  }
  auto flag_id_attr = getSyncId(self, flag_id, kMaxBlockFlagId, "flagid");
  auto mode_attr = hivm::SyncBlockInstrModeAttr::get(
      ctx, static_cast<hivm::SyncBlockInstrMode>(sync_mode));
  if (is_set)
    self.create<hivm::SyncBlockSetOp>(coreAttr, pipeAttr, pipeAttr,
                                      flag_id_attr, Value{}, mode_attr);
  else
    self.create<hivm::SyncBlockWaitOp>(coreAttr, pipeAttr, pipeAttr,
                                       flag_id_attr, mode_attr);
}

// Expand the TLA global barrier before handing IR to the affinity backend.
// Delay until all scopes are known: ALL in a pure AIV kernel is different
// from ALL in a mixed kernel, and a top-level mixed barrier needs both scopes.
void lowerTlaSyncAll(ModuleOp module) {
  auto *ctx = module.getContext();
  auto moduleCore =
      module->getAttrOfType<hivm::TModuleCoreTypeAttr>("hivm.module_core_type");
  bool mixed = moduleCore &&
               moduleCore.getModuleCoreType() == hivm::TModuleCoreType::MIX;
  SmallVector<hivm::SyncBlockOp> barriers;
  module.walk([&](hivm::SyncBlockOp op) { barriers.push_back(op); });
  for (auto op : barriers) {
    auto mode = op.getSyncBlockModeAttr().getSyncMode();
    if (mode != hivm::SyncBlockMode::ALL &&
        mode != hivm::SyncBlockMode::ALL_VECTOR)
      throw std::invalid_argument("unexpected TLA global barrier mode");
    OpBuilder builder(op);
    auto loc = op.getLoc();
    auto pipe = [&](hivm::PIPE value) {
      return hivm::PipeAttr::get(ctx, value);
    };
    auto cube = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::CUBE);
    auto vector = hivm::TCoreTypeAttr::get(ctx, hivm::TCoreType::VECTOR);
    auto inter = hivm::SyncBlockInstrModeAttr::get(
        ctx, hivm::SyncBlockInstrMode::INTER_BLOCK_SYNCHRONIZATION);
    auto intra = hivm::SyncBlockInstrModeAttr::get(
        ctx, hivm::SyncBlockInstrMode::INTRA_BLOCK_SYNCHRONIZATION);
    auto set = [&](hivm::TCoreTypeAttr target, hivm::PipeAttr src,
                   hivm::PipeAttr dst, int64_t id,
                   hivm::SyncBlockInstrModeAttr instrMode) {
      builder.create<hivm::SyncBlockSetOp>(loc, target, src, dst,
                                           builder.getI64IntegerAttr(id),
                                           Value{}, instrMode);
    };
    auto wait = [&](hivm::TCoreTypeAttr target, hivm::PipeAttr src,
                    hivm::PipeAttr dst, int64_t id,
                    hivm::SyncBlockInstrModeAttr instrMode) {
      builder.create<hivm::SyncBlockWaitOp>(
          loc, target, src, dst, builder.getI64IntegerAttr(id), instrMode);
    };
    auto emit = [&](bool isCube) {
      auto core = isCube ? cube : vector;
      auto src = pipe(isCube ? hivm::PIPE::PIPE_FIX : hivm::PIPE::PIPE_MTE3);
      auto dst = pipe(hivm::PIPE::PIPE_S);
      builder.create<hivm::PipeBarrierOp>(loc, pipe(hivm::PIPE::PIPE_ALL));
      if (mode == hivm::SyncBlockMode::ALL_VECTOR || !mixed) {
        // Match HIVMDecomposeOp's default flags: ALL_VECTOR=14,
        // ALL on a pure Cube=13 or Vector=12 kernel.
        int64_t id = mode == hivm::SyncBlockMode::ALL_VECTOR ? 14
                     : isCube                                ? 13
                                                             : 12;
        set(core, src, dst, id, inter);
        wait(core, src, dst, id, inter);
      } else if (isCube) {
        wait(cube, pipe(hivm::PIPE::PIPE_MTE3), dst, 10, intra);
        set(cube, src, dst, 13, inter);
        wait(cube, src, dst, 13, inter);
        set(cube, src, dst, 11, intra);
      } else {
        set(vector, src, dst, 10, intra);
        wait(vector, pipe(hivm::PIPE::PIPE_FIX), dst, 11, intra);
      }
    };
    std::optional<bool> enclosingCube;
    for (Operation *parent = op->getParentOp(); parent;
         parent = parent->getParentOp()) {
      if (parent->hasAttr("hivm.vector_function"))
        throw std::invalid_argument(
            "sync_all must be outside SIMD/SIMT scopes");
      if (!enclosingCube) {
        if (auto core =
                parent->getAttrOfType<hivm::TCoreTypeAttr>("hivm.tcore_type"))
          enclosingCube = core.getTcoretype() == hivm::TCoreType::CUBE;
      }
    }
    if (mode == hivm::SyncBlockMode::ALL_VECTOR &&
        enclosingCube.value_or(moduleCore && moduleCore.getModuleCoreType() ==
                                                 hivm::TModuleCoreType::AIC))
      throw std::invalid_argument("sync_all(False) requires a vector core");
    if (mixed && !enclosingCube) {
      auto emitScope = [&](bool isCube) {
        OpBuilder::InsertionGuard guard(builder);
        auto scope = builder.create<scope::ScopeOp>(loc, TypeRange{});
        scope->setAttr("hivm.tcore_type", isCube ? cube : vector);
        if (scope.getRegion().empty())
          scope.getRegion().emplaceBlock();
        builder.setInsertionPointToStart(&scope.getRegion().front());
        emit(isCube);
        builder.create<scope::ReturnOp>(loc);
      };
      if (mode == hivm::SyncBlockMode::ALL)
        emitScope(true);
      emitScope(false);
    } else {
      bool isCube =
          enclosingCube.value_or(moduleCore && moduleCore.getModuleCoreType() ==
                                                   hivm::TModuleCoreType::AIC);
      emit(isCube);
    }
    op.erase();
  }
}

template <typename Op>
void createBufferSync(TLABuilder &self, const py::object &buf_id,
                      const std::string &pipe, bool nonblocking) {
  auto &builder = self.getBuilder();
  // PR4478 buffer-sync ops identify the core by their pipe and no longer
  // carry a TCoreTypeAttr. Keep the frontend pipe/core check for diagnostics.
  self.create<Op>(getSyncPipe(builder, pipe),
                  getSyncId(self, buf_id, kMaxBufferId, "bufid"),
                  builder.getBoolAttr(nonblocking));
}

// vload/masked_store indices must be index-typed.
llvm::SmallVector<Value> toIndexValues(TLABuilder &self,
                                       std::vector<Value> &indices) {
  auto &builder = self.getBuilder();
  llvm::SmallVector<Value> indexValues;
  for (auto val : indices) {
    if (!val.getType().isIndex()) {
      val = self.create<arith::IndexCastOp>(builder.getIndexType(), val);
    }
    indexValues.push_back(val);
  }
  return indexValues;
}

// Zero-offset row-major memrefs stay identity: the canonical form, and the
// only layout PointerCastOp's LLVM lowering accepts. Subview/reinterpret
// results keep strided layouts via the dynamic-offset branch below.
MemRefLayoutAttrInterface tlaBufferLayout(MLIRContext *ctx,
                                          ArrayRef<int64_t> shape,
                                          ArrayRef<int64_t> strides,
                                          std::optional<int64_t> offset) {
  bool isIdentity = offset.has_value() && *offset == 0;
  for (int64_t expectedStride = 1, i = shape.size(); i-- > 0;) {
    if (shape[i] > 1 && strides[i] != expectedStride)
      isIdentity = false;
    expectedStride *= shape[i];
  }
  if (isIdentity)
    return {};
  return StridedLayoutAttr::get(ctx, offset.value_or(ShapedType::kDynamic),
                                strides);
}

void checkDmaBuffers(Value src, Value dst) {
  if (!isa<MemRefType>(src.getType()) || !isa<MemRefType>(dst.getType()))
    throw std::invalid_argument("TLA DMA requires memref operands");
}

void setDmaAttrs(Operation *op, const py::dict &attrs) {
  for (auto item : attrs)
    if (!item.second.is_none())
      op->setAttr(py::cast<std::string>(item.first),
                  py::cast<Attribute>(item.second));
}

Value dmaIndexOperand(TLABuilder &self, Value value) {
  if (value && !value.getType().isIndex())
    return self.create<arith::IndexCastOp>(self.getBuilder().getIndexType(),
                                           value);
  return value;
}

// UnitFlag encoding belongs to native op construction. MMAD and Fixpipe
// share the backend enum mapping without exposing an attribute factory.
void setUnitFlagAttrs(OpBuilder &builder, Operation *op,
                      std::optional<int32_t> control,
                      std::optional<int64_t> groupId) {
  if (control) {
    auto mode = hivm::symbolizeUNIT_FLAG(static_cast<uint32_t>(*control));
    if (!mode)
      throw std::invalid_argument("unknown HIVM UnitFlag mode: " +
                                  std::to_string(*control));
    op->setAttr("unit_flag_mode", builder.getArrayAttr({hivm::UnitFlagAttr::get(
                                      builder.getContext(), *mode)}));
  }
  if (groupId)
    op->setAttr("unit_flag_group_id", builder.getI64IntegerAttr(*groupId));
}

// Static rank-reshape pieces: memref.collapse_shape / expand_shape with an
// explicit reassociation (groups of source dims per result dim).  The
// decomposition logic lives in the Python frontend (tla.reshape); these
// bindings only materialize the ops.  Shapes must be static and identity
// layouts are enforced there, so the emitted ops stay canonical.
template <typename ReshapeOp>
Value createReshapeOp(TLABuilder &self, Value source, Type resultType,
                      const std::vector<std::vector<int64_t>> &reassociation) {
  SmallVector<ReassociationIndices> groups;
  for (const auto &group : reassociation)
    groups.push_back(ReassociationIndices(group.begin(), group.end()));
  return self.getBuilder()
      .create<ReshapeOp>(self.getLastLoc(), resultType, source, groups)
      .getResult();
}

// Mode-aware lowering shared by the SIMD value ops that expose
// MODE_ZEROING / MODE_MERGING (simd.py).  AVE regbase value ops have no
// mutable dst operand, so "merging" cannot be CANN's dst-preserving predicate.
// The TLA contract defines the merge source as the same-typed +0 SSA vector:
//   * "zeroing": return the op result unchanged; these intrinsics already
//     read inactive lanes as zero.
//   * "merging": materialize that +0 vector (arith.constant zero +
//     ave.hir.scalar_broadcast) and select it for the inactive lanes with
//     ave.hir.vsel.  The mode is consumed here, producing a visible
//     zero-init + merge combination rather than an unconsumed attribute.
// Python passes the normalized mode string; an unknown spelling is rejected
// instead of silently degrading to zeroing.
Value applySimdMode(TLABuilder &self, Value result, Value mask,
                    const std::string &mode) {
  if (mode == "zeroing")
    return result;
  if (mode != "merging")
    throw std::invalid_argument("unknown simd mode \"" + mode +
                                "\"; expected \"zeroing\" or \"merging\"");
  auto &builder = self.getBuilder();
  auto loc = self.getLastLoc();
  auto vecTy = cast<VectorType>(result.getType());
  auto elemTy = vecTy.getElementType();
  Value zeroScalar;
  if (isa<IntegerType>(elemTy)) {
    zeroScalar =
        builder.create<arith::ConstantOp>(loc, IntegerAttr::get(elemTy, 0));
  } else {
    zeroScalar =
        builder.create<arith::ConstantOp>(loc, FloatAttr::get(elemTy, 0.0));
  }
  Value zeroVector =
      self.create<hivmave::VFBroadcastScalarOp>(vecTy, zeroScalar).getResult();
  return self.create<hivmave::VFSelectOp>(vecTy, mask, result, zeroVector)
      .getResult();
}

} // namespace

void init_tla_builder(py::module &&m) {
  m.def("lower_tla_sync_all", &lowerTlaSyncAll, py::arg("module"));
  py::class_<TLABuilder, AscendNPUIROpBuilder>(
      m, "tla_builder", py::module_local(), py::dynamic_attr())
      .def(py::init<MLIRContext *, std::string, std::string>(),
           py::arg("context"), py::arg("target") = "",
           py::arg("compile_mode") = "simd")
      .def(
          "create_tla_vector_type",
          [](TLABuilder &, int64_t lanes, Type elementType) -> Type {
            if (lanes <= 0 || !elementType.isIntOrFloat())
              throw std::invalid_argument("vector requires positive lanes and "
                                          "an integer or float element type");
            return VectorType::get({lanes}, elementType);
          },
          py::arg("lanes"), py::arg("element_type"))
      // Intra-core pipeline synchronization.
      .def(
          "create_tla_barrier",
          [](TLABuilder &self, const std::string &pipe) {
            self.create<hivm::PipeBarrierOp>(
                getSyncPipe(self.getBuilder(), pipe, /*allowAll=*/true));
          },
          py::arg("pipe"))
      .def(
          "create_tla_set_flag",
          [](TLABuilder &self, const std::string &pipe,
             const std::string &waitPipe, const py::object &event_id) {
            createEventSync<hivm::SetFlagOp>(self, pipe, waitPipe, event_id);
          },
          py::arg("pipe"), py::arg("wait_pipe"), py::arg("event_id"))
      .def(
          "create_tla_wait_flag",
          [](TLABuilder &self, const std::string &pipe,
             const std::string &waitPipe, const py::object &event_id) {
            createEventSync<hivm::WaitFlagOp>(self, pipe, waitPipe, event_id);
          },
          py::arg("pipe"), py::arg("wait_pipe"), py::arg("event_id"))
      // Hardware global barrier: AIV-only (false) or all AIC/AIV (true).
      .def(
          "create_tla_sync_all",
          [](TLABuilder &self, bool all_cores) {
            auto *ctx = self.getBuilder().getContext();
            auto pipeAll = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
            auto syncMode = hivm::SyncBlockModeAttr::get(
                ctx, all_cores ? hivm::SyncBlockMode::ALL
                               : hivm::SyncBlockMode::ALL_VECTOR);
            self.create<hivm::SyncBlockOp>(
                syncMode, IntegerAttr{}, Value{},
                all_cores ? pipeAll : hivm::PipeAttr{}, pipeAll);
          },
          py::arg("all_cores"))
      // Logical block modes 0..2 and mode 3 for one physical A5 counter.
      .def(
          "create_tla_block_set_flag",
          [](TLABuilder &self, int sync_mode, const std::string &pipe,
             const py::object &flag_id) {
            createBlockSync(self, pipe, flag_id, sync_mode, /*is_set=*/true);
          },
          py::arg("mode"), py::arg("pipe"), py::arg("flag_id"))
      .def(
          "create_tla_block_wait_flag",
          [](TLABuilder &self, int sync_mode, const std::string &pipe,
             const py::object &flag_id) {
            createBlockSync(self, pipe, flag_id, sync_mode, /*is_set=*/false);
          },
          py::arg("mode"), py::arg("pipe"), py::arg("flag_id"))
      // A5 buffer synchronization tokens.
      .def(
          "create_tla_mutex_lock",
          [](TLABuilder &self, const py::object &buf_id,
             const std::string &pipe, bool nonblocking) {
            createBufferSync<hivm::GetBufOp>(self, buf_id, pipe, nonblocking);
          },
          py::arg("buf_id"), py::arg("pipe"), py::arg("nonblocking") = false)
      .def(
          "create_tla_mutex_unlock",
          [](TLABuilder &self, const py::object &buf_id,
             const std::string &pipe, bool nonblocking) {
            createBufferSync<hivm::RlsBufOp>(self, buf_id, pipe, nonblocking);
          },
          py::arg("buf_id"), py::arg("pipe"), py::arg("nonblocking") = false)
      // SIMD vector-register ops (AVE dialect, ave.hir.*).
      .def(
          "create_simd_vgather",
          [](TLABuilder &self, Value ptr, std::vector<Value> indices,
             Value index, Value mask, py::object elem, int32_t lanes) -> Value {
            auto &builder = self.getBuilder();
            auto vecTy = VectorType::get(
                {lanes}, resolveSimdElementType(builder.getContext(), elem));
            auto indicesIdx = toIndexValues(self, indices);
            return self
                .create<hivmave::VFGatherOp>(vecTy, ptr, ValueRange(indicesIdx),
                                             index, mask)
                .getResult();
          },
          py::arg("ptr"), py::arg("indices"), py::arg("index"), py::arg("mask"),
          py::arg("elem"), py::arg("lanes"))
      .def(
          "create_simd_pst",
          [](TLABuilder &self, Value ptr, std::vector<Value> indices, Value val,
             const std::string &pb) {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto pbEnum = hivmave::symbolizeFunctionDistType(pb);
            if (!pbEnum || (pb != "pb8" && pb != "pb16" && pb != "pb32"))
              throw std::invalid_argument("invalid predicate store density");
            if (indices.size() != 1)
              throw std::invalid_argument(
                  "predicate store needs one byte offset");
            auto indicesIdx = toIndexValues(self, indices);
            ptr = predicateByteView(self, ptr, indicesIdx.front());
            Value zero = self.create<arith::ConstantIndexOp>(0);
            Value mask = self.create<hivmave::VFPgeOp>(
                val.getType(), hivmave::PgePattern::ALL);
            auto op = self.create<hivmave::VFMaskedStoreOp>(
                hivmave::StoreDistAttr::get(ctx, hivmave::StoreDist::NORM_B8),
                ptr, ValueRange{zero}, mask, val);
            op->setAttr("functionType",
                        hivmave::FunctionDistTypeAttr::get(ctx, *pbEnum));
          },
          py::arg("ptr"), py::arg("indices"), py::arg("val"), py::arg("pb"))
      .def(
          "create_simd_pld",
          [](TLABuilder &self, Value ptr, std::vector<Value> indices,
             int32_t lanes, const std::string &dist,
             const std::string &pb) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto distEnum = hivmave::symbolizeLoadDist(dist);
            auto pbEnum = hivmave::symbolizeFunctionDistType(pb);
            if (!distEnum || !pbEnum ||
                (dist != "NORM" && dist != "US" && dist != "DS") ||
                (pb != "pb8" && pb != "pb16" && pb != "pb32"))
              throw std::invalid_argument(
                  "invalid predicate load distribution");
            if (indices.size() != 1)
              throw std::invalid_argument(
                  "predicate load needs one byte offset");
            auto vecTy = VectorType::get({lanes}, builder.getI1Type());
            auto indicesIdx = toIndexValues(self, indices);
            ptr = predicateByteView(self, ptr, indicesIdx.front());
            Value zero = self.create<arith::ConstantIndexOp>(0);
            auto op = self.create<hivmave::VFLoadOp>(
                vecTy, hivmave::LoadDistAttr::get(ctx, *distEnum), ptr,
                ValueRange{zero});
            op->setAttr("functionType",
                        hivmave::FunctionDistTypeAttr::get(ctx, *pbEnum));
            return op.getResult(0);
          },
          py::arg("ptr"), py::arg("indices"), py::arg("lanes"), py::arg("dist"),
          py::arg("pb"))
      .def(
          "create_simd_vlds",
          [](TLABuilder &self, Value &ptr, std::vector<Value> &indices,
             py::object elem_tag, int32_t lanes, std::string &dist) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto loc = self.getLastLoc();
            auto vecTy =
                VectorType::get({lanes}, resolveSimdElementType(ctx, elem_tag));
            auto distEnum = hivmave::symbolizeLoadDist(dist);
            if (!distEnum) {
              throw std::invalid_argument("unknown load dist: " + dist);
            }
            auto distAttr = hivmave::LoadDistAttr::get(ctx, *distEnum);
            auto indicesIdx = toIndexValues(self, indices);
            auto op = builder.create<hivmave::VFLoadOp>(
                loc, vecTy, distAttr, ptr, ValueRange(indicesIdx));
            return op.getResult(0);
          },
          py::arg("ptr"), py::arg("indices"), py::arg("elem_tag"),
          py::arg("lanes"), py::arg("dist"))
      .def(
          "create_simd_vsts",
          [](TLABuilder &self, Value &ptr, std::vector<Value> &indices,
             Value &val, Value &mask, std::string &dist) -> void {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto loc = self.getLastLoc();
            auto distEnum = hivmave::symbolizeStoreDist(dist);
            if (!distEnum) {
              throw std::invalid_argument("unknown store dist: " + dist);
            }
            auto distAttr = hivmave::StoreDistAttr::get(ctx, *distEnum);
            auto indicesIdx = toIndexValues(self, indices);
            builder.create<hivmave::VFMaskedStoreOp>(
                loc, distAttr, ptr, ValueRange(indicesIdx), mask, val);
          },
          py::arg("ptr"), py::arg("indices"), py::arg("val"), py::arg("mask"),
          py::arg("dist"))
      .def(
          "create_simd_binary",
          [](TLABuilder &self, std::string &kind, Value &lhs, Value &rhs,
             Value &mask, std::string mode) -> Value {
            auto resTy = lhs.getType();
            Value result;
            if (kind == "mul") {
              result = self.create<hivmave::VFMulOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "add") {
              result = self.create<hivmave::VFAddOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "sub") {
              result = self.create<hivmave::VFSubOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "or") {
              result = self.create<hivmave::VFOrOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "and") {
              result = self.create<hivmave::VFAndOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "div") {
              result = self.create<hivmave::VFDivOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "divfhp") {
              // f32 precision='exact' / 'vdiv_0ulp_ftz_true': the real
              // high-precision AVE division (ave.hir.vdivfhp).  Its pinned
              // regbase lowering (vdiv_float_hp) is TileLang's
              // vdiv_0ulp_ftz_true 0ULP sequence.  The caller mask is passed
              // straight through; Python rejects non-f32 before this point
              // because the library wrapper only has a float overload.
              result = self.create<hivmave::VFDivFHPOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "xor") {
              result = self.create<hivmave::VFXorOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "mod") {
              result = self.create<hivmave::VFModOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "absdiff") {
              result = self.create<hivmave::VFAbsDiffOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "expdif") {
              result = self.create<hivmave::VFExpdifOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "prelu") {
              result = self.create<hivmave::VFPReluOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "max") {
              result = self.create<hivmave::VFMaxOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "min") {
              result = self.create<hivmave::VFMinOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "umax") {
              // Unsigned integer variants.  MLIR integers are signless, so
              // the frontend's signedness cannot live in the element type;
              // unsigned comparison/division needs its own AVE op (lib name
              // ..._uintN) or, for vdiv, the `cast` attribute.
              result = self.create<hivmave::VMaxUIOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "umin") {
              result = self.create<hivmave::VMinUIOp>(resTy, lhs, rhs, mask)
                           .getResult();
            } else if (kind == "divu") {
              auto op = self.create<hivmave::VFDivOp>(resTy, lhs, rhs, mask);
              // ODS generates setCast(hivm::TypeFn) for the enum attribute.
              op.setCast(hivm::TypeFn::cast_unsigned);
              result = op.getResult();
            } else {
              throw std::invalid_argument("unknown binary kind: " + kind);
            }
            return applySimdMode(self, result, mask, mode);
          },
          py::arg("kind"), py::arg("lhs"), py::arg("rhs"), py::arg("mask"),
          py::arg("mode") = "zeroing")
      // Elementwise vector-vector shift (``ave.hir.vshl`` / ``ave.hir.vshr``).
      // ``sign`` selects arithmetic vs logical right shift; ``mode`` is
      // consumed by applySimdMode.
      .def(
          "create_simd_shift",
          [](TLABuilder &self, std::string &kind, Value &lhs, Value &rhs,
             Value &mask, bool sign, std::string mode) -> Value {
            auto &builder = self.getBuilder();
            auto loc = self.getLastLoc();
            auto signVal = builder.create<arith::ConstantOp>(
                loc, builder.getBoolAttr(sign));
            auto resTy = lhs.getType();
            Value result;
            if (kind == "shl") {
              result = self.create<hivmave::VFShlOp>(resTy, lhs, rhs, mask,
                                                     signVal)
                           .getResult();
            } else if (kind == "shr") {
              result = self.create<hivmave::VFShrOp>(resTy, lhs, rhs, mask,
                                                     signVal)
                           .getResult();
            } else {
              throw std::invalid_argument("unknown shift kind: " + kind);
            }
            return applySimdMode(self, result, mask, mode);
          },
          py::arg("kind"), py::arg("lhs"), py::arg("rhs"), py::arg("mask"),
          py::arg("sign"), py::arg("mode") = "zeroing")
      // Elementwise vector-scalar left shift (``ave.hir.vshls``).  The scalar
      // must already have the vector element type; ``mode`` is consumed by
      // applySimdMode.
      .def(
          "create_simd_shift_scalar",
          [](TLABuilder &self, std::string &kind, Value &vec, Value &scalar,
             Value &mask, bool sign, std::string mode) -> Value {
            auto &builder = self.getBuilder();
            auto loc = self.getLastLoc();
            auto signVal = builder.create<arith::ConstantOp>(
                loc, builder.getBoolAttr(sign));
            auto resTy = vec.getType();
            if (kind != "shls") {
              throw std::invalid_argument("unknown scalar shift kind: " + kind);
            }
            auto result = self
                              .create<hivmave::VFShlsOp>(resTy, vec, scalar,
                                                         mask, signVal)
                              .getResult();
            return applySimdMode(self, result, mask, mode);
          },
          py::arg("kind"), py::arg("vec"), py::arg("scalar"),
          py::arg("mask"), py::arg("sign"), py::arg("mode") = "zeroing")
      // Elementwise vector-scalar (``ave.hir.vmaxs`` / ``ave.hir.vsmaxs`` /
      // ``ave.hir.vumaxs``): the scalar operand must have exactly the vector's
      // element type (AVE_VecScalarOp::TypesMatchWith), so the Python frontend
      // materializes it with create_simd_scalar[_int] first.
      .def(
          "create_simd_binary_scalar",
          [](TLABuilder &self, std::string &kind, Value &vec, Value &scalar,
             Value &mask, std::string mode) -> Value {
            auto resTy = vec.getType();
            Value result;
            if (kind == "adds") {
              result = self.create<hivmave::VFAddsOp>(resTy, vec, scalar, mask)
                           .getResult();
            } else if (kind == "muls") {
              result = self.create<hivmave::VFMulsOp>(resTy, vec, scalar, mask)
                           .getResult();
            } else if (kind == "maxs") {
              result = self.create<hivmave::VFMaxsOp>(resTy, vec, scalar, mask)
                           .getResult();
            } else if (kind == "umaxs") {
              result = self.create<hivmave::VMaxsUIOp>(resTy, vec, scalar, mask)
                           .getResult();
            } else if (kind == "mins") {
              result = self.create<hivmave::VFMinsOp>(resTy, vec, scalar, mask)
                           .getResult();
            } else if (kind == "umins") {
              result = self.create<hivmave::VMinsUIOp>(resTy, vec, scalar, mask)
                           .getResult();
            } else if (kind == "lrelus") {
              result = self.create<hivmave::VFLRelusOp>(resTy, vec, scalar, mask)
                           .getResult();
            } else {
              throw std::invalid_argument("unknown vector-scalar kind: " +
                                          kind);
            }
            return applySimdMode(self, result, mask, mode);
          },
          py::arg("kind"), py::arg("vec"), py::arg("scalar"), py::arg("mask"),
          py::arg("mode") = "zeroing")
      .def(
          "create_simd_vec_scalar",
          [](TLABuilder &self, std::string &kind, Value &vec, Value &scalar,
             Value &mask) -> Value {
            auto resTy = vec.getType();
            if (kind == "adds") {
              return self.create<hivmave::VFAddsOp>(resTy, vec, scalar, mask)
                  .getResult();
            }
            if (kind == "muls") {
              return self.create<hivmave::VFMulsOp>(resTy, vec, scalar, mask)
                  .getResult();
            }
            if (kind == "maxs") {
              return self.create<hivmave::VFMaxsOp>(resTy, vec, scalar, mask)
                  .getResult();
            }
            if (kind == "mins") {
              return self.create<hivmave::VFMinsOp>(resTy, vec, scalar, mask)
                  .getResult();
            }
            throw std::invalid_argument("unknown vector-scalar kind: " + kind);
          },
          py::arg("kind"), py::arg("vec"), py::arg("scalar"), py::arg("mask"))
      .def(
          "create_simd_vfma",
          [](TLABuilder &self, Value &acc, Value &lhs, Value &rhs, Value &mask,
             std::string mode) -> Value {
            auto resTy = acc.getType();
            // vmula %acc, %lhs, %rhs, %mask == acc + lhs * rhs
            auto op =
                self.create<hivmave::VFMulaOp>(resTy, acc, lhs, rhs, mask);
            return applySimdMode(self, op.getResult(), mask, mode);
          },
          py::arg("acc"), py::arg("lhs"), py::arg("rhs"), py::arg("mask"),
          py::arg("mode") = "zeroing")
      .def(
          "create_simd_preg_binary",
          [](TLABuilder &self, std::string &kind, Value &lhs, Value &rhs,
             Value &mask) -> Value {
            auto resTy = lhs.getType();
            auto &builder = self.getBuilder();
            auto mwAttr = hivmave::MaskWidthAttr::get(builder.getContext(),
                                                      hivmave::MaskWidth::B8);
            if (kind == "and")
              return self
                  .create<hivmave::PregAndOp>(resTy, mwAttr, lhs, rhs, mask)
                  .getResult();
            if (kind == "or")
              return self
                  .create<hivmave::PregOrOp>(resTy, mwAttr, lhs, rhs, mask)
                  .getResult();
            if (kind == "xor")
              return self
                  .create<hivmave::PregXorOp>(resTy, mwAttr, lhs, rhs, mask)
                  .getResult();
            throw std::invalid_argument("unknown preg binary kind: " + kind);
          },
          py::arg("kind"), py::arg("lhs"), py::arg("rhs"), py::arg("mask"))
      .def(
          "create_simd_preg_unary",
          [](TLABuilder &self, std::string &kind, Value &src,
             Value &mask) -> Value {
            auto resTy = src.getType();
            auto &builder = self.getBuilder();
            auto mwAttr = hivmave::MaskWidthAttr::get(builder.getContext(),
                                                      hivmave::MaskWidth::B8);
            if (kind == "not")
              return self.create<hivmave::PregNotOp>(resTy, mwAttr, src, mask)
                  .getResult();
            throw std::invalid_argument("unknown preg unary kind: " + kind);
          },
          py::arg("kind"), py::arg("src"), py::arg("mask"))
      .def(
          "create_simd_vmull",
          [](TLABuilder &self, Value &src1, Value &src2,
             Value &mask) -> py::tuple {
            auto ty = src1.getType();
            auto op = self.create<hivmave::VFVMULLOp>(ty, ty, src1, src2, mask);
            return py::make_tuple(op.getRes1(), op.getRes2());
          },
          py::arg("src1"), py::arg("src2"), py::arg("mask"))
      .def(
          "create_simd_unary",
          [](TLABuilder &self, std::string &kind, Value &src, Value &mask,
             std::string mode) -> Value {
            auto resTy = src.getType();
            Value result;
            if (kind == "abs") {
              result =
                  self.create<hivmave::VFAbsOp>(resTy, src, mask).getResult();
            } else if (kind == "neg") {
              result =
                  self.create<hivmave::VFNegOp>(resTy, src, mask).getResult();
            } else if (kind == "not") {
              result =
                  self.create<hivmave::VFNotOp>(resTy, src, mask).getResult();
            } else if (kind == "relu") {
              result =
                  self.create<hivmave::VFReluOp>(resTy, src, mask).getResult();
            } else if (kind == "sqrt") {
              result =
                  self.create<hivmave::VFSqrtOp>(resTy, src, mask).getResult();
            } else if (kind == "rsqrt") {
              result =
                  self.create<hivmave::VFRsqrtOp>(resTy, src, mask).getResult();
            } else if (kind == "exp") {
              result =
                  self.create<hivmave::VFExpOp>(resTy, src, mask).getResult();
            } else if (kind == "ln") {
              result =
                  self.create<hivmave::VFLnOp>(resTy, src, mask).getResult();
            } else {
              throw std::invalid_argument("unknown unary kind: " + kind);
            }
            return applySimdMode(self, result, mask, mode);
          },
          py::arg("kind"), py::arg("src"), py::arg("mask"),
          py::arg("mode") = "zeroing")
      .def(
          "create_simd_vreduce",
          [](TLABuilder &self, Value &vector, Value &mask,
             std::string &kind) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto resTy = vector.getType();
            auto kindEnum = hivmave::symbolizeCombiningKind(kind);
            if (!kindEnum) {
              throw std::invalid_argument("unknown reduction kind: " + kind);
            }
            auto op = self.create<hivmave::ReductionOp>(
                resTy, hivmave::CombiningKindAttr::get(ctx, *kindEnum), vector,
                mask);
            return op.getResult();
          },
          py::arg("vector"), py::arg("mask"), py::arg("kind"))
      .def(
          "create_simd_vci",
          [](TLABuilder &self, Value &index, py::object elem_tag, int32_t lanes,
             bool increase) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto elemTy = resolveSimdElementType(ctx, elem_tag);
            auto vecTy = VectorType::get({lanes}, elemTy);
            auto vciTy = increase ? hivmave::VCIType::DECREASE
                                  : hivmave::VCIType::INCREASE;
            auto op = self.create<hivmave::VFVCIOp>(
                vecTy, index, hivmave::VCITypeAttr::get(ctx, vciTy));
            return op.getResult();
          },
          py::arg("index"), py::arg("elem_tag"), py::arg("lanes"),
          py::arg("increase"))
      .def(
          "create_simd_vcmp",
          [](TLABuilder &self, std::string &cmp, Value &lhs, Value &rhs,
             Value &mask) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto cmpEnum = hivmave::symbolizeCmpType(cmp);
            if (!cmpEnum) {
              throw std::invalid_argument("unknown cmp type: " + cmp);
            }
            auto lhsTy = cast<VectorType>(lhs.getType());
            auto maskTy =
                VectorType::get(lhsTy.getShape(), IntegerType::get(ctx, 1));
            auto op = self.create<hivmave::VFCmpOp>(
                maskTy, hivmave::CmpTypeAttr::get(ctx, *cmpEnum), lhs, rhs,
                mask);
            return op.getResult();
          },
          py::arg("cmp"), py::arg("lhs"), py::arg("rhs"), py::arg("mask"))
      .def(
          "create_simd_vbroadcast",
          [](TLABuilder &self, Value &src, Value &mask, bool low,
             std::string mode) -> Value {
            auto &builder = self.getBuilder();
            auto resTy = src.getType();
            auto op = self.create<hivmave::VFBroadcastVectorOp>(
                resTy, src, mask, builder.getBoolAttr(low));
            return applySimdMode(self, op.getResult(), mask, mode);
          },
          py::arg("src"), py::arg("mask"), py::arg("low"),
          py::arg("mode") = "zeroing")
      .def(
          "create_simd_vmembar",
          [](TLABuilder &self, std::string &scope) {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto loc = self.getLastLoc();
            // AVE membar scope -> i32 (see HIVMMemBarOpLowering):
            // 0=VV_ALL 1=VST_VLD 2=VLD_VST 3=VST_VST 4=VS_ALL 5=VST_LD
            // 6=VLD_ST 7=VST_ST 8=SV_ALL 9=ST_VLD 10=LD_VST 11=ST_VST
            // 12=SS_ALL 13=ST_LD 14=LD_ST 15=ST_ST
            static const llvm::StringMap<int32_t> kMemBarScopes = {
                {"VV_ALL", 0},  {"VST_VLD", 1}, {"VLD_VST", 2}, {"VST_VST", 3},
                {"VS_ALL", 4},  {"VST_LD", 5},  {"VLD_ST", 6},  {"VST_ST", 7},
                {"SV_ALL", 8},  {"ST_VLD", 9},  {"LD_VST", 10}, {"ST_VST", 11},
                {"SS_ALL", 12}, {"ST_LD", 13},  {"LD_ST", 14},  {"ST_ST", 15},
            };
            auto it = kMemBarScopes.find(scope);
            if (it == kMemBarScopes.end()) {
              throw std::invalid_argument("unknown membar scope: " + scope);
            }
            auto src = builder.create<arith::ConstantOp>(
                loc, IntegerAttr::get(builder.getI32Type(), it->second));
            builder.create<hivmave::VFMemBarOp>(loc, src);
          },
          py::arg("scope"))
      .def(
          "create_simd_scalar",
          [](TLABuilder &self, py::object elem_tag, double value) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto loc = self.getLastLoc();
            auto elemTy = resolveSimdElementType(ctx, elem_tag);
            if (isa<IntegerType>(elemTy)) {
              return builder.create<arith::ConstantOp>(
                  loc, IntegerAttr::get(elemTy, static_cast<int64_t>(value)));
            }
            return builder.create<arith::ConstantOp>(
                loc, FloatAttr::get(elemTy, value));
          },
          py::arg("elem_tag"), py::arg("value"))
      .def(
          "create_simd_vsplat",
          [](TLABuilder &self, Value &scalar, py::object elem_tag,
             int32_t lanes) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto elemTy = resolveSimdElementType(ctx, elem_tag);
            auto vecTy = VectorType::get({lanes}, elemTy);
            auto op = self.create<hivmave::VFBroadcastScalarOp>(vecTy, scalar);
            return op.getResult();
          },
          py::arg("scalar"), py::arg("elem_tag"), py::arg("lanes"))
      // Exact integer scalar.  create_simd_scalar() takes a python double and
      // static_casts it, which silently rounds integers above 2**53; integer
      // consumers (vdup / vmaxs on i64) must route through this binding.
      .def(
          "create_simd_scalar_int",
          [](TLABuilder &self, py::object elem_tag, int64_t value) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto loc = self.getLastLoc();
            auto elemTy = resolveSimdElementType(ctx, elem_tag);
            if (!isa<IntegerType>(elemTy)) {
              throw std::invalid_argument(
                  "create_simd_scalar_int requires an integer element tag");
            }
            return builder.create<arith::ConstantOp>(
                loc, IntegerAttr::get(elemTy, value));
          },
          py::arg("elem_tag"), py::arg("value"))
      // Masked scalar broadcast (``ave.hir.broadcast``, lib
      // _mlir_ciface_vdups_<dtype>).  This is deliberately *not* vsplat():
      // scalar_broadcast ignores the predicate.
      .def(
          "create_simd_vdup",
          [](TLABuilder &self, Value &scalar, py::object elem_tag,
             int32_t lanes, Value &mask, std::string mode) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto elemTy = resolveSimdElementType(ctx, elem_tag);
            auto vecTy = VectorType::get({lanes}, elemTy);
            auto op = self.create<hivmave::VFBroadcastScalarMaskOp>(
                vecTy, scalar, mask);
            return applySimdMode(self, op.getResult(), mask, mode);
          },
          py::arg("scalar"), py::arg("elem_tag"), py::arg("lanes"),
          py::arg("mask"), py::arg("mode") = "zeroing")
      // Pack b16 -> b8 or b32 -> b16 into one half of the destination.
      // ``part`` is 0 for part_even and 1 for part_odd; the unwritten half is
      // left unspecified by a single call.
      .def(
          "create_simd_vpack",
          [](TLABuilder &self, Value &src, int32_t part,
             py::object out_elem_tag, int32_t out_lanes) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto outElemTy = resolveSimdElementType(ctx, out_elem_tag);
            auto outTy = VectorType::get({out_lanes}, outElemTy);
            auto op = self.create<hivmave::VFVpackOp>(
                outTy, src, builder.getI32IntegerAttr(part));
            return op.getResult();
          },
          py::arg("src"), py::arg("part"), py::arg("out_elem_tag"),
          py::arg("out_lanes"))
      .def(
          "create_simd_vunpack",
          [](TLABuilder &self, Value src, py::object elem,
             const std::string &part, const std::string &pp,
             bool isSigned) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto srcTy = cast<VectorType>(src.getType());
            auto elemTy = resolveSimdElementType(ctx, elem);
            auto outTy = VectorType::get(
                {2048 / elemTy.getIntOrFloatBitWidth()}, elemTy);
            auto partEnum = hivmave::symbolizeVCVT_PartType(part);
            auto ppEnum = hivmave::symbolizeVCVT_PPType(pp);
            if (!partEnum || !ppEnum)
              throw std::invalid_argument("invalid integer extension selector");
            // Extension predicates are indexed by source position. Apply the
            // user-supplied output mask with vsel in the Python frontend.
            auto maskTy = VectorType::get({srcTy.getNumElements()},
                                          builder.getI1Type());
            Value mask = self.create<hivmave::VFPgeOp>(
                maskTy, hivmave::PgePattern::ALL);
            bool factor4 = srcTy.getElementType().isInteger(8) &&
                           elemTy.isInteger(32);
            auto partAttr = factor4 ? hivmave::VCVT_PartTypeAttr{}
                                   : hivmave::VCVT_PartTypeAttr::get(ctx, *partEnum);
            auto ppAttr = factor4 ? hivmave::VCVT_PPTypeAttr::get(ctx, *ppEnum)
                                 : hivmave::VCVT_PPTypeAttr{};
            if (isSigned)
              return self.create<hivmave::VFExtSIOp>(
                  outTy, src, mask, partAttr, ppAttr).getResult();
            return self.create<hivmave::VFExtUIOp>(
                outTy, src, mask, partAttr, ppAttr).getResult();
          },
          py::arg("src"), py::arg("elem"), py::arg("part"), py::arg("pp"),
          py::arg("signed"))
      .def(
          "create_simd_vcvt",
          [](TLABuilder &self, std::string &kind, Value &src, Value &mask,
             py::object out_elem_tag, std::string &part, std::string &round,
             bool sat, std::string mode) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto outElemTy = resolveSimdElementType(ctx, out_elem_tag);
            auto partEnum = hivmave::symbolizeVCVT_PartType(part);
            if (!partEnum) {
              throw std::invalid_argument("unknown vcvt part: " + part);
            }
            auto partAttr = hivmave::VCVT_PartTypeAttr::get(ctx, *partEnum);
            auto roundEnum = hivm::symbolizeRoundMode(round);
            if (!roundEnum) {
              throw std::invalid_argument("unknown vcvt round mode: " + round);
            }
            auto roundAttr = hivm::RoundModeAttr::get(ctx, *roundEnum);
            auto srcTy = cast<VectorType>(src.getType());
            int64_t totalBits = srcTy.getNumElements() *
                                srcTy.getElementType().getIntOrFloatBitWidth();
            auto outVecTy = VectorType::get(
                {totalBits / outElemTy.getIntOrFloatBitWidth()}, outElemTy);
            Value result;
            if (kind == "truncf") {
              // vtruncf narrows keeping the total register bits: f32 (64) ->
              // bf16/f16 (128) -> f8 (256), lane count = total_bits / out_bits.
              result = self.create<hivmave::VFTruncFOp>(
                               outVecTy, src, mask, roundAttr,
                               builder.getBoolAttr(sat), partAttr)
                           .getResult();
            } else if (kind == "extf") {
              // vextf widens to a full register: 2048 / out_bits lanes.
              result =
                  self.create<hivmave::VFExtFOp>(outVecTy, src, mask, partAttr)
                      .getResult();
            } else if (kind == "s2f" || kind == "u2f") {
              // int -> fp conversion, sign chosen by the source dtype.
              // ISA attribute rules (VFSIntToFpOp/VFUIntToFpOp::verify):
              //   i8->f16, i16->f32, u8->f16 : part set, rnd unset
              //   i16->f16, i32->f32         : part unset, rnd set
              //   i64->f32, u64->f32         : part set, rnd set
              unsigned sb = srcTy.getElementType().getIntOrFloatBitWidth();
              bool outF16 = outElemTy.isF16();
              if (kind == "s2f") {
                auto op = self.create<hivmave::VFSIntToFpOp>(
                    outVecTy, src, mask, roundAttr, partAttr);
                bool needPart =
                    (sb == 8 && outF16) || (sb == 16 && !outF16) || sb == 64;
                bool needRnd = !needPart || sb == 64;
                if (!needPart)
                  op.removePartAttr();
                if (!needRnd)
                  op.removeRndAttr();
                result = op.getResult();
              } else {
                auto op = self.create<hivmave::VFUIntToFpOp>(
                    outVecTy, src, mask, roundAttr, partAttr);
                bool needRnd = sb == 64;
                if (!needRnd)
                  op.removeRndAttr();
                result = op.getResult();
              }
            } else if (kind == "f2s" || kind == "f2u") {
              // fp -> int conversion with rounding and saturation.
              // ISA attribute rules (VFFpToSIntOp::verify):
              //   f32->i32, f16->i16 : part unset, sat set
              //   f16->i32           : part set,   sat unset
              //   everything else    : part set,   sat set
              auto satAttr = builder.getBoolAttr(sat);
              if (kind == "f2s") {
                auto op = self.create<hivmave::VFFpToSIntOp>(
                    outVecTy, src, mask, roundAttr, satAttr, partAttr);
                unsigned sb = srcTy.getElementType().getIntOrFloatBitWidth();
                bool sameWidth = sb == outElemTy.getIntOrFloatBitWidth();
                if (sameWidth)
                  op.removePartAttr();
                else if (srcTy.getElementType().isF16() &&
                         outElemTy.isInteger(32))
                  op.removeSatAttr();
                result = op.getResult();
              } else {
                // vfptoui only supports f16 -> u8 (part/rnd/sat all set).
                result = self.create<hivmave::VFFpToUIntOp>(outVecTy, src, mask,
                                                            roundAttr, satAttr,
                                                            partAttr)
                             .getResult();
              }
            } else {
              throw std::invalid_argument("unknown vcvt kind: " + kind);
            }
            return applySimdMode(self, result, mask, mode);
          },
          py::arg("kind"), py::arg("src"), py::arg("mask"),
          py::arg("out_elem_tag"), py::arg("part"), py::arg("round") = "round",
          py::arg("sat") = true, py::arg("mode") = "zeroing")
      .def(
          "create_simd_pge",
          [](TLABuilder &self, std::string &pattern, int32_t lanes) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto maskTy = VectorType::get({lanes}, IntegerType::get(ctx, 1));
            auto patternEnum = hivmave::symbolizePgePattern(pattern);
            if (!patternEnum) {
              throw std::invalid_argument("unknown pge pattern: " + pattern);
            }
            auto op = self.create<hivmave::VFPgeOp>(maskTy, *patternEnum);
            return op.getResult();
          },
          py::arg("pattern"), py::arg("lanes"))
      .def(
          "create_simd_vsel",
          [](TLABuilder &self, Value &mask, Value &trueVal,
             Value &falseVal) -> Value {
            auto op = self.create<hivmave::VFSelectOp>(trueVal.getType(), mask,
                                                       trueVal, falseVal);
            return op.getResult();
          },
          py::arg("mask"), py::arg("true_val"), py::arg("false_val"))
      .def(
          "create_simd_vintlv",
          [](TLABuilder &self, Value &src1, Value &src2) -> py::tuple {
            auto resTy = src1.getType();
            auto op = self.create<hivmave::VFInterleaveOp>(
                resTy, resTy, src1, src2, hivmave::Layout_Change::SPARSE);
            return py::make_tuple(op.getResult(0), op.getResult(1));
          },
          py::arg("src1"), py::arg("src2"))
      .def(
          "create_simd_vdintlv",
          [](TLABuilder &self, Value &src1, Value &src2) -> py::tuple {
            auto resTy = src1.getType();
            auto op = self.create<hivmave::VFDeInterleaveOp>(
                resTy, resTy, src1, src2, hivmave::Layout_Change::SPARSE);
            return py::make_tuple(op.getResult(0), op.getResult(1));
          },
          py::arg("src1"), py::arg("src2"))
      .def(
          "create_simd_bitcast",
          [](TLABuilder &self, Value &src, py::object out_elem_tag,
             int32_t out_lanes) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto outVecTy = VectorType::get(
                {out_lanes}, resolveSimdElementType(ctx, out_elem_tag));
            auto op = self.create<vector::BitCastOp>(outVecTy, src);
            return op.getResult();
          },
          py::arg("src"), py::arg("out_elem_tag"), py::arg("out_lanes"))
      .def(
          "create_simd_vsstb",
          [](TLABuilder &self, Value &ptr, std::vector<Value> &indices,
             int64_t stride, Value &val, Value &mask) -> void {
            auto &builder = self.getBuilder();
            auto strideVal = builder.create<arith::ConstantOp>(
                self.getLastLoc(), builder.getIndexAttr(stride));
            auto indicesIdx = toIndexValues(self, indices);
            builder.create<hivmave::VFStoreWithStrideOp>(self.getLastLoc(), ptr,
                                                         ValueRange(indicesIdx),
                                                         strideVal, mask, val);
          },
          py::arg("ptr"), py::arg("indices"), py::arg("stride"), py::arg("val"),
          py::arg("mask"))
      .def("create_tla_buffer_type",
           [](TLABuilder &self, const std::vector<int64_t> &shape,
              Type elementType, const std::string &space,
              const std::vector<int64_t> &strides,
              std::optional<int64_t> offset) -> Type {
             auto memorySpace = hivm::symbolizeAddressSpace(space);
             if (!memorySpace) {
               // TLA frontend names -> HIVM #hivm.address_space mnemonics.
               static const llvm::StringMap<llvm::StringRef> kTlaToHivm = {
                   {"l1", "cbuf"}, {"l0a", "ca"},     {"l0b", "cb"},
                   {"l0c", "cc"},  {"bt", "biasbuf"}, {"fb", "fixbuf"}};
               auto it = kTlaToHivm.find(space);
               if (it != kTlaToHivm.end())
                 memorySpace = hivm::symbolizeAddressSpace(it->second);
             }
             if (!memorySpace)
               throw std::invalid_argument("unknown TLA address space: " +
                                           space);
             if (shape.size() != strides.size())
               throw std::invalid_argument(
                   "TLA buffer shape and strides must have the same rank");
             auto *ctx = self.getBuilder().getContext();
             // Match converted Triton pointer arguments. GM is inferred from
             // the argument/view chain; local buffers retain explicit spaces.
             return MemRefType::get(
                 shape, elementType,
                 tlaBufferLayout(ctx, shape, strides, offset),
                 *memorySpace == hivm::AddressSpace::GM
                     ? Attribute{}
                     : hivm::AddressSpaceAttr::get(ctx, *memorySpace));
           })
      .def("create_tla_alloc",
           [](TLABuilder &self, Type type) -> Value {
             auto memrefType = dyn_cast<MemRefType>(type);
             if (!memrefType || !memrefType.hasStaticShape())
               throw std::invalid_argument(
                   "TLA alloc requires a static memref");
             return self.create<memref::AllocOp>(memrefType).getResult();
           })
      // Scalar broadcast into one buffer: hivm.hir.vbrc with an empty
      // broadcast_dims array, which the verifier requires for scalar sources.
      // Only UB is enabled: InferCoreType maps UB to VECTOR and rejects other
      // spaces for this op.
      .def(
          "create_tla_fill",
          [](TLABuilder &self, Value dst, Value value) {
            auto dstType = dyn_cast<MemRefType>(dst.getType());
            if (!dstType)
              throw std::invalid_argument(
                  "TLA fill requires a memref destination");
            if (hivm::getOptionalHIVMAddressSpace(dstType) !=
                hivm::AddressSpace::UB)
              throw std::invalid_argument(
                  "TLA fill requires a UB destination buffer");
            if (isa<ShapedType>(value.getType()))
              throw std::invalid_argument("TLA fill requires a scalar value");
            if (value.getType() != dstType.getElementType())
              throw std::invalid_argument(
                  "TLA fill scalar type must match the destination element "
                  "type");
            self.create<hivm::VBrcOp>(TypeRange{}, value, dst);
          },
          py::arg("dst"), py::arg("value"))
      .def("create_tla_bind",
           [](TLABuilder &self, Value addr, Type type) -> Value {
             auto memrefType = dyn_cast<MemRefType>(type);
             if (!memrefType)
               throw std::invalid_argument("TLA bind requires a memref type");
             if (!addr.getType().isInteger(64))
               throw std::invalid_argument(
                   "TLA bind requires an i64 address operand");
             return self.create<hivm::PointerCastOp>(memrefType, addr)
                 .getResult();
           })
      .def("create_tla_pointer_view",
           [](TLABuilder &self, Value source, Value linear,
              Type resultType) -> Value {
             auto type = dyn_cast<MemRefType>(resultType);
             auto ptrType = dyn_cast<triton::PointerType>(source.getType());
             SmallVector<int64_t> strides;
             int64_t offset;
             if (!type || !ptrType || !type.hasStaticShape() ||
                 type.getRank() == 0 || type.getMemorySpace() ||
                 type.getElementType() != ptrType.getPointeeType() ||
                 failed(type.getStridesAndOffset(strides, offset)) ||
                 !ShapedType::isDynamic(offset))
               throw std::invalid_argument("invalid TLA GM view type");
             if (!linear.getType().isInteger(64))
               throw std::invalid_argument(
                   "TLA GM view requires an i64 element offset");
             // Put the full signed displacement in the scalar base. Block
             // offsets are i32 and their standard lowering clamps negatives.
             Value base =
                 self.create<triton::AddPtrOp>(source.getType(), source, linear);
             SmallVector<Value> shapeValues, strideValues, offsets;
             SmallVector<int32_t> order;
             auto &builder = self.getBuilder();
             for (int64_t i = 0; i < type.getRank(); ++i) {
               if (type.getDimSize(i) <= 0 || strides[i] <= 0)
                 throw std::invalid_argument(
                     "TLA GM view requires positive sizes and strides");
               shapeValues.push_back(
                   self.create<arith::ConstantIntOp>(type.getDimSize(i), 64));
               strideValues.push_back(
                   self.create<arith::ConstantIntOp>(strides[i], 64));
               offsets.push_back(self.create<arith::ConstantIntOp>(0, 32));
               order.push_back(type.getRank() - 1 - i);
             }
             auto blockType =
                 RankedTensorType::get(type.getShape(), type.getElementType());
             auto blockPtrType = triton::PointerType::get(
                 blockType, ptrType.getAddressSpace());
             Value block = self.create<triton::MakeTensorPtrOp>(
                 blockPtrType, base, shapeValues, strideValues, offsets,
                 builder.getDenseI32ArrayAttr(order));
             auto bridge = self.create<UnrealizedConversionCastOp>(
                 TypeRange{resultType}, ValueRange{block});
             bridge->setAttr("tla.gm_view", builder.getUnitAttr());
             return bridge.getResult(0);
           })
      .def("create_tla_subview",
           [](TLABuilder &self, Value source, const std::vector<Value> &offsets,
              const std::vector<int64_t> &sizes,
              const std::vector<int64_t> &strides) -> Value {
             auto type = dyn_cast<MemRefType>(source.getType());
             if (!type || offsets.size() != type.getRank() ||
                 sizes.size() != offsets.size() ||
                 strides.size() != offsets.size())
               throw std::invalid_argument(
                   "TLA subview operands must match memref rank");
             SmallVector<OpFoldResult> mixedOffsets, mixedSizes, mixedStrides;
             for (Value offset : offsets) {
               if (!offset.getType().isIndex())
                 offset = self.create<arith::IndexCastOp>(
                     self.getBuilder().getIndexType(), offset);
               mixedOffsets.push_back(offset);
             }
             for (auto size : sizes)
               mixedSizes.push_back(self.getBuilder().getIndexAttr(size));
             for (auto stride : strides)
               mixedStrides.push_back(self.getBuilder().getIndexAttr(stride));
             return self
                 .create<memref::SubViewOp>(source, mixedOffsets, mixedSizes,
                                            mixedStrides)
                 .getResult();
           })
      .def("create_tla_reinterpret_cast",
           [](TLABuilder &self, Value source, Value offset,
              const std::vector<int64_t> &sizes,
              const std::vector<int64_t> &strides, Type resultType) -> Value {
             if (!isa<MemRefType>(source.getType()) ||
                 sizes.size() != strides.size())
               throw std::invalid_argument(
                   "TLA reinterpret_cast requires a memref source and matching "
                   "sizes/strides");
             if (!offset.getType().isIndex())
               offset = self.create<arith::IndexCastOp>(
                   self.getBuilder().getIndexType(), offset);
             // resultType is strided<strides, offset: ?>: sizes/strides are
             // static, only the linear offset is dynamic.
             return self
                 .create<memref::ReinterpretCastOp>(
                     resultType, source, ValueRange{offset}, ValueRange{},
                     ValueRange{},
                     self.getBuilder().getDenseI64ArrayAttr(
                         {ShapedType::kDynamic}),
                     self.getBuilder().getDenseI64ArrayAttr(sizes),
                     self.getBuilder().getDenseI64ArrayAttr(strides))
                 .getResult();
           })
      // memref.collapse_shape / expand_shape materialization (see
      // createReshapeOp above for the reassociation plumbing).
      .def("create_tla_collapse_shape",
           &createReshapeOp<memref::CollapseShapeOp>, py::arg("source"),
           py::arg("result_type"), py::arg("reassociation"))
      .def("create_tla_expand_shape", &createReshapeOp<memref::ExpandShapeOp>,
           py::arg("source"), py::arg("result_type"), py::arg("reassociation"))
      // Each explicit DMA binding materializes one native HIVM operation.
      .def(
          "create_tla_load",
          [](TLABuilder &self, Value src, Value dst, const py::dict &attrs,
             std::optional<Value> padValue, std::optional<Value> leftPadding,
             std::optional<Value> rightPadding) {
            checkDmaBuffers(src, dst);
            auto op = self.create<hivm::LoadOp>(
                TypeRange{}, src, dst, hivm::PadModeAttr{},
                padValue.value_or(Value{}),
                dmaIndexOperand(self, leftPadding.value_or(Value{})),
                dmaIndexOperand(self, rightPadding.value_or(Value{})));
            setDmaAttrs(op, attrs);
          },
          py::arg("src"), py::arg("dst"), py::arg("attrs"),
          py::arg("pad_value") = std::nullopt,
          py::arg("left_padding_num") = std::nullopt,
          py::arg("right_padding_num") = std::nullopt)
      .def(
          "create_tla_store",
          [](TLABuilder &self, Value src, Value dst, const py::dict &attrs) {
            checkDmaBuffers(src, dst);
            auto op = self.create<hivm::StoreOp>(TypeRange{}, src, dst);
            setDmaAttrs(op, attrs);
          },
          py::arg("src"), py::arg("dst"), py::arg("attrs"))
      .def(
          "create_tla_nd2nz",
          [](TLABuilder &self, Value src, Value dst, const py::dict &attrs) {
            checkDmaBuffers(src, dst);
            auto op = self.create<hivm::ND2NZOp>(
                TypeRange{}, src, dst, self.getBuilder().getUnitAttr());
            setDmaAttrs(op, attrs);
          },
          py::arg("src"), py::arg("dst"), py::arg("attrs"))
      .def(
          "create_tla_copy",
          [](TLABuilder &self, Value src, Value dst, const py::dict &attrs,
             std::optional<Value> padValue) {
            checkDmaBuffers(src, dst);
            auto op = self.create<hivm::CopyOp>(TypeRange{}, src, dst);
            if (padValue)
              op.getPadValueMutable().assign(*padValue);
            setDmaAttrs(op, attrs);
          },
          py::arg("src"), py::arg("dst"), py::arg("attrs"),
          py::arg("pad_value") = std::nullopt)
      .def("create_tla_l12ub",
           [](TLABuilder &self, Value src, Value dst) {
             checkDmaBuffers(src, dst);
             self.create<hivm::L12UBOp>(TypeRange{}, src, dst);
           })
      .def("create_tla_l12bt",
           [](TLABuilder &self, Value src, Value dst, Value n) {
             checkDmaBuffers(src, dst);
             self.create<hivm::L12BTOp>(TypeRange{}, src, dst,
                                        dmaIndexOperand(self, n));
           })
      .def(
          "create_tla_l12l0",
          [](TLABuilder &self, Value src, Value dst, int64_t kPartIdx,
             int64_t kPart, int64_t kPartCeil, int64_t kPartLoop,
             int64_t kPartActual, int64_t m, bool isTransposed) {
            checkDmaBuffers(src, dst);
            auto &builder = self.getBuilder();
            self.create<hivm::L12L0Op>(TypeRange{}, src, dst,
                                       builder.getI64IntegerAttr(kPartIdx),
                                       builder.getI64IntegerAttr(kPart),
                                       builder.getI64IntegerAttr(kPartCeil),
                                       builder.getI64IntegerAttr(kPartLoop),
                                       builder.getI64IntegerAttr(kPartActual),
                                       builder.getI64IntegerAttr(m),
                                       builder.getBoolAttr(isTransposed));
          },
          py::arg("src"), py::arg("dst"), py::arg("k_part_idx"),
          py::arg("k_part"), py::arg("k_part_ceil"), py::arg("k_part_loop"),
          py::arg("k_part_actual"), py::arg("m"), py::arg("is_transposed"))
      .def(
          "create_tla_fixpipe",
          [](TLABuilder &self, Value src, Value dst, const py::dict &attrs,
             std::optional<Value> quantScale,
             std::optional<int32_t> unitFlagCtrl,
             std::optional<int64_t> unitFlagGroupId) {
            checkDmaBuffers(src, dst);
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto op = self.create<hivm::FixpipeOp>(
                TypeRange{}, src, dst,
                hivm::FixpipeDMAModeAttr::get(ctx, hivm::FixpipeDMAMode::NZ2NZ),
                hivm::FixpipeDualDstModeAttr{}, hivm::FixpipeSubBlockAttr{},
                hivm::FixpipePreQuantModeAttr::get(
                    ctx, hivm::FixpipePreQuantMode::NO_QUANT),
                hivm::FixpipePreReluModeAttr::get(
                    ctx, hivm::FixpipePreReluMode::NO_RELU),
                builder.getBoolAttr(false), builder.getBoolAttr(true),
                quantScale.value_or(Value{}));
            setDmaAttrs(op, attrs);
            setUnitFlagAttrs(builder, op, unitFlagCtrl, unitFlagGroupId);
          },
          py::arg("src"), py::arg("dst"), py::arg("attrs"),
          py::arg("quant_scale") = std::nullopt,
          py::arg("unit_flag_ctrl") = std::nullopt,
          py::arg("unit_flag_group_id") = std::nullopt)
      .def("create_tla_get_sub_block_idx",
           [](TLABuilder &self) -> Value {
             return self
                 .create<hivm::GetSubBlockIdxOp>(self.getBuilder().getI64Type())
                 .getResult();
           })
      .def("create_tla_set_ctrl",
           [](TLABuilder &self, bool enable, int64_t idx) {
             self.create<hivm::SetCtrlOp>(
                 self.getBuilder().getBoolAttr(enable),
                 self.getBuilder().getI64IntegerAttr(idx));
           })
      .def("create_tla_set_atomic",
           [](TLABuilder &self, const std::string &kind, Type type) {
             hivm::AtomicKind atomicKind;
             if (kind == "none")
               atomicKind = hivm::AtomicKind::NONE;
             else if (kind == "add")
               atomicKind = hivm::AtomicKind::ADD;
             else if (kind == "max")
               atomicKind = hivm::AtomicKind::MAX;
             else if (kind == "min")
               atomicKind = hivm::AtomicKind::MIN;
             else
               throw std::invalid_argument("unsupported TLA atomic kind: " + kind);
             auto *ctx = self.getBuilder().getContext();
             self.create<hivm::SetAtomicOp>(
                 hivm::AtomicKindAttr::get(ctx, atomicKind),
                 TypeAttr::get(type));
           },
           py::arg("kind"), py::arg("type"))
      .def("create_hivm_load",
           [](TLABuilder &self, Value src, Value dst) {
             checkDmaBuffers(src, dst);
             self.create<hivm::LoadOp>(TypeRange{}, src, dst);
           })
      .def("create_hivm_store",
           [](TLABuilder &self, Value src, Value dst) {
             checkDmaBuffers(src, dst);
             self.create<hivm::StoreOp>(TypeRange{}, src, dst);
           })
      // Print at the buffer stage: hivm.hir.debug accepts memref/scalar
      // operands directly, no tt.print detour.
      .def(
          "create_hivm_debug",
          [](TLABuilder &self, std::string debugtype, std::string prefix,
             bool hex, Value value) {
            if (!value)
              throw std::invalid_argument("TLA debug requires a value operand");
            self.create<hivm::DebugOp>(debugtype, prefix, hex, value);
          },
          py::arg("debugtype"), py::arg("prefix"), py::arg("hex"),
          py::arg("value"))
      // GM(ND) -> L1(NZ): dst_continuous is required by the mte2 library-call
      // path that nd2nz lowers into.
      .def(
          "create_hivm_nd2nz",
          [](TLABuilder &self, Value src, Value dst) {
            checkDmaBuffers(src, dst);
            self.create<hivm::ND2NZOp>(TypeRange{}, src, dst,
                                       self.getBuilder().getUnitAttr());
          },
          py::arg("src"), py::arg("dst"))
      // L0C -> GM/UB. The base create_fixpipe (ascend_ir.cc) forces a
      // RankedTensorType src; TLA buffers are memrefs, so route through the
      // FixpipeOp memref builder (dma_mode: "nz2nd"/"normal").
      .def(
          "create_hivm_fixpipe",
          [](TLABuilder &self, Value src, Value dst, std::string &dma_mode) {
            checkDmaBuffers(src, dst);
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto dmaEnum = hivm::symbolizeFixpipeDMAMode(dma_mode);
            if (!dmaEnum)
              throw std::invalid_argument("unknown fixpipe dma_mode: " +
                                          dma_mode);
            auto dmaAttr = hivm::FixpipeDMAModeAttr::get(ctx, *dmaEnum);
            auto dualAttr = hivm::FixpipeDualDstModeAttr::get(
                ctx, hivm::FixpipeDualDstMode::NO_DUAL);
            auto quantAttr = hivm::FixpipePreQuantModeAttr::get(
                ctx, hivm::FixpipePreQuantMode::NO_QUANT);
            auto reluAttr = hivm::FixpipePreReluModeAttr::get(
                ctx, hivm::FixpipePreReluMode::NO_RELU);
            self.create<hivm::FixpipeOp>(TypeRange{}, src, dst, dmaAttr,
                                         dualAttr, nullptr, quantAttr, reluAttr,
                                         builder.getBoolAttr(false));
          },
          py::arg("src"), py::arg("dst"), py::arg("dma_mode"))
      // C = A x B (+C): A=(M,K)/B=(K,N) in L1, C in L0C (f32/i32).
      // init_condition clears L0C first (clear_accum), real_m/k/n are the
      // actual tile sizes (arith.constant index); transposes set UnitAttrs.
      .def(
          "create_hivm_mmadl1",
          [](TLABuilder &self, Value a, Value b, Value c, bool init_condition,
             int64_t real_m, int64_t real_k, int64_t real_n, bool transposeA,
             bool transposeB, std::optional<int32_t> unitFlagCtrl,
             std::optional<int64_t> unitFlagGroupId) {
            auto &builder = self.getBuilder();
            auto loc = self.getLastLoc();
            auto init = builder.create<arith::ConstantOp>(
                loc, builder.getI1Type(), builder.getBoolAttr(init_condition));
            auto m = builder.create<arith::ConstantIndexOp>(loc, real_m);
            auto k = builder.create<arith::ConstantIndexOp>(loc, real_k);
            auto n = builder.create<arith::ConstantIndexOp>(loc, real_n);
            auto aTrans = transposeA ? builder.getUnitAttr() : UnitAttr{};
            auto bTrans = transposeB ? builder.getUnitAttr() : UnitAttr{};
            auto op = self.create<hivm::MmadL1Op>(
                TypeRange{}, a, b, init, m, k, n, c,
                /*per_channel_bias=*/Value{}, /*a_transpose=*/aTrans,
                /*b_transpose=*/bTrans, /*enable_HF32=*/UnitAttr{},
                /*enable_I4=*/UnitAttr{});
            setUnitFlagAttrs(builder, op, unitFlagCtrl, unitFlagGroupId);
          },
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("init_condition"),
          py::arg("real_m"), py::arg("real_k"), py::arg("real_n"),
          py::arg("transpose_a") = false, py::arg("transpose_b") = false,
          py::arg("unit_flag_ctrl") = std::nullopt,
          py::arg("unit_flag_group_id") = std::nullopt)
      // C = MX(A, ScaleA) x MX(B, ScaleB) (+bias): A/B and the E8M0 scales are
      // in L1, C is an L0C accumulator. lhs_format/rhs_format are the native
      // HIVMMatmulDataformat codes (1=E5M2, 2=E4M3, 3=E2M1) and are only set
      // for packed integer operands; FP8 element types encode the format.
      .def(
          "create_hivm_mmadmx",
          [](TLABuilder &self, Value a, Value b, Value scaleA, Value scaleB,
             Value c, bool init_condition, int64_t real_m, int64_t real_k,
             int64_t real_n, std::optional<int32_t> lhsFormat,
             std::optional<int32_t> rhsFormat, bool transposeA,
             bool transposeB, std::optional<Value> perChannelBias) {
            auto &builder = self.getBuilder();
            auto loc = self.getLastLoc();
            auto init = builder.create<arith::ConstantOp>(
                loc, builder.getI1Type(), builder.getBoolAttr(init_condition));
            auto m = builder.create<arith::ConstantIndexOp>(loc, real_m);
            auto k = builder.create<arith::ConstantIndexOp>(loc, real_k);
            auto n = builder.create<arith::ConstantIndexOp>(loc, real_n);
            auto aTrans = transposeA ? builder.getUnitAttr() : UnitAttr{};
            auto bTrans = transposeB ? builder.getUnitAttr() : UnitAttr{};
            auto lhsAttr = lhsFormat
                               ? builder.getI32IntegerAttr(*lhsFormat)
                               : IntegerAttr{};
            auto rhsAttr = rhsFormat
                               ? builder.getI32IntegerAttr(*rhsFormat)
                               : IntegerAttr{};
            self.create<hivm::MmadMxL1Op>(
                TypeRange{}, a, b, scaleA, scaleB, init, m, k, n, c,
                /*lhsFormat=*/lhsAttr, /*rhsFormat=*/rhsAttr,
                /*a_transpose=*/aTrans, /*b_transpose=*/bTrans,
                /*per_channel_bias=*/perChannelBias.value_or(Value{}));
          },
          py::arg("a"), py::arg("b"), py::arg("scale_a"), py::arg("scale_b"),
          py::arg("c"), py::arg("init_condition"), py::arg("real_m"),
          py::arg("real_k"), py::arg("real_n"),
          py::arg("lhs_format") = std::nullopt,
          py::arg("rhs_format") = std::nullopt,
          py::arg("transpose_a") = false, py::arg("transpose_b") = false,
          py::arg("per_channel_bias") = std::nullopt)
      // GM -> L1 MX scale load (hivm.hir.load_scale). The source is a 2-D ND
      // E8M0 tensor and the destination the 4-D zZ/nN fractal buffer.
      .def(
          "create_hivm_load_mx_scale",
          [](TLABuilder &self, Value src, Value dst, bool isTransposed) {
            checkDmaBuffers(src, dst);
            self.create<hivm::LoadMXScaleOp>(TypeRange{}, src, dst,
                                             isTransposed);
          },
          py::arg("src"), py::arg("dst"), py::arg("is_transposed") = false)
      .def("create_scope_op",
           [](TLABuilder &self, py::dict &scopeAttrs,
              std::vector<Type> resultTypes) -> OpState {
             llvm::SmallVector<NamedAttribute> attrs;
             for (auto item : scopeAttrs) {
               std::string key = py::cast<std::string>(item.first);
               Attribute value = py::cast<Attribute>(item.second);
               attrs.push_back(
                   NamedAttribute(self.getBuilder().getStringAttr(key), value));
             }
             auto scopeOp = self.create<scope::ScopeOp>(TypeRange(resultTypes));
             scopeOp->setAttrs(attrs);
             return OpState(scopeOp);
           })
      .def("create_tla_pipeline",
           [](TLABuilder &self, Value stop) {
             auto &builder = self.getBuilder();
             if (!stop.getType().isInteger(32) && !stop.getType().isInteger(64))
               throw py::value_error("pipeline stop must be i32 or i64");
             Value lb = self.create<arith::ConstantIndexOp>(0);
             Value ub =
                 self.create<arith::IndexCastOp>(builder.getIndexType(), stop);
             Value step = self.create<arith::ConstantIndexOp>(1);
             auto domain = self.create<scope::ScopeOp>(TypeRange{});
             domain->setAttr("hivm.pipeline", builder.getUnitAttr());
             Block *body = builder.createBlock(&domain.getRegion());
             auto loop = self.create<scf::ForOp>(lb, ub, step);
             builder.setInsertionPointToEnd(body);
             self.create<scope::ReturnOp>(ValueRange{});
             // Python emits the body and its yield after scope/carry checks.
             loop.getBody()->getTerminator()->erase();
             builder.setInsertionPointToStart(loop.getBody());
             Value iv = self.create<arith::IndexCastOp>(builder.getI64Type(),
                                                        loop.getInductionVar());
             return std::make_pair(loop, iv);
           })
      // Ensures the TLA scope has its entry block; the cann-shared
      // create_scope_op makes its own, so the emplace stays here to avoid
      // two blocks in non-TLA scopes.
      // pybind11 wraps Block in unique_ptr; reference keeps it alive.
      .def(
          "get_scope_entry_block",
          [](TLABuilder &self, OpState op) -> Block
                                               * {
                                                 Region &region =
                                                     op->getRegion(0);
                                                 if (region.empty())
                                                   region.emplaceBlock();
                                                 return &region.front();
                                               },
          py::return_value_policy::reference)
      // Set an attribute on an existing operation. module.get_function is not
      // usable for the ttadapter entry: triton_to_linalg lowers it from
      // triton::FuncOp to mlir func::FuncOp, so lookupSymbol<FuncOp> misses it
      // (module.walk returns the operation itself).
      .def(
          "set_op_attr",
          [](TLABuilder &self, Operation *op, std::string &name,
             Attribute &attr) -> void { op->setAttr(name, attr); },
          py::arg("op"), py::arg("name"), py::arg("attr"));

  // Minimal func::FuncOp exposure for frontends that outline functions.
  py::class_<func::FuncOp, OpState>(m, "func_op", py::module_local())
      .def("args",
           [](func::FuncOp &self, unsigned idx) -> BlockArgument {
             if (idx >= self.getNumArguments())
               throw pybind11::index_error(
                   "Function argument index out of range");
             return self.getArgument(idx);
           })
      .def("get_num_args", &func::FuncOp::getNumArguments)
      .def(
          "add_entry_block",
          [](func::FuncOp &self) -> Block * { return self.addEntryBlock(); },
          py::return_value_policy::reference);
}
