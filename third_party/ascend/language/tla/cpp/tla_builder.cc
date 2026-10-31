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

#include "pybind11/pybind11.h"
#include <pybind11/stl.h>

#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/HIVMAVE/IR/HIVMAVE.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Utils/ReshapeOpsUtils.h"
#include "mlir/Dialect/Vector/IR/VectorOps.h"
#include "mlir/IR/Block.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/Region.h"
#include "mlir/IR/Types.h"
#include "mlir/Support/LLVM.h"
#include "llvm/ADT/StringMap.h"

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
  throw std::invalid_argument("unknown simd element tag \"" + tag +
                              "\"; expected i8 / bf16 / f32 / fp8e4nv");
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

// Resolve a frontend pipe name ("PIPE_MTE2") to its HIVM attribute.
hivm::PipeAttr getSyncPipe(OpBuilder &builder, const std::string &pipe) {
  auto pipeEnum = hivm::symbolizePIPE(pipe);
  if (!pipeEnum)
    throw std::invalid_argument("unknown sync pipe: " + pipe);
  return hivm::PipeAttr::get(builder.getContext(), *pipeEnum);
}

// Validate a frontend constant against its hardware range [0, upper].
int64_t checkSyncRange(int64_t value, int64_t upper, const char *what) {
  if (value < 0 || value > upper)
    throw std::invalid_argument(std::string(what) + " must be in [0, " +
                                std::to_string(upper) + "]");
  return value;
}

// Cross-core set/wait are block-level synchronization: emit them with the
// block instruction mode requested by the frontend and the core type of the
// enclosing TLA scope.
void createBlockSync(TLABuilder &self, const std::string &pipe, int64_t flag,
                     hivm::SyncBlockInstrMode instrMode, bool isSet) {
  auto &builder = self.getBuilder();
  auto *ctx = builder.getContext();
  auto coreAttr = hivm::TCoreTypeAttr::get(ctx, enclosingCoreType(builder));
  auto pipeAttr = getSyncPipe(builder, pipe);
  auto flagAttr = builder.getI64IntegerAttr(checkSyncRange(flag, 15, "flag"));
  auto modeAttr = hivm::SyncBlockInstrModeAttr::get(ctx, instrMode);
  if (isSet)
    self.create<hivm::SyncBlockSetOp>(coreAttr, pipeAttr, pipeAttr, flagAttr,
                                      Value{}, modeAttr);
  else
    self.create<hivm::SyncBlockWaitOp>(coreAttr, pipeAttr, pipeAttr, flagAttr,
                                       modeAttr);
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

} // namespace

void init_tla_builder(py::module &&m) {
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
                getSyncPipe(self.getBuilder(), pipe));
          },
          py::arg("pipe"))
      .def(
          "create_tla_set_flag",
          [](TLABuilder &self, const std::string &pipe,
             const std::string &waitPipe, int64_t eventId) {
            auto *ctx = self.getBuilder().getContext();
            self.create<hivm::SetFlagOp>(
                getSyncPipe(self.getBuilder(), pipe),
                getSyncPipe(self.getBuilder(), waitPipe),
                hivm::EventAttr::get(
                    ctx, static_cast<hivm::EVENT>(
                             checkSyncRange(eventId, 7, "event_id"))),
                Value{});
          },
          py::arg("pipe"), py::arg("wait_pipe"), py::arg("event_id"))
      .def(
          "create_tla_wait_flag",
          [](TLABuilder &self, const std::string &pipe,
             const std::string &waitPipe, int64_t eventId) {
            auto *ctx = self.getBuilder().getContext();
            self.create<hivm::WaitFlagOp>(
                getSyncPipe(self.getBuilder(), pipe),
                getSyncPipe(self.getBuilder(), waitPipe),
                hivm::EventAttr::get(
                    ctx, static_cast<hivm::EVENT>(
                             checkSyncRange(eventId, 7, "event_id"))),
                Value{});
          },
          py::arg("pipe"), py::arg("wait_pipe"), py::arg("event_id"))
      // Hardware global barrier: AIV-only (false) or all AIC/AIV (true).
      .def(
          "create_tla_sync_all",
          [](TLABuilder &self, bool allCores) {
            auto *ctx = self.getBuilder().getContext();
            auto pipeAll = hivm::PipeAttr::get(ctx, hivm::PIPE::PIPE_ALL);
            auto syncMode = hivm::SyncBlockModeAttr::get(
                ctx, allCores ? hivm::SyncBlockMode::ALL
                              : hivm::SyncBlockMode::ALL_VECTOR);
            self.create<hivm::SyncBlockOp>(
                syncMode, IntegerAttr{}, Value{},
                allCores ? pipeAll : hivm::PipeAttr{}, pipeAll);
          },
          py::arg("all_cores"))
      // Cross-core FFTS flags: mode 0 inter-block / 1 inter-subblock /
      // 2 intra-block. The wait has no mode and always waits on the flag.
      .def(
          "create_tla_block_set_flag",
          [](TLABuilder &self, int mode, const std::string &pipe,
             int64_t flag) {
            createBlockSync(self, pipe, flag,
                            static_cast<hivm::SyncBlockInstrMode>(
                                checkSyncRange(mode, 2, "sync mode")),
                            /*isSet=*/true);
          },
          py::arg("mode"), py::arg("pipe"), py::arg("flag"))
      .def(
          "create_tla_block_wait_flag",
          [](TLABuilder &self, const std::string &pipe, int64_t flag) {
            createBlockSync(
                self, pipe, flag,
                hivm::SyncBlockInstrMode::INTER_BLOCK_SYNCHRONIZATION,
                /*isSet=*/false);
          },
          py::arg("pipe"), py::arg("flag"))
      // SIMD vector-register ops (AVE dialect, ave.hir.*).
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
             Value &mask) -> Value {
            auto resTy = lhs.getType();
            if (kind == "mul") {
              return self.create<hivmave::VFMulOp>(resTy, lhs, rhs, mask)
                  .getResult();
            }
            if (kind == "add") {
              return self.create<hivmave::VFAddOp>(resTy, lhs, rhs, mask)
                  .getResult();
            }
            if (kind == "div") {
              return self.create<hivmave::VFDivOp>(resTy, lhs, rhs, mask)
                  .getResult();
            }
            if (kind == "xor") {
              return self.create<hivmave::VFXorOp>(resTy, lhs, rhs, mask)
                  .getResult();
            }
            if (kind == "mod") {
              return self.create<hivmave::VFModOp>(resTy, lhs, rhs, mask)
                  .getResult();
            }
            if (kind == "max") {
              return self.create<hivmave::VFMaxOp>(resTy, lhs, rhs, mask)
                  .getResult();
            }
            if (kind == "min") {
              return self.create<hivmave::VFMinOp>(resTy, lhs, rhs, mask)
                  .getResult();
            }
            throw std::invalid_argument("unknown binary kind: " + kind);
          },
          py::arg("kind"), py::arg("lhs"), py::arg("rhs"), py::arg("mask"))
      .def(
          "create_simd_vfma",
          [](TLABuilder &self, Value &acc, Value &lhs, Value &rhs,
             Value &mask) -> Value {
            auto resTy = acc.getType();
            // vmula %acc, %lhs, %rhs, %mask == acc + lhs * rhs
            auto op =
                self.create<hivmave::VFMulaOp>(resTy, acc, lhs, rhs, mask);
            return op.getResult();
          },
          py::arg("acc"), py::arg("lhs"), py::arg("rhs"), py::arg("mask"))
      .def(
          "create_simd_unary",
          [](TLABuilder &self, std::string &kind, Value &src,
             Value &mask) -> Value {
            auto resTy = src.getType();
            if (kind == "neg") {
              return self.create<hivmave::VFNegOp>(resTy, src, mask)
                  .getResult();
            }
            if (kind == "sqrt") {
              return self.create<hivmave::VFSqrtOp>(resTy, src, mask)
                  .getResult();
            }
            if (kind == "rsqrt") {
              return self.create<hivmave::VFRsqrtOp>(resTy, src, mask)
                  .getResult();
            }
            if (kind == "exp") {
              return self.create<hivmave::VFExpOp>(resTy, src, mask)
                  .getResult();
            }
            throw std::invalid_argument("unknown unary kind: " + kind);
          },
          py::arg("kind"), py::arg("src"), py::arg("mask"))
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
          [](TLABuilder &self, Value &src, Value &mask, bool low) -> Value {
            auto &builder = self.getBuilder();
            auto resTy = src.getType();
            auto op = self.create<hivmave::VFBroadcastVectorOp>(
                resTy, src, mask, builder.getBoolAttr(low));
            return op.getResult();
          },
          py::arg("src"), py::arg("mask"), py::arg("low"))
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
      .def(
          "create_simd_vcvt",
          [](TLABuilder &self, std::string &kind, Value &src, Value &mask,
             py::object out_elem_tag, std::string &part) -> Value {
            auto &builder = self.getBuilder();
            auto *ctx = builder.getContext();
            auto outElemTy = resolveSimdElementType(ctx, out_elem_tag);
            auto partEnum = hivmave::symbolizeVCVT_PartType(part);
            if (!partEnum) {
              throw std::invalid_argument("unknown vcvt part: " + part);
            }
            auto partAttr = hivmave::VCVT_PartTypeAttr::get(ctx, *partEnum);
            auto srcTy = cast<VectorType>(src.getType());
            if (kind == "truncf") {
              // vtruncf narrows keeping the total register bits: f32 (64) ->
              // bf16/f16 (128) -> f8 (256), lane count = total_bits / out_bits.
              int64_t totalBits =
                  srcTy.getNumElements() *
                  srcTy.getElementType().getIntOrFloatBitWidth();
              auto outVecTy = VectorType::get(
                  {totalBits / outElemTy.getIntOrFloatBitWidth()}, outElemTy);
              return self
                  .create<hivmave::VFTruncFOp>(
                      outVecTy, src, mask,
                      hivm::RoundModeAttr::get(ctx, hivm::RoundMode::ROUND),
                      builder.getBoolAttr(true), partAttr)
                  .getResult();
            }
            // vextf widens to a full register: 2048 / out_bits lanes.
            auto outVecTy = VectorType::get(
                {2048 / outElemTy.getIntOrFloatBitWidth()}, outElemTy);
            return self.create<hivmave::VFExtFOp>(outVecTy, src, mask, partAttr)
                .getResult();
          },
          py::arg("kind"), py::arg("src"), py::arg("mask"),
          py::arg("out_elem_tag"), py::arg("part"))
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
      .def("create_tla_buffer_type",
           [](TLABuilder &self, const std::vector<int64_t> &shape,
              Type elementType, const std::string &space,
              const std::vector<int64_t> &strides,
              std::optional<int64_t> offset) -> Type {
             auto memorySpace = hivm::symbolizeAddressSpace(space);
             if (!memorySpace) {
               // TLA frontend names -> HIVM #hivm.address_space mnemonics.
               static const llvm::StringMap<llvm::StringRef> kTlaToHivm = {
                   {"l1", "cbuf"}, {"l0a", "ca"}, {"l0b", "cb"}, {"l0c", "cc"}};
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
             return MemRefType::get(
                 shape, elementType,
                 tlaBufferLayout(ctx, shape, strides, offset),
                 hivm::AddressSpaceAttr::get(ctx, *memorySpace));
           })
      .def("create_tla_alloc",
           [](TLABuilder &self, Type type) -> Value {
             auto memrefType = dyn_cast<MemRefType>(type);
             if (!memrefType || !memrefType.hasStaticShape())
               throw std::invalid_argument(
                   "TLA alloc requires a static memref");
             return self.create<memref::AllocOp>(memrefType).getResult();
           })
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
      .def("create_hivm_load",
           [](TLABuilder &self, Value src, Value dst) {
             if (!isa<MemRefType>(src.getType()) ||
                 !isa<MemRefType>(dst.getType()))
               throw std::invalid_argument("TLA DMA requires memref operands");
             self.create<hivm::LoadOp>(TypeRange{}, src, dst);
           })
      .def("create_hivm_store",
           [](TLABuilder &self, Value src, Value dst) {
             if (!isa<MemRefType>(src.getType()) ||
                 !isa<MemRefType>(dst.getType()))
               throw std::invalid_argument("TLA DMA requires memref operands");
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
            if (!isa<MemRefType>(src.getType()) ||
                !isa<MemRefType>(dst.getType()))
              throw std::invalid_argument("TLA DMA requires memref operands");
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
            if (!isa<MemRefType>(src.getType()) ||
                !isa<MemRefType>(dst.getType()))
              throw std::invalid_argument("TLA DMA requires memref operands");
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
      // actual tile sizes (arith.constant index); transpose_b sets UnitAttr.
      .def(
          "create_hivm_mmadl1",
          [](TLABuilder &self, Value a, Value b, Value c, bool init_condition,
             int64_t real_m, int64_t real_k, int64_t real_n, bool transpose_b) {
            auto &builder = self.getBuilder();
            auto loc = self.getLastLoc();
            auto init = builder.create<arith::ConstantOp>(
                loc, builder.getI1Type(), builder.getBoolAttr(init_condition));
            auto m = builder.create<arith::ConstantIndexOp>(loc, real_m);
            auto k = builder.create<arith::ConstantIndexOp>(loc, real_k);
            auto n = builder.create<arith::ConstantIndexOp>(loc, real_n);
            auto bTrans = transpose_b ? builder.getUnitAttr() : UnitAttr{};
            self.create<hivm::MmadL1Op>(
                TypeRange{}, a, b, init, m, k, n, c,
                /*per_channel_bias=*/Value{}, /*a_transpose=*/UnitAttr{},
                /*b_transpose=*/bTrans, /*enable_HF32=*/UnitAttr{},
                /*enable_I4=*/UnitAttr{});
          },
          py::arg("a"), py::arg("b"), py::arg("c"), py::arg("init_condition"),
          py::arg("real_m"), py::arg("real_k"), py::arg("real_n"),
          py::arg("transpose_b"))
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
             Region &region = scopeOp->getRegion(0);
             // If the region was left empty, create the entry block.
             if (region.empty())
               region.emplaceBlock();
             return OpState(scopeOp);
           })
      // Returns a non-owning reference to the scope entry block.  pybind11
      // wraps Block with a unique_ptr holder, so a bare Block* under the
      // default automatic policy would delete the block when the Python
      // wrapper is GC'd; reference keeps the block alive in the IR.
      .def(
          "get_scope_entry_block",
          [](TLABuilder &self, OpState op) -> Block * {
            return &op->getRegion(0).front();
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
