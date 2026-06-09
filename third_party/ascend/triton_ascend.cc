#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Dialect/GPU/IR/GPUDialect.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/Linalg/IR/Linalg.h"
#include "mlir/Dialect/Linalg/Passes.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tensor/IR/Tensor.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Parser/Parser.h"

#include "ascend/include/AutoBlockify/Passes.h"
#include "ascend/include/TritonToStructured/Passes.h"
#include "ascend/include/TritonToAnnotation/Passes.h"
#include "ascend/include/TritonControlFlowOpt/Passes.h"
#include "ascend/include/TritonToLinalg/Passes.h"
#include "ascend/include/Dialect/TritonAscend/IR/TritonAscendDialect.h"
#include "ascend/include/DiscreteMaskAccessConversion/Passes.h"
#include "ascend/include/TritonToUnstructure/Passes.h"
#include "ascend/include/TritonToHIVM/Passes.h"
#include "ascend/include/TritonToHFusion/Passes.h"
#include "ascend/include/TritonToLLVM/Passes.h"

#include "ascend/include/DynamicCVPipeline/Passes.h"
#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
// todo: this code will be removed in version 530.
#include "ascend/include/TritonAffinityOpt/Passes.h"

#include "triton/Dialect/Triton/IR/Dialect.h"
#include "ir.h" // TritonOpBuilder

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
#include "AscendModel/IR/AscendModelDialect.h"
#include "AscendModel/Transforms/Passes.h"

#include "mlir/Dialect/Affine/IR/AffineOps.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Math/IR/Math.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include <cstring>
#include <sstream>
#include <stdexcept>
#include <vector>
#endif

namespace py = pybind11;
using namespace ir;
using namespace mlir;

void init_triton_ascend_ir(py::module &&m) {
  auto *builder_cls = ir::getBuilderClass();
  builder_cls
    ->def("create_extract_scalar",
      [](TritonOpBuilder &self, Value &src, std::vector<Value> &indices) -> Value {
      llvm::SmallVector<Value> arg_indices;
      for (const auto &i : indices) {
        auto iTy = i.getType();
        if (!iTy.isIndex()) {
          auto v = self.create<arith::IndexCastOp>(
            self.getBuilder().getIndexType(), i);
          arg_indices.push_back(v);
        } else {
          arg_indices.push_back(i);
        }
      }
      auto ret = self.create<tensor::ExtractOp>(src, arg_indices);
      return ret;
    })
    .def("create_extract_slice",
      [](TritonOpBuilder &self, Value &ful, std::vector<Value> &offs_vec,
        std::vector<int> &sizs_vec, std::vector<int> &strd_vec) -> Value {
        llvm::SmallVector<Value> offsets;
        for (const auto &o : offs_vec) {
          auto oTy = o.getType();
          if (!oTy.isIndex()) {
            auto v = self.create<arith::IndexCastOp>(
              self.getBuilder().getIndexType(), o);
            offsets.push_back(v);
          } else {
            offsets.push_back(o);
          }
        }
        llvm::SmallVector<Value> sizes;
        llvm::SmallVector<int64_t> retSizes;
        for (const auto &s : sizs_vec) {
          auto v = self.create<arith::ConstantIndexOp>(s);
          sizes.push_back(v);
          retSizes.push_back(s);
        }
        llvm::SmallVector<Value> strides;
        for (const auto &s : strd_vec) {
          auto v = self.create<arith::ConstantIndexOp>(s);
          strides.push_back(v);
        }
        auto retTy = RankedTensorType::get(retSizes,
          cast<RankedTensorType>(ful.getType()).getElementType());

        return self.create<tensor::ExtractSliceOp>(retTy, ful, offsets, sizes, strides);
      })
    .def("create_insert_slice",
      [](TritonOpBuilder &self, Value &ful, Value &sub,
        std::vector<Value> &offs_vec, std::vector<int> &sizs_vec,
        std::vector<int> &strd_vec) -> Value {
        llvm::SmallVector<Value> offsets;
        for (const auto &o : offs_vec) {
          auto oTy = o.getType();
          if (!oTy.isIndex()) {
            auto v = self.create<arith::IndexCastOp>(
              self.getBuilder().getIndexType(), o);
            offsets.push_back(v);
          } else {
            offsets.push_back(o);
          }
        }
        llvm::SmallVector<Value> sizes;
        llvm::SmallVector<int64_t> retSizes;
        for (const auto &s : sizs_vec) {
          auto v = self.create<arith::ConstantIndexOp>(s);
          sizes.push_back(v);
          retSizes.push_back(s);
        }
        llvm::SmallVector<Value> strides;
        for (const auto &s : strd_vec) {
          auto v = self.create<arith::ConstantIndexOp>(s);
          strides.push_back(v);
        }
        auto retTy = RankedTensorType::get(
          retSizes,
          cast<RankedTensorType>(ful.getType()).getElementType());
        auto ret = self.create<tensor::InsertSliceOp>(sub, ful, offsets,
                                                      sizes, strides);
        return ret;
      })
    .def("create_custom_op_for_inter_core_sync",
      [](TritonOpBuilder &self, std::string &op_name,
        std::string &mode_or_sender, int id) -> void {
          auto args = self.getBuilder().getArrayAttr(
              {self.getBuilder().getStringAttr(mode_or_sender),
              self.getBuilder().getI32IntegerAttr(id)}
          );
          self.create<triton::ascend::CustomOp>(op_name, args, ValueRange());
      })
    .def("create_index_select_simd",
      [](TritonOpBuilder &self, Value &src, Value &index, int32_t dim,
        std::vector<Value> &srcShape, std::vector<Value> &srcOffset,
        std::vector<int32_t> &readShape, std::vector<int32_t> &returnShape) -> Value {
          auto &builder = self.getBuilder();
          auto loc = self.getLastLoc();

          // Get element type from source pointer
          Type elemType;
          if (auto ptrTy = dyn_cast<triton::PointerType>(src.getType())) {
            elemType = ptrTy.getPointeeType();
          } else {
            llvm::report_fatal_error("index_select_simd: src must be pointer type");
          }

          // Create return tensor type
          llvm::SmallVector<int64_t> retShape;
          for (const auto &s : returnShape) {
            retShape.push_back(s);
          }
          auto retTensorType = RankedTensorType::get(retShape, elemType);

          // Convert srcShape and srcOffset values to index type if needed
          llvm::SmallVector<Value> srcShapeIndex;
          for (auto val : srcShape) {
            if (!val.getType().isIndex()) {
              val = self.create<arith::IndexCastOp>(builder.getIndexType(), val);
            }
            srcShapeIndex.push_back(val);
          }
          
          llvm::SmallVector<Value> srcOffsetIndex;
          for (auto val : srcOffset) {
            if (!val.getType().isIndex()) {
              val = self.create<arith::IndexCastOp>(builder.getIndexType(), val);
            }
            srcOffsetIndex.push_back(val);
          }

          // Create attributes
          auto dimAttr = builder.getI32IntegerAttr(dim);
          auto readShapeAttr = builder.getDenseI32ArrayAttr(readShape);

          // Create the IndexSelectSimdOp
          // Parameter order must match TritonOps.td definition:
          // src, index, dim, src_shape, src_offset, read_shape
          auto indexSelectSimdOp = builder.create<triton::ascend::IndexSelectSimdOp>(
              loc,
              retTensorType,        // result type
              src,                  // src pointer
              index,                // index tensor
              dimAttr,              // dim attribute
              srcShapeIndex,        // src_shape (variadic, index type)
              srcOffsetIndex,       // src_offset (variadic, index type)
              readShapeAttr         // read_shape attribute
          );

          return indexSelectSimdOp.getResult();
      })
    .def("create_index_put",
      [](TritonOpBuilder &self, Value &ptr, Value &index,
        Value &value, const int32_t dim, const int64_t indexBoundary,
        std::vector<Value> &endOffset, std::vector<Value> &startOffset,
        std::vector<Value> &dstStride) -> void {
          // dim need to be i32 type
          auto dimI32Ty = self.getBuilder().getI32Type();
          auto dim_val = self.create<arith::ConstantIntOp>(dim, dimI32Ty);
          // indexBoundary need to be i64 type
          auto BoundI64Ty = self.getBuilder().getI64Type();
          auto bound_val = self.create<arith::ConstantIntOp>(indexBoundary, BoundI64Ty);

          self.create<triton::ascend::IndexPutOp>(
            ptr,
            index,
            value,
            dim_val,
            bound_val,
            endOffset,
            startOffset,
            dstStride
          );
      })
    .def("create_gather_out_to_ub",
      [](TritonOpBuilder &self, Value &src, Value &index, const int64_t indexBoundary,
        const int32_t dim, std::vector<Value> &srcStride, std::vector<Value> &endOffset,
        std::vector<Value> &startOffset, std::optional<Value> &other) -> Value {
          auto elemTy = cast<PointerType>(src.getType()).getPointeeType();
          auto idxTy = cast<RankedTensorType>(index.getType());
          auto idxShape = idxTy.getShape();
          std::vector<int64_t> retShape(idxShape.begin(), idxShape.end());
          auto resType = RankedTensorType::get(retShape, elemTy);

          // indexBoundary need to be i64 type
          auto BoundI64Ty = self.getBuilder().getI64Type();
          auto bound_val = self.create<arith::ConstantIntOp>(indexBoundary, BoundI64Ty);
          // dim need to be i32 type
          auto dimI32Ty = self.getBuilder().getI32Type();
          auto dim_val = self.create<arith::ConstantIntOp>(dim, dimI32Ty);
          return self.create<triton::ascend::GatherOutToUbOp>(
            resType,
            src,
            index,
            bound_val,
            dim_val,
            srcStride,
            endOffset,
            startOffset,
            other.value_or(Value())
          );
      })
    .def("create_scatter_ub_to_out",
      [](TritonOpBuilder &self, Value &ptr, Value &value, Value &index,
        const int64_t indexBoundary, const int32_t dim, std::vector<Value> &dstStride,
        std::vector<Value> &endOffset, std::vector<Value> &startOffset) -> void {
          auto idxTy = cast<RankedTensorType>(index.getType());

          // indexBoundary need to be i64 type
          auto BoundI64Ty = self.getBuilder().getI64Type();
          auto bound_val = self.create<arith::ConstantIntOp>(indexBoundary, BoundI64Ty);
          // dim need to be i32 type
          auto dimI32Ty = self.getBuilder().getI32Type();
          auto dim_val = self.create<arith::ConstantIntOp>(dim, dimI32Ty);

          self.create<triton::ascend::ScatterUbToOutOp>(
            ptr,
            value,
            index,
            bound_val,
            dim_val,
            dstStride,
            endOffset,
            startOffset
          );
      })
    // Add sort
    .def("create_sort",
      [](TritonOpBuilder &self, Value src, int64_t dim, bool descending) -> Value {
        auto &builder = self.getBuilder();
        auto loc = self.getLastLoc();

        auto dimAttr = builder.getI64IntegerAttr(dim);
        auto descendingAttr = builder.getBoolAttr(descending);

        auto op = builder.create<triton::ascend::SortOp>(loc, src, dimAttr, descendingAttr);

        return op->getResult(0);
      })
    // Add flip
    .def("create_flip",
      [](TritonOpBuilder &self, Value src, int64_t dim) -> Value {
        auto &builder = self.getBuilder();
        auto loc = self.getLastLoc();

        auto dimAttr = builder.getI64IntegerAttr(dim);

        auto op = builder.create<triton::ascend::FlipOp>(loc, src, dimAttr);

        return op->getResult(0);
      })
    .def("create_tanh",
      [](TritonOpBuilder &self, Value &val) -> Value {
        return self.create<math::TanhOp>(val);
      })
    // Add an annotation
    .def("create_annotation",
      [](TritonOpBuilder &self, Value &ptr, const std::string &attrKey,
        Attribute &attrVal) {
        auto annotationOp = self.create<triton::ascend::AnnotationOp>(ptr);
        annotationOp->setAttr(self.getBuilder().getStringAttr(attrKey),
                              attrVal);
      });
}

void init_triton_ascend_passes_ttir(py::module &&m) {
  m.def("add_auto_blockify", [](mlir::PassManager &pm,
    int autoBlockifySize) {
    AutoBlockifyOptions opts;
    opts.autoBlockifySize = autoBlockifySize;
    pm.addPass(mlir::triton::createAutoBlockifyPass(opts));});

  m.def("add_triton_to_structure", [](mlir::PassManager &pm,
    bool enableMaskFallbackConversion, bool optimizeDynamicOffset) {
    pm.addPass(mlir::triton::createTritonToStructuredPass(
      enableMaskFallbackConversion, optimizeDynamicOffset)); });

  m.def("add_triton_control_flow_opt", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonControlFlowOptPass());});

  m.def("add_triton_to_annotation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToAnnotationPass());});

  m.def("add_triton_to_linalg", [](mlir::PassManager &pm, bool globalKernel,
    bool namedOps, bool enableNd2nzOnVector, bool enableSelectAnalysis,
    bool compileOn91095) {
    pm.addPass(mlir::triton::createTritonToLinalgPass(
      globalKernel, namedOps, enableNd2nzOnVector,
      enableSelectAnalysis, compileOn91095)); });

  m.def("add_triton_to_unstructure", [](mlir::PassManager &pm,
    bool compileOn91095, bool forceSimtTemplate) {
    TritonToUnstructureOptions opts;
    opts.compileOn91095 = compileOn91095;
    opts.forceSimtTemplate = forceSimtTemplate;
    pm.addPass(mlir::triton::createTritonToUnstructurePass(opts));});

  m.def("add_triton_to_hfusion", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToHFusionPass());});

  m.def("add_discrete_mask_access_conversion", [](mlir::PassManager &pm,
    bool compileOn91095, bool forceSimtTemplate, bool enableSyncBlockLock) {
    DiscreteMaskAccessConversionOptions opts;
    opts.compileOn91095 = compileOn91095;
    opts.forceSimtTemplate = forceSimtTemplate;
    opts.enableSyncBlockLock = enableSyncBlockLock;
    pm.addPass(mlir::triton::createDiscreteMaskAccessConversionPass(opts));});

  m.def("add_triton_to_hivm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToHIVMPass());});

  m.def("add_triton_to_llvm", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createTritonToLLVMPass());});
  
  m.def("add_bubble_up_operation", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createBubbleUpOperationPass());});

  m.def("add_dynamic_cv_pipeline", [](mlir::PassManager &pm,
    bool compileOn91095) {
      AddDynamicCVPipelineOptions opts;
      opts.compileOn91095 = compileOn91095;
      pm.addPass(mlir::triton::createAddDynamicCVPipelinePass(opts));
    });

  // todo: this code will be removed in version 530.
  m.def("add_dag_sync", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createDAGSyncPass());});
 	   
  m.def("add_dag_scope", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createDAGScopePass());});
 	   
  m.def("add_dag_ssbuffer", [](mlir::PassManager &pm) {
    pm.addPass(mlir::triton::createDAGSSBufferPass());});

  m.def("set_buffer_count", [](const std::string& type, int count) {
    if (type == "INTRA") {
      mlir::triton::BufferCountManager::getInstance().setBufferCount(
          mlir::triton::BufferCountManager::DepType::IntraCore, count);
    } else if (type == "INTER") {
      mlir::triton::BufferCountManager::getInstance().setBufferCount(
          mlir::triton::BufferCountManager::DepType::InterCore, count);
    } else if (type == "LOAD") {
      mlir::triton::BufferCountManager::getInstance().setBufferCount(
          mlir::triton::BufferCountManager::DepType::LoadStore, count);
    }
  });
}

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
namespace {
struct AscendCostModelRuntimeOptions {
  bool allowUnregisteredDialect = false;
  bool enableCostModelPipeline = false;
  std::string argBindings;
  std::string hardwareConfig;
};

static void parseAscendCostModelOptionString(const std::string &option,
                                              AscendCostModelRuntimeOptions &opts) {
  // Parse known keys while allowing commas inside values, e.g.
  // arg-bindings=arg8=256,arg9=80,arg10=32,pid_x=0
  auto startsWith = [](const std::string &s, const std::string &p) {
    return s.rfind(p, 0) == 0;
  };

  auto trim = [](std::string v) {
    while (!v.empty() && (v.back() == ' ' || v.back() == ',')) {
      v.pop_back();
    }
    size_t i = 0;
    while (i < v.size() && v[i] == ' ') {
      ++i;
    }
    return v.substr(i);
  };

  auto extractValue = [&](const std::string &key) -> std::string {
    const std::string needle = key + "=";
    size_t keyPos = option.find(needle);
    if (keyPos == std::string::npos) {
      return "";
    }

    size_t valueStart = keyPos + needle.size();
    size_t valueEnd = option.size();

    const std::string keys[] = {"arg-bindings", "hardware-config"};
    for (const auto &otherKey : keys) {
      if (otherKey == key) {
        continue;
      }
      const std::string otherNeedle = otherKey + "=";
      const std::string commaBoundary = "," + otherNeedle;
      const std::string spaceBoundary = " " + otherNeedle;

      size_t p1 = option.find(commaBoundary, valueStart);
      if (p1 != std::string::npos) {
        valueEnd = std::min(valueEnd, p1);
      }
      size_t p2 = option.find(spaceBoundary, valueStart);
      if (p2 != std::string::npos) {
        valueEnd = std::min(valueEnd, p2);
      }
    }

    return trim(option.substr(valueStart, valueEnd - valueStart));
  };

  // Fast path for single-key payloads passed as next argv token.
  if (startsWith(option, "arg-bindings=")) {
    opts.argBindings = trim(option.substr(std::string("arg-bindings=").size()));
    return;
  }
  if (startsWith(option, "hardware-config=")) {
    opts.hardwareConfig = trim(option.substr(std::string("hardware-config=").size()));
    return;
  }

  auto argBindings = extractValue("arg-bindings");
  if (!argBindings.empty()) {
    opts.argBindings = argBindings;
  }
  auto hardwareConfig = extractValue("hardware-config");
  if (!hardwareConfig.empty()) {
    opts.hardwareConfig = hardwareConfig;
  }
}

static AscendCostModelRuntimeOptions parseAscendCostmodelArgs(const std::vector<std::string> &extraArgs) {
  AscendCostModelRuntimeOptions opts;
  for (size_t i = 0; i < extraArgs.size(); ++i) {
    const auto &arg = extraArgs[i];
    if (arg.empty()) {
      continue;
    }
    if (arg == "-allow-unregistered-dialect") {
      opts.allowUnregisteredDialect = true;
      continue;
    }
    if (arg == "-ascend-perf-model" || arg == "--ascend-perf-model") {
      opts.enableCostModelPipeline = true;
      if (i + 1 < extraArgs.size() && !extraArgs[i + 1].empty() && extraArgs[i + 1][0] != '-') {
        parseAscendCostModelOptionString(extraArgs[++i], opts);
      }
      continue;
    }
    constexpr const char *kShortPrefix = "-ascend-perf-model=";
    constexpr const char *kLongPrefix = "--ascend-perf-model=";
    if (arg.rfind(kShortPrefix, 0) == 0) {
      opts.enableCostModelPipeline = true;
      parseAscendCostModelOptionString(arg.substr(std::strlen(kShortPrefix)), opts);
      continue;
    }
    if (arg.rfind(kLongPrefix, 0) == 0) {
      opts.enableCostModelPipeline = true;
      parseAscendCostModelOptionString(arg.substr(std::strlen(kLongPrefix)), opts);
      continue;
    }
  }
  return opts;
}

static double extractEstimatedTimeUs(mlir::ModuleOp module) {
  constexpr double CYCLES_PER_US = 1850.0;
  if (auto scheduledCyclesAttr = module->getAttrOfType<mlir::IntegerAttr>("ascend.scheduled_cycles")) {
    return static_cast<double>(scheduledCyclesAttr.getInt()) / CYCLES_PER_US;
  }

  int64_t fallbackCycles = 0;
  module.walk([&](mlir::Operation *op) {
    if (auto cyclesAttr = op->getAttrOfType<mlir::IntegerAttr>("estimated_cycles")) {
      fallbackCycles += cyclesAttr.getInt();
    }
  });
  return static_cast<double>(fallbackCycles) / CYCLES_PER_US;
}
} // namespace

static std::string runAscendCostModelInProcess(const std::string &mlirText,
                                               const std::vector<std::string> &extraArgs) {
  mlir::DialectRegistry registry;
  registry.insert<mlir::affine::AffineDialect>();
  registry.insert<mlir::arith::ArithDialect>();
  registry.insert<mlir::func::FuncDialect>();
  registry.insert<mlir::math::MathDialect>();
  registry.insert<mlir::memref::MemRefDialect>();
  registry.insert<mlir::scf::SCFDialect>();
  registry.insert<mlir::tensor::TensorDialect>();
  registry.insert<mlir::ascend::AscendModelDialect>();
  registry.insert<mlir::triton::TritonDialect>();

  auto opts = parseAscendCostmodelArgs(extraArgs);
  if (!opts.enableCostModelPipeline) {
    throw std::runtime_error("in-process costmodel requires '-ascend-perf-model'");
  }

  mlir::MLIRContext context(registry);
  if (opts.allowUnregisteredDialect) {
    context.allowUnregisteredDialects(true);
  }

  auto module = mlir::parseSourceString<mlir::ModuleOp>(mlirText, &context);
  if (!module) {
    throw std::runtime_error("in-process costmodel failed to parse input MLIR module");
  }

  mlir::PassManager pm(&context);
  pm.addPass(mlir::ascend::createConvertTritonToAscendPass());
  pm.addPass(mlir::ascend::createInsertDataTransfersPass());
  pm.addPass(mlir::ascend::createAssignOpIDsPass());
  {
    mlir::ascend::EstimateCyclesPassOptions estimateOpts;
    estimateOpts.argBindingsStr = opts.argBindings;
    estimateOpts.hardwareConfigPath = opts.hardwareConfig;
    pm.addPass(mlir::ascend::createEstimateCyclesPass(estimateOpts));
  }
  {
    mlir::ascend::PipelineAnalysisPassOptions pipelineOpts;
    pipelineOpts.argBindingsStr = opts.argBindings;
    pipelineOpts.hardwareConfigPath = opts.hardwareConfig;
    pm.addPass(mlir::ascend::createPipelineAnalysisPass(pipelineOpts));
  }

  if (mlir::failed(pm.run(*module))) {
    throw std::runtime_error("in-process costmodel pass pipeline failed");
  }

  const double timeUs = extractEstimatedTimeUs(*module);
  std::ostringstream os;
  os.setf(std::ios::fixed);
  os.precision(3);
  os << "Estimated Time: " << timeUs << " us\n";
  return os.str();
}
#endif

// Forward declaration for ascend_ir bindings (defined in ascend_ir.cc)
void init_ascend_ir(py::module &&m);

void init_triton_ascend(py::module &&m) {
  auto passes = m.def_submodule("passes");
  // load dialects
  m.def("load_dialects", [](mlir::MLIRContext &context) {
    mlir::DialectRegistry registry;
    registry.insert<mlir::triton::ascend::TritonAscendDialect>();
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  init_triton_ascend_passes_ttir(passes.def_submodule("ttir"));
  init_triton_ascend_ir(m.def_submodule("ascend_ir"));

#if TRITON_ASCEND_HAS_INPROC_COSTMODEL
  m.def("run_costmodel_inproc", [](const std::string &mlirText, const std::vector<std::string> &extraArgs) {
    py::gil_scoped_release release;
    return runAscendCostModelInProcess(mlirText, extraArgs);
  }, py::arg("mlir_text"), py::arg("extra_args") = std::vector<std::string>{});
#else
  m.def("run_costmodel_inproc", [](const std::string &, const std::vector<std::string> &) {
    throw std::runtime_error("in-process costmodel bridge is not enabled in this build");
    return std::string();
  }, py::arg("mlir_text"), py::arg("extra_args") = std::vector<std::string>{});
#endif

  // Initialize ascend IR bindings (ascendnpu_ir_builder, scope/hivm dialects)
  init_ascend_ir(m.def_submodule("ir"));
}
