// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef SPECIALIZED_MEMORY_MAIN_CASES_H
#define SPECIALIZED_MEMORY_MAIN_CASES_H
#include "AscendModel/RouteModel/SpecializedMemoryMain.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Parser/Parser.h"
#include "triton/Dialect/Triton/IR/Dialect.h"

namespace mlir::ascend::specialized_test {
inline std::string directFragmentIR(bool scatter = false) {
  std::string ir = R"mlir(module {
 tt.func public @fragment(%a: !tt.ptr<f32>, %idx: !tt.ptr<i64>, %b: !tt.ptr<f32>) {
  %r = tt.make_range {start = 0 : i32, end = 4 : i32} : tensor<4xi32>
  %c = tt.make_range {start = 0 : i32, end = 8 : i32} : tensor<8xi32>
  %ip = tt.splat %idx : !tt.ptr<i64> -> tensor<4x!tt.ptr<i64>>
  %ix = tt.addptr %ip, %r : tensor<4x!tt.ptr<i64>>, tensor<4xi32>
  %i = tt.load %ix : tensor<4x!tt.ptr<i64>>
  %ri = tt.expand_dims %i {axis = 1 : i32} : tensor<4xi64> -> tensor<4x1xi64>
  %rr = tt.expand_dims %r {axis = 1 : i32} : tensor<4xi32> -> tensor<4x1xi32>
  %cc = tt.expand_dims %c {axis = 0 : i32} : tensor<8xi32> -> tensor<1x8xi32>
  %s64 = arith.constant dense<8> : tensor<4x1xi64>
  %s32 = arith.constant dense<8> : tensor<4x1xi32>
  %d = arith.muli %ri, %s64 : tensor<4x1xi64>
  %s = arith.muli %rr, %s32 : tensor<4x1xi32>
  %ds = tt.broadcast %d : tensor<4x1xi64> -> tensor<4x8xi64>
  %ss = tt.broadcast %s : tensor<4x1xi32> -> tensor<4x8xi32>
  %cs = tt.broadcast %cc : tensor<1x8xi32> -> tensor<4x8xi32>
  %ap = tt.splat %a : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
  %bp = tt.splat %b : !tt.ptr<f32> -> tensor<4x8x!tt.ptr<f32>>
  %a0 = tt.addptr %ap, %ds : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi64>
  %a1 = tt.addptr %a0, %cs : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
  %b0 = tt.addptr %bp, %ss : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
  %b1 = tt.addptr %b0, %cs : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>
  %v = tt.load %a1 : tensor<4x8x!tt.ptr<f32>>
  tt.store %b1, %v : tensor<4x8x!tt.ptr<f32>>
  tt.return
 }
})mlir";
  if (scatter) {
    auto a = ir.find("%a0 = tt.addptr %ap, %ds :");
    ir.replace(
        a,
        std::string("%a0 = tt.addptr %ap, %ds : tensor<4x8x!tt.ptr<f32>>, "
                    "tensor<4x8xi64>")
            .size(),
        "%a0 = tt.addptr %ap, %ss : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi32>");
    auto b = ir.find("%b0 = tt.addptr %bp, %ss :");
    ir.replace(
        b,
        std::string("%b0 = tt.addptr %bp, %ss : tensor<4x8x!tt.ptr<f32>>, "
                    "tensor<4x8xi32>")
            .size(),
        "%b0 = tt.addptr %bp, %ds : tensor<4x8x!tt.ptr<f32>>, tensor<4x8xi64>");
  }
  return ir;
}
inline bool compilerChecks() {
  MLIRContext context;
  context.disableMultithreading();
  context.getOrLoadDialect<arith::ArithDialect>();
  context.getOrLoadDialect<triton::TritonDialect>();
  bool allParsed = true;
  auto proof = [&](const std::string &ir) {
    auto module = parseSourceString<ModuleOp>(ir, &context);
    allParsed &= bool(module);
    return module ? proveSpecializedMemoryFragment(*module)
                  : std::optional<SpecializedMemoryGeometry>();
  };
  for (bool scatter : {false, true}) {
    auto ir = directFragmentIR(scatter);
    auto g = proof(ir);
    if (!g ||
        g->operation != (scatter ? SpecializedMemoryOperation::Store
                                 : SpecializedMemoryOperation::Load) ||
        !g->completeFragmentBoundary || !g->directThreePointerEnvelope)
      return false;
    auto i32 = ir;
    for (auto pos = i32.find("i64"); pos != std::string::npos;
         pos = i32.find("i64", pos + 3))
      i32.replace(pos, 3, "i32");
    if (!proof(i32))
      return false;
    for (auto mode :
         {SpecializedMemoryMode::SIMD, SpecializedMemoryMode::SIMT}) {
      g->mode = mode;
      if (!estimateSpecializedMemory(*g).supported)
        return false;
    }
    // An additional sink, payload arithmetic, shifted index origin and an
    // unused computation must not be removed by full-envelope replacement.
    for (auto extra :
         {std::string("tt.store %b1, %v : tensor<4x8x!tt.ptr<f32>>\n"),
          std::string("%unused = arith.constant 1 : i32\n")}) {
      auto changed = ir;
      changed.insert(changed.find("tt.return"), extra);
      if (proof(changed))
        return false;
    }
    auto shifted = ir;
    shifted.replace(shifted.find("start = 0 : i32, end = 4"),
                    std::string("start = 0 : i32, end = 4").size(),
                    "start = 1 : i32, end = 5");
    if (proof(shifted))
      return false;
    auto computed = ir;
    computed.insert(computed.find("tt.store"),
                    "%sum = arith.addf %v, %v : tensor<4x8xf32>\n");
    computed.replace(computed.find("tt.store %b1, %v"),
                     std::string("tt.store %b1, %v").size(),
                     "tt.store %b1, %sum");
    if (proof(computed))
      return false;
    auto masked = ir;
    masked.insert(masked.find("%i = tt.load"),
                  "%mask = arith.constant dense<true> : tensor<4xi1>\n");
    masked.replace(masked.find("%i = tt.load %ix :"),
                   std::string("%i = tt.load %ix :").size(),
                   "%i = tt.load %ix, %mask :");
    if (proof(masked))
      return false;
  }
  return allParsed;
}
} // namespace mlir::ascend::specialized_test
#endif
