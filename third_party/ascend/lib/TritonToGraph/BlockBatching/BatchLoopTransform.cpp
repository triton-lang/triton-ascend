// Copyright (c) Huawei Technologies Co., Ltd. 2026. All rights reserved.
#include "TritonToGraph/BlockBatching/BatchLoopTransform.h"
#include "mlir/Dialect/SCF/Utils/Utils.h"
#include "mlir/IR/Matchers.h"
#include "mlir/Interfaces/SideEffectInterfaces.h"

namespace mlir::triton::cfg {
LogicalResult groupBatchLoopIterations(scf::ForOp loop, int64_t factor) {
  // Recheck the actual IR immediately before the upstream utility, which
  // performs host arithmetic for static bounds. Keep direct callers safe too.
  BatchLoopFact fact;
  auto type = dyn_cast<IntegerType>(loop.getInductionVar().getType());
  fact.bits = type ? type.getWidth() : 64;
  auto bound = [&](Value value) -> BatchExprRef {
    llvm::APInt constant;
    if (!matchPattern(value, m_ConstantInt(&constant)))
      return BatchExpr::leaf(BatchExpr::Argument, 0, fact.bits);
    if (!constant.isSignedIntN(64))
      return {};
    return BatchExpr::leaf(BatchExpr::Constant, constant.getSExtValue(),
                           fact.bits);
  };
  fact.lower = bound(loop.getLowerBound());
  fact.upper = bound(loop.getUpperBound());
  fact.step = bound(loop.getStep());
  if (!fact.lower || !fact.upper || !fact.step ||
      batchLoopIterationArithmetic(fact, BatchRangeFacts(0), factor) ==
          BatchLoopArithmetic::Unsafe)
    return failure();
  MLIRContext *context = loop.getContext();
  auto grouped = loopUnrollByFactor(loop, factor);
  if (failed(grouped))
    return failure();
  // Match the existing Triton loop-unroll epilogue contract.
  if (grouped->epilogueLoopOp)
    (*grouped->epilogueLoopOp)
        ->setAttr("tt.num_stages",
                  IntegerAttr::get(IntegerType::get(context, 32), 1));
  return success();
}
namespace {

// 界表达式在实测 kernel 里都只有两三个 op，这个预算只是防病态输入。
constexpr unsigned kBudget = 256;

// value 的定义链**可能**读到 programId 吗。证得出读不到才返回 false——那种子链
// 整条交给调用方的 remap，一个 op 都不克隆。
//
// 不确定时一律返回 true，由主路径按原样往下走，那里有带区域的 op 与预算这两道
// 拒绝。反过来写（不确定就当无关）会把两道都绕过去：一个从区域里捕获程序号的 op
// 操作数上看不见它，会被当成无关子链原样留下，而那条界仍然用着原来的程序号。
bool mayReadProgramId(Value value, Value programId, unsigned &budget) {
  if (value == programId)
    return true;
  if (!budget)
    return true;
  --budget;
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return false; // 函数参数：定义在循环之外，与程序号无关。
  if (definition->getNumRegions())
    return true;
  return llvm::any_of(definition->getOperands(), [&](Value operand) {
    return mayReadProgramId(operand, programId, budget);
  });
}

// 把 value 的定义链重建一份，其中对 programId 的读取换成 folded。预算用尽或遇到
// 不敢克隆的 op 返回空，由调用方整条放弃——绝不返回一条只换了一半的表达式。
Value substitute(OpBuilder &builder, Value value, Value programId, Value folded,
                 BatchValueRemap remap, unsigned &budget) {
  if (value == programId)
    return folded;
  if (!budget)
    return nullptr;
  --budget;
  // 与程序号无关的整条子链：不重建，问调用方要新体里的对应物。原样留下会在原函数
  // 体被 erase 之后变成悬空引用。
  unsigned probe = budget;
  if (!mayReadProgramId(value, programId, probe))
    return remap(value);
  Operation *definition = value.getDefiningOp();
  if (!definition)
    return remap(value); // 函数参数之类，与程序号无关。
  // 带区域的 op 能从外层捕获值，程序号可以藏在区域里而不出现在操作数上。那样
  // 下面"操作数都没变，所以整条与程序号无关"的推断就不成立，会把一条仍然用着
  // 原程序号的界原样留下。这一关必须在那个推断之前，不能并进克隆前的判断。
  if (definition->getNumRegions())
    return nullptr;
  SmallVector<Value> operands;
  bool changed = false;
  for (Value operand : definition->getOperands()) {
    Value rebuilt =
        substitute(builder, operand, programId, folded, remap, budget);
    if (!rebuilt)
      return nullptr;
    changed |= rebuilt != operand;
    operands.push_back(rebuilt);
  }
  if (!changed)
    return remap(value); // 走到这里说明预算在探测时用尽了，保守交给调用方。
  // 克隆一条读内存的指令会改变访存次数，证不出等价就不做（C9）。
  if (!isMemoryEffectFree(definition))
    return nullptr;
  Operation *clone = builder.clone(*definition);
  clone->setOperands(operands);
  // 按结果序号取，不假定单结果：value 可能是第二个结果。
  return clone->getResult(cast<OpResult>(value).getResultNumber());
}

} // namespace

BatchRetiledBounds retileBatchLoopBounds(OpBuilder &builder, scf::ForOp loop,
                                         const BatchLogicalAxis &axis,
                                         Value programId, Value foldedProgramId,
                                         BatchValueRemap remap) {
  // 两种分布共用这一段，不按 distribution 分支：它们的界都写成程序号的函数
  // （成块发 p*R，定长分块 p*R → min((p+1)*R, N)），把程序号换成折叠后的
  // lane 号，算出来的就是这条 lane 该走的那一段。差别只在圈数——定长分块各
  // lane 可能差一个尾部，那是循环体掩码的事，不是界的事。
  (void)axis;
  if (!programId || !foldedProgramId ||
      programId.getType() != foldedProgramId.getType())
    return {};

  unsigned budget = kBudget;
  BatchRetiledBounds bounds;
  bounds.lower = substitute(builder, loop.getLowerBound(), programId,
                            foldedProgramId, remap, budget);
  bounds.upper = substitute(builder, loop.getUpperBound(), programId,
                            foldedProgramId, remap, budget);
  // 步长不换程序号，但仍要过 remap：它同样读的是原 IR。tt.get_num_programs 已被
  // 重映射回隐藏的原始网格参数（BatchRewrite.cpp:407-413），所以这个表达式读到的
  // 是原始的 G 而不是 G/F；再乘一个 F 会让它变成 G*R*F，跳过 (F-1)/F 的行。
  bounds.step = remap(loop.getStep());
  if (!bounds.ok())
    return {};
  return bounds;
}

} // namespace mlir::triton::cfg
