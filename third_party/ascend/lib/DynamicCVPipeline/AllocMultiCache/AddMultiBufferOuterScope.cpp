#include "ascend/include/DynamicCVPipeline/AllocMultiCache/AddMultiBufferOuterScope.h"

#include <set>

#include "llvm/ADT/DenseMap.h"
#include "llvm/ADT/SmallSet.h"
#include "llvm/Support/Debug.h"

#include "bishengir/Dialect/Annotation/IR/Annotation.h"
#include "bishengir/Dialect/HIVM/IR/HIVM.h"
#include "bishengir/Dialect/Scope/IR/Scope.h"

#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Bufferization/IR/Bufferization.h"
#include "mlir/Dialect/LLVMIR/LLVMDialect.h"
#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/SCF/IR/SCF.h"

#include "ascend/include/DynamicCVPipeline/Common/BufferCountManager.h"
#include "ascend/include/DynamicCVPipeline/Common/FlagIdManager.h"
#include "ascend/include/DynamicCVPipeline/Common/Utils.h"

static constexpr const char *DEBUG_TYPE = "AddMultiBufferOuterScope";
#define LDBG(...)                                                              \
  LLVM_DEBUG(llvm::dbgs() << " [" << DEBUG_TYPE << "] " << __VA_ARGS__ << "\n")

using namespace mlir;
using namespace triton;
using namespace hivm;

namespace mlir {
namespace triton {

// Maximum number of flag allocation attempts per transfer group
static constexpr int kMaxFlagAttempts = 16;
// Flag ID 15 is reserved for pipe synchronization (e.g. PIPE_S) and must not
// be allocated to cross-core transfers. Usable IDs are 0..MAX_FLAG_ID (14).
static constexpr int kReservedPipeFlagId = 15;

// --- Attribute helpers ---

static int getFlagFromSyncOp(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>("flag_id")) {
    return attr.getInt();
  }
  if (auto attr = op->getAttrOfType<IntegerAttr>("static_flag_id")) {
    return attr.getInt();
  }
  if (auto attr = op->getAttrOfType<IntegerAttr>("flag")) {
    return attr.getInt();
  }
  return -1;
}

static int getBlockId(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(mlir::CVPipeline::kBlockId))
    return attr.getInt();
  return -1;
}

static int getTransferId(Operation *op) {
  if (auto attr = op->getAttrOfType<IntegerAttr>(mlir::CVPipeline::kTransferId))
    return attr.getInt();
  return -1;
}

// --- Address space helpers ---

static bool isInVectorScope(Operation *op) {
  auto scopeOp = op->getParentOfType<scope::ScopeOp>();
  if (!scopeOp) {
    return false;
  }
  if (auto tcoreAttr = scopeOp->getAttrOfType<TCoreTypeAttr>("hivm.tcore_type"))
    return tcoreAttr.getTcoretype() == TCoreType::VECTOR;
  return false;
}

// --- main_loop attribute helpers ---

/// Check if a sync op's direct parent is a main_loop op (forOp / whileOp
/// carrying the ssbuffer.main_loop attribute)
static bool parentOpHasMainLoopAttr(Operation *syncOp) {
  if (!syncOp) {
    return false;
  }
  return CVPipeline::isMainLoopOp(syncOp->getParentOp());
}

// --- Operation search helpers ---

/// Find sync op with a specific flag, searching forward or backward in a block
static Operation *findSyncOpWithFlag(Block *block, Operation *start, int flag,
                                     bool forward, bool wantWait) {
  if (!block) {
    return nullptr;
  }
  auto it = start->getIterator();
  if (forward) {
    for (auto e = block->end(); it != e; ++it) {
      Operation *op = &*it;
      if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
        continue;
      }
      if (getFlagFromSyncOp(op) != flag) {
        continue;
      }
      if (wantWait && isa<hivm::SyncBlockWaitOp>(op)) {
        return op;
      }
      if (!wantWait && isa<hivm::SyncBlockSetOp>(op)) {
        return op;
      }
    }
  } else {
    if (it == block->begin()) {
      return nullptr;
    }
    do {
      --it;
      Operation *op = &*it;
      if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
        continue;
      }
      if (getFlagFromSyncOp(op) != flag) {
        continue;
      }
      if (wantWait && isa<hivm::SyncBlockWaitOp>(op)) {
        return op;
      }
      if (!wantWait && isa<hivm::SyncBlockSetOp>(op)) {
        return op;
      }
    } while (it != block->begin());
  }
  return nullptr;
}

/// Find the to_tensor op after a given op in the same block
static Operation *findToTensorAfter(Block *block, Operation *start) {
  if (!block) {
    return nullptr;
  }
  auto it = start->getIterator();
  for (auto e = block->end(); it != e; ++it) {
    if (isa<bufferization::ToTensorOp>(&*it)) {
      return &*it;
    }
  }
  return nullptr;
}

// ============================================================================
// Step 1: Collect transfer group info
// ============================================================================

/// Collect all ops with ssbuffer.transfer_id, grouped by transfer_id
static int
collectOpsByTransferId(ModuleOp module,
                       DenseMap<int, SmallVector<Operation *>> &opsByTid) {
  module.walk([&](Operation *op) {
    if (!op->hasAttr(mlir::CVPipeline::kTransferId)) {
      return;
    }
    int tid = getTransferId(op);
    if (tid >= 0) {
      opsByTid[tid].push_back(op);
    }
  });
  LDBG("Collected " << opsByTid.size() << " transfer groups.");

  for (auto &p : opsByTid) {
    LDBG("  tid=" << p.first << " has " << p.second.size() << " ops.");
    DenseMap<int, int> blockIdCount;
    for (auto *op : p.second) {
      int bid = getBlockId(op);
      blockIdCount[bid]++;
    }
    for (auto &bp : blockIdCount) {
      LDBG("    block_id=" << bp.first << ": " << bp.second << " ops.");
    }
  }
  return 0;
}

/// Collect alloc/mark pairs from transfer ops in the group.
/// Identifies the correct cross-core buffer (ub/cbuf) used by each transfer op,
/// ignoring local buffers (cc on CUBE side) that are not part of the data
/// transfer.
static int collectBufferAllocs(const SmallVector<Operation *> &ops,
                               TransferGroupInfo &info) {
  // Helper: find the annotation.mark for a given alloc op
  auto findMarkForAlloc = [](Operation *allocOp) -> Operation * {
    Value allocResult = allocOp->getResult(0);
    for (auto *user : allocResult.getUsers()) {
      if (isa<annotation::MarkOp>(user))
        return user;
    }
    return nullptr;
  };

  // Identify sender's cross-core buffer from transferOp's outs operand
  if (info.senderChain.transferOp) {
    Operation *transferOp = info.senderChain.transferOp;
    // fixpipe / hir.copy: cross-core buffer is the last operand (outs)
    Value crossCoreBuf =
        transferOp->getOperand(transferOp->getNumOperands() - 1);
    if (auto *defOp = crossCoreBuf.getDefiningOp()) {
      if (isa<memref::AllocOp>(defOp)) {
        info.senderBuf.allocOp = defOp;
        info.senderBuf.markOp = findMarkForAlloc(defOp);
        LDBG("Sender cross-core buffer: alloc from transferOp outs.");
      }
    }
  }

  // Identify receiver's cross-core buffer from transferOp's input operand
  if (info.receiverChain.transferOp) {
    Operation *transferOp = info.receiverChain.transferOp;
    // memref.memory_space_cast / hivm.convert_layout: cross-core buffer is
    // the first operand
    Value crossCoreBuf = transferOp->getOperand(0);
    if (auto *defOp = crossCoreBuf.getDefiningOp()) {
      if (isa<memref::AllocOp>(defOp)) {
        info.receiverBuf.allocOp = defOp;
        info.receiverBuf.markOp = findMarkForAlloc(defOp);
        LDBG("Receiver cross-core buffer: alloc from transferOp input.");
      }
    }
  }

  // Collect alloc/mark for the OTHER side if not yet found.
  // Some transfer ops (e.g. fixpipe) have both a local input (cc) and a
  // cross-core output (ub). The receiver side's buffer is the cross-core one.
  // Walk all allocs in the group to find any remaining unassigned buffer.
  SmallVector<Operation *> allocs;
  for (Operation *op : ops) {
    if (isa<memref::AllocOp>(op))
      allocs.push_back(op);
  }

  // Fill missing side from remaining allocs (prefer allocs with marks)
  for (auto *allocOp : allocs) {
    if (allocOp == info.senderBuf.allocOp ||
        allocOp == info.receiverBuf.allocOp)
      continue;
    Operation *mark = findMarkForAlloc(allocOp);
    if (!info.senderBuf.allocOp) {
      info.senderBuf.allocOp = allocOp;
      info.senderBuf.markOp = mark;
    } else if (!info.receiverBuf.allocOp) {
      info.receiverBuf.allocOp = allocOp;
      info.receiverBuf.markOp = mark;
    }
  }

  LDBG("Sender buffer: " << (info.senderBuf.allocOp ? "alloc" : "none") << " + "
                         << (info.senderBuf.markOp ? "mark" : "none") << ".");
  LDBG("Receiver buffer: " << (info.receiverBuf.allocOp ? "alloc" : "none")
                           << " + "
                           << (info.receiverBuf.markOp ? "mark" : "none")
                           << ".");
  return 0;
}

/// Collect llvm.load volatile and llvm.store volatile ops by transfer_id
static int collectLoadStoreOpsByTransferId(
    ModuleOp module, DenseMap<int, SmallVector<Operation *>> &loadStoreByTid) {
  module.walk([&](Operation *op) {
    if (!op->hasAttr(mlir::CVPipeline::kTransferId)) {
      return;
    }
    int tid = getTransferId(op);
    if (tid < 0) {
      return;
    }
    if (isa<mlir::LLVM::LoadOp>(op) || isa<mlir::LLVM::StoreOp>(op)) {
      loadStoreByTid[tid].push_back(op);
    }
  });
  LDBG("Collected load/store ops for " << loadStoreByTid.size()
                                       << " transfer groups.");
  return 0;
}

/// Tag load/store ops with crossDeps (producer=store, consumer=load)
static int tagLoadStoreOpsWithCrossDeps(
    DenseMap<int, SmallVector<Operation *>> &loadStoreByTid) {
  for (auto &p : loadStoreByTid) {
    int tid = p.first;
    for (auto *op : p.second) {
      MLIRContext *ctx = op->getContext();
      OpBuilder builder(ctx);
      if (auto storeOp = dyn_cast<mlir::LLVM::StoreOp>(op)) {
        // producer: crossDeps = {tid, 1}
        // Tag the defining op of the store's second operand (ptr), not the
        // store itself
        Value ptr = storeOp.getOperand(1);
        if (auto *ptrDefOp = ptr.getDefiningOp()) {
          ptrDefOp->setAttr(
              mlir::CVPipeline::kCrossCoreDeps,
              builder.getArrayAttr({builder.getI32IntegerAttr(tid),
                                    builder.getI32IntegerAttr(1)}));
          LDBG("Tagged ptr-defining-op with crossDeps={tid=" << tid << ", 1}.");
        }
      } else if (auto loadOp = dyn_cast<mlir::LLVM::LoadOp>(op)) {
        // consumer: crossDeps = {tid, 0}
        // Tag the load op itself
        op->setAttr(mlir::CVPipeline::kCrossCoreDeps,
                    builder.getArrayAttr({builder.getI32IntegerAttr(tid),
                                          builder.getI32IntegerAttr(0)}));
        LDBG("Tagged llvm.load volatile with crossDeps={tid=" << tid
                                                              << ", 0}.");
      }
    }
  }
  return 0;
}

/// Collect extra sync ops (parent has no main_loop), paired by flag
static int collectExtraSync(const SmallVector<Operation *> &ops,
                            int originalFlag, ExtraSyncInfo &info) {
  SmallVector<Operation *> extraSets;
  SmallVector<Operation *> extraWaits;

  for (Operation *op : ops) {
    if (!(isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
      continue;
    }

    bool hasMainLoop = parentOpHasMainLoopAttr(op);
    LDBG("sync op: flag=" << getFlagFromSyncOp(op)
                          << ", block_id=" << getBlockId(op)
                          << ", parentHasMainLoop=" << hasMainLoop << ".");

    if (!hasMainLoop) {
      if (isa<hivm::SyncBlockSetOp>(op)) {
        extraSets.push_back(op);
      } else if (isa<hivm::SyncBlockWaitOp>(op)) {
        extraWaits.push_back(op);
      }
    }
  }

  // Match by flag
  for (auto *setOp : extraSets) {
    if (getFlagFromSyncOp(setOp) != originalFlag) {
      continue;
    }
    for (auto *waitOp : extraWaits) {
      if (getFlagFromSyncOp(waitOp) != originalFlag) {
        continue;
      }
      info.setOp = setOp;
      info.waitOp = waitOp;
      LDBG("Extra sync pair: set(flag="
           << originalFlag << ", block_id=" << getBlockId(setOp)
           << "), wait(flag=" << originalFlag
           << ", block_id=" << getBlockId(waitOp) << ".");
      return 0;
    }
  }

  // Fallback: use first available pair if exact match not found
  if (!extraSets.empty() && !extraWaits.empty()) {
    info.setOp = extraSets.front();
    info.waitOp = extraWaits.front();
  }

  return 0;
}

/// Collect transfer chain ops (parent has main_loop)
static int collectTransferChains(const SmallVector<Operation *> &ops,
                                 int originalFlag, TransferChainInfo &info) {
  for (Operation *op : ops) {
    if ((isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) ||
        !op->getBlock()) {
      continue;
    }
    if (!parentOpHasMainLoopAttr(op)) {
      continue;
    }

    Block *block = op->getBlock();

    if (isa<hivm::FixpipeOp>(op)) {
      info.sender.transferOp = op;
      info.sender.waitOp =
          findSyncOpWithFlag(block, op, originalFlag, false, true);
      info.sender.setOp =
          findSyncOpWithFlag(block, op, originalFlag, true, false);
      LDBG("Sender chain (CUBE): fixpipe, flag=" << originalFlag << ".");
    } else if (isa<hivm::CopyOp>(op)) {
      info.sender.transferOp = op;
      info.sender.waitOp =
          findSyncOpWithFlag(block, op, originalFlag, false, true);
      info.sender.setOp =
          findSyncOpWithFlag(block, op, originalFlag, true, false);
      LDBG("Sender chain (VECTOR): hir.copy, flag=" << originalFlag << ".");
    } else if (isa<memref::MemorySpaceCastOp>(op) && isInVectorScope(op)) {
      info.receiver.transferOp = op;
      info.receiver.waitOp =
          findSyncOpWithFlag(block, op, originalFlag, false, true);
      info.receiver.setOp =
          findSyncOpWithFlag(block, op, originalFlag, true, false);
      info.receiver.toTensorOp = findToTensorAfter(block, op);
      LDBG("Receiver chain (VECTOR): memory_space_cast, flag=" << originalFlag
                                                               << ".");
    } else if (isa<hivm::ConvertLayoutOp>(op)) {
      info.receiver.transferOp = op;
      info.receiver.waitOp =
          findSyncOpWithFlag(block, op, originalFlag, false, true);
      info.receiver.setOp =
          findSyncOpWithFlag(block, op, originalFlag, true, false);
      info.receiver.toTensorOp = findToTensorAfter(block, op);
      LDBG("Receiver chain (CUBE): convert_layout, flag=" << originalFlag
                                                          << ".");
    }
  }

  return 0;
}

/// Build TransferGroupInfo for a single transfer_id
static int buildTransferGroupData(int tid, const SmallVector<Operation *> &ops,
                                  FlagIdManager &flagIdMgr,
                                  TransferGroupInfo &info) {
  info.tid = tid;

  LDBG("Building group tid=" << tid << ", ops=" << ops.size() << ".");

  // 1. Determine original flag
  for (Operation *op : ops) {
    if ((isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op))) {
      int f = getFlagFromSyncOp(op);
      if (f >= 0) {
        info.originalFlag = f;
        break;
      }
    }
  }

  // 2. Collect extra sync (parent has no main_loop)
  ExtraSyncInfo extraInfo;
  if (collectExtraSync(ops, info.originalFlag, extraInfo)) {
    return -1;
  }
  info.extraSyncSetOp = extraInfo.setOp;
  info.extraSyncWaitOp = extraInfo.waitOp;
  if (extraInfo.setOp && extraInfo.waitOp) {
    LDBG("Extra sync: set(block_id=" << getBlockId(extraInfo.setOp)
                                     << "), wait(block_id="
                                     << getBlockId(extraInfo.waitOp) << ".");
  } else {
    LDBG("Extra sync: not found.");
  }

  // 3. Collect transfer chain (parent has main_loop)
  TransferChainInfo chainInfo;
  if (collectTransferChains(ops, info.originalFlag, chainInfo)) {
    return -1;
  }
  info.senderChain = chainInfo.sender;
  info.receiverChain = chainInfo.receiver;

  // 4. Determine direction
  if (info.senderChain.transferOp) {
    if (isa<hivm::FixpipeOp>(info.senderChain.transferOp)) {
      info.isCtoV = true;
    } else if (isa<hivm::CopyOp>(info.senderChain.transferOp)) {
      info.isCtoV = false;
    }
  }

  // 5. Collect buffer alloc/mark pairs from transfer ops
  //    Must run after transfer chain collection to identify the correct
  //    cross-core buffer (ub/cbuf) from each transfer op's operands,
  //    ignoring local buffers (e.g. cc on CUBE side).
  if (collectBufferAllocs(ops, info)) {
    return -1;
  }

  // 6. Acquire output flag
  for (int attempt = 0; attempt < kMaxFlagAttempts; ++attempt) {
    int64_t pf = flagIdMgr.acquireId();
    if (pf == FlagIdManager::INVALID_FLAG_ID) {
      break;
    }
    if (pf != info.originalFlag) {
      info.outputFlag = static_cast<int>(pf);
      break;
    }
  }

  if (info.senderChain.transferOp || info.receiverChain.transferOp) {
    LDBG("Direction: " << (info.isCtoV ? "C→V" : "V→C")
                       << ", flag=" << info.originalFlag
                       << ", outputFlag=" << info.outputFlag << ".");
  }

  return 0;
}

/// Collect TransferGroupInfo for all transfer groups
static int collectTransferGroupData(
    ModuleOp module, DenseMap<int, SmallVector<Operation *>> &opsByTid,
    FlagIdManager &flagIdMgr, DenseMap<int, TransferGroupInfo> &groups) {
  for (auto &p : opsByTid) {
    TransferGroupInfo info;
    if (buildTransferGroupData(p.first, p.second, flagIdMgr, info)) {
      continue;
    }
    if (info.senderChain.transferOp || info.receiverChain.transferOp) {
      groups[p.first] = info;
    }
  }

  // Output flag reuse: groups with same (originalFlag, direction) share an
  // output flag
  std::map<std::pair<int, bool>, int> outputFlagByKey;
  for (auto &p : groups) {
    auto &g = p.second;
    auto key = std::make_pair(g.originalFlag, g.isCtoV);
    auto it = outputFlagByKey.find(key);
    if (it != outputFlagByKey.end()) {
      g.outputFlag = it->second;
      LDBG("Group tid=" << g.tid << " reuses outputFlag=" << g.outputFlag
                        << " (shared originalFlag=" << g.originalFlag << ").");
    } else {
      outputFlagByKey[key] = g.outputFlag;
      LDBG("Group tid=" << g.tid
                        << " gets new shared outputFlag=" << g.outputFlag
                        << " for originalFlag=" << g.originalFlag << ".");
    }
  }

  return 0;
}

// ============================================================================
// Step 2: Create output buffers
// ============================================================================

static constexpr int kMaxTcbSearch = 100;

static int allocateNewTcbId(int startFrom, std::set<int> &usedTcbIds) {
  for (int id = startFrom; id < kMaxTcbSearch; ++id) {
    if (!usedTcbIds.count(id)) {
      usedTcbIds.insert(id);
      return id;
    }
  }
  return -1;
}

/// Create an output buffer for an input/output buffer pair
static int createOutputBufferPair(Operation *inputAllocOp, int tid, int tcbId,
                                  Value &inputBuffer, Value &outputBuffer,
                                  OpBuilder &builder, bool isSender) {
  if (!inputAllocOp) {
    return -1;
  }

  Location loc = builder.getUnknownLoc();

  inputBuffer = inputAllocOp->getResult(0);
  auto memRefType = dyn_cast<MemRefType>(inputBuffer.getType());
  if (!memRefType) {
    return -1;
  }

  int origBlockId = getBlockId(inputAllocOp);
  int outputBlockId = origBlockId;

  builder.setInsertionPointAfter(inputAllocOp);
  auto outputAlloc = builder.create<memref::AllocOp>(loc, memRefType);
  outputAlloc->setAttr(mlir::CVPipeline::kBlockId,
                       builder.getI32IntegerAttr(outputBlockId));
  outputAlloc->setAttr(mlir::CVPipeline::kTransferId,
                       builder.getI32IntegerAttr(tid));
  outputBuffer = outputAlloc.getResult();

  // NOTE: output alloc carries no ssbuffer.crossCoreDeps — alloc is a
  // buffer-creation op, not a behavior op. Producer tag lives on the
  // fixpipe/copy transferOp itself (pre-tagged by upstream passes such as
  // AddControlFlowCondition), and consumer tag also lives on the receiver
  // transferOp (memref.memory_space_cast / hivm.convert_layout). Do NOT
  // re-introduce crossDeps here.

  auto outputMark = builder.create<annotation::MarkOp>(loc, outputBuffer);
  outputMark->setAttr("effects", builder.getStrArrayAttr({"write", "read"}));
  outputMark->setAttr(mlir::CVPipeline::kBlockId,
                      builder.getI32IntegerAttr(outputBlockId));
  outputMark->setAttr(mlir::CVPipeline::kTransferId,
                      builder.getI32IntegerAttr(tid));
  outputMark->setAttr(
      "hivm.tightly_coupled_buffer",
      hivm::HIVMTightlyCoupledBufferAttr::get(builder.getContext(), tcbId));
  LDBG("Created " << (isSender ? "sender" : "receiver")
                  << " output buffer: block_id=" << outputBlockId
                  << ", tcb_id=" << tcbId << ".");
  return 0;
}

/// Create output buffers for a single transfer group.
///
/// With the select-chain refactor, we no longer create additional sync
/// ops for the output buffer — the existing wait/set ops (carrying the
/// original flag) cover the synchronization for both buffers. Only the
/// output alloc + mark are produced here.
///
/// For interCoreBufNum = N, creates N-1 extra output buffers per side,
/// all sharing the same tcb id. The original input buffer is element 0
/// of g.senderBuffers / g.receiverBuffers.
static int createOutputBuffersForGroup(TransferGroupInfo &g,
                                       OpBuilder &builder,
                                       int interCoreBufNum) {
  // First buffer is the original input alloc (kept as-is).
  g.senderBuffers.push_back(g.senderBuf.allocOp->getResult(0));
  for (int i = 1; i < interCoreBufNum; ++i) {
    Value inputBuf;
    Value outBuf;
    if (createOutputBufferPair(g.senderBuf.allocOp, g.tid, g.tcbId,
                               inputBuf, outBuf, builder,
                               /*isSender=*/true)) {
      return -1;
    }
    g.senderBuffers.push_back(outBuf);
  }

  // Same for receiver.
  g.receiverBuffers.push_back(g.receiverBuf.allocOp->getResult(0));
  for (int i = 1; i < interCoreBufNum; ++i) {
    Value inputBuf;
    Value outBuf;
    if (createOutputBufferPair(g.receiverBuf.allocOp, g.tid, g.tcbId,
                               inputBuf, outBuf, builder,
                               /*isSender=*/false)) {
      return -1;
    }
    g.receiverBuffers.push_back(outBuf);
  }
  return 0;
}

/// Create output buffers for all transfer groups
static int createOutputBuffers(DenseMap<int, TransferGroupInfo> &groups,
                               ModuleOp module, int interCoreBufNum) {
  OpBuilder builder(module.getContext());
  std::set<int> usedTcbIds;

  // Collect existing tcb ids
  module.walk([&](Operation *op) {
    if (auto tcbAttr = op->getAttrOfType<hivm::HIVMTightlyCoupledBufferAttr>(
            "hivm.tightly_coupled_buffer")) {
      auto id = tcbAttr.getId();
      if (id.has_value()) {
        LDBG("Found mark op with tcb_id=" << id.value() << ".");
        usedTcbIds.insert(id.value());
      }
    }
  });

  LDBG("=== Step 2: Creating output buffers ===.");
  {
    std::string ids;
    llvm::raw_string_ostream os(ids);
    for (int id : usedTcbIds)
      os << id << " ";
    LDBG("Collected existing tcb_ids: " << ids << ".");
  }

  int maxExistingTcbId = usedTcbIds.empty() ? 0 : *usedTcbIds.rbegin();
  LDBG("Max existing tcb_id: " << maxExistingTcbId << ".");

  int nextTcbId = maxExistingTcbId + 1;

  for (auto &p : groups) {
    TransferGroupInfo &g = p.second;
    LDBG("Group tid=" << g.tid << " (" << (g.isCtoV ? "C→V" : "V→C") << ").");

    g.tcbId = allocateNewTcbId(nextTcbId, usedTcbIds);
    LDBG("Allocated tcb_id=" << g.tcbId << ".");

    nextTcbId = g.tcbId + 1;

    createOutputBuffersForGroup(g, builder, interCoreBufNum);
  }
  return 0;
}

/// Tag consumer-side alloc and transferOp with crossDeps marks
static int addConsumerCrossDepsTags(TransferGroupInfo &g, ModuleOp module) {
  auto &consumerBuf = g.receiverBuf;
  auto &consumerChain = g.receiverChain;

  OpBuilder builder(module.getContext());

  if (consumerBuf.allocOp) {
    consumerBuf.allocOp->setAttr(
        mlir::CVPipeline::kCrossCoreDeps,
        builder.getArrayAttr(
            {builder.getI32IntegerAttr(g.tid), builder.getI32IntegerAttr(1)}));
  }
  if (consumerChain.transferOp) {
    consumerChain.transferOp->setAttr(
        mlir::CVPipeline::kCrossCoreDeps,
        builder.getArrayAttr(
            {builder.getI32IntegerAttr(g.tid), builder.getI32IntegerAttr(0)}));
  }
  return 0;
}

// ============================================================================
// Step 3: Add polling control flow
// ============================================================================

/// Set ssbuffer tags on an op
static int setSsbufferTags(Operation *op, OpBuilder &builder, int blockId,
                           int tid) {
  op->setAttr(mlir::CVPipeline::kBlockId, builder.getI32IntegerAttr(blockId));
  op->setAttr(mlir::CVPipeline::kTransferId, builder.getI32IntegerAttr(tid));
  return 0;
}

/// Ensure a WhileOp has an i32 iteration counter loop-carried variable.
/// Returns the counter Value; polling condition is (counter % 2) == 0.
/// Reuses an existing counter (e.g. one injected by InnerScope, detected via
/// ssbuffer.iterCounter); injects a new one only when absent.
static Value ensureWhileOpHasCounter(scf::WhileOp whileOp) {
  if (whileOp->hasAttr(CVPipeline::kIterCounter)) {
    Block &after = whileOp.getAfter().front();
    return after.getArgument(after.getNumArguments() - 1);
  }

  OpBuilder builder(whileOp);
  Location loc = whileOp.getLoc();
  auto oldWhile = whileOp;
  Type i32Type = builder.getI32Type();

  // Init counter = 0
  Value zero = builder.create<arith::ConstantIntOp>(loc, 0, 32);

  SmallVector<Value> newInits(oldWhile.getInits());
  newInits.push_back(zero);
  SmallVector<Type> newResultTypes(oldWhile.getResultTypes());
  newResultTypes.push_back(i32Type);

  Value counterIterArg;

  // Rebuild via the Builder callback API (matching InnerScope's
  // setupWhileIterArgCounter)
  auto newWhile = builder.create<scf::WhileOp>(
      loc, newResultTypes, newInits,
      [&](OpBuilder &bb, Location bl, ValueRange iterArgs) {
        Block *oldBefore = oldWhile.getBeforeBody();
        unsigned n = oldBefore->getNumArguments();
        IRMapping map;
        for (unsigned i = 0; i < n; ++i)
          map.map(oldBefore->getArgument(i), iterArgs[i]);

        for (Operation &op : oldBefore->without_terminator())
          bb.clone(op, map);

        auto oldCond = cast<scf::ConditionOp>(oldBefore->getTerminator());
        SmallVector<Value> condArgs;
        for (Value a : oldCond.getArgs())
          condArgs.push_back(map.lookupOrDefault(a));
        condArgs.push_back(iterArgs[n]); // counter
        bb.create<scf::ConditionOp>(
            bl, map.lookupOrDefault(oldCond.getCondition()), condArgs);
      },
      [&](OpBuilder &ab, Location al, ValueRange iterArgs) {
        Block *oldAfter = oldWhile.getAfterBody();
        unsigned n = oldAfter->getNumArguments();
        counterIterArg = iterArgs[n];
        IRMapping map;
        for (unsigned i = 0; i < n; ++i)
          map.map(oldAfter->getArgument(i), iterArgs[i]);

        for (Operation &op : oldAfter->without_terminator())
          ab.clone(op, map);

        auto oldYield = cast<scf::YieldOp>(oldAfter->getTerminator());
        Value one = ab.create<arith::ConstantIntOp>(al, 1, 32);
        Value nextCounter = ab.create<arith::AddIOp>(al, counterIterArg, one);
        SmallVector<Value> yOps;
        for (Value v : oldYield.getOperands())
          yOps.push_back(map.lookupOrDefault(v));
        yOps.push_back(nextCounter);
        ab.create<scf::YieldOp>(al, yOps);
      });

  // Copy attrs (must include ssbuffer.main_loop) and mark as processed
  for (auto attr : oldWhile->getAttrs())
    newWhile->setAttr(attr.getName(), attr.getValue());
  newWhile->setAttr(CVPipeline::kIterCounter, builder.getUnitAttr());

  // Replace results (exclude counter result)
  for (unsigned i = 0, e = oldWhile.getNumResults(); i < e; ++i)
    oldWhile.getResult(i).replaceAllUsesWith(newWhile.getResult(i));
  oldWhile.erase();

  return counterIterArg;
}

/// Build an arith.select chain that selects one of `buffers` based on
/// `counter % N` where N = buffers.size(). The result is a single Value
/// that can be used directly as the transferOp's output operand.
///
/// For N == 2:
///   %remsi = arith.remsi %counter, %c2
///   %cond0 = arith.cmpi eq, %remsi, %c0
///   return arith.select %cond0, %buffers[0], %buffers[1]
///
/// For N > 2:
///   Nested selects with cmpi(remsi, k) for k = 1..N-1, producing
///   %sel = select(cmp_eq_0, b0,
///                 select(cmp_eq_1, b1,
///                        ...select(cmp_eq_{N-1}, b_{N-1}, b_N))).
///
/// For N == 1: returns buffers[0] without emitting any op.
///
/// All emitted ops (except the constants) are tagged with
/// kBlockId / kTransferId via setSsbufferTags.
static Value buildBufferSelectChain(OpBuilder &builder, Location loc,
                                    Value counter, ArrayRef<Value> buffers,
                                    int blockId, int transferId) {
  assert(!buffers.empty() && "buildBufferSelectChain requires >=1 buffer");

  if (buffers.size() == 1) {
    return buffers.front();
  }

  // Counter may be index or any int width; arith.remsi / arith.cmpi require
  // both operands to share the type. Emit cN/ck with the counter's type
  // instead of hard-coding i32 (which would fail MLIR verification when
  // counter is i64 / index).
  Type counterTy = counter.getType();
  auto buildCounterConst = [&](int64_t v) -> Value {
    Operation *cOp = nullptr;
    if (counterTy.isIndex()) {
      cOp = builder.create<arith::ConstantIndexOp>(loc, v);
    } else {
      cOp = builder.create<arith::ConstantIntOp>(
          loc, v, counterTy.getIntOrFloatBitWidth());
    }
    setSsbufferTags(cOp, builder, blockId, transferId);
    return cOp->getResult(0);
  };

  Value cN = buildCounterConst(buffers.size());

  auto remsiOp = builder.create<arith::RemSIOp>(loc, counter, cN);
  setSsbufferTags(remsiOp, builder, blockId, transferId);
  Value remsi = remsiOp.getResult();

  // Build selects from the inside out: start with buffers[N-1] as the
  // fallback, then wrap each earlier buffer in a select(cmp_eq_k, buf_k,
  // prevSel).
  auto mkCondEq = [&](int k) -> Value {
    Value ck = buildCounterConst(k);
    auto cmpOp =
        builder.create<arith::CmpIOp>(loc, arith::CmpIPredicate::eq, remsi, ck);
    setSsbufferTags(cmpOp, builder, blockId, transferId);
    return cmpOp.getResult();
  };

  Value selected = buffers.back();
  // k = N-2 down to 1: each new layer wraps the prior with select(cmp_eq_k,
  // buf_k, prev). We start at N-2 because buffers[N-1] is already the
  // initial fallback (selected); emitting select(eq_{N-1}, buf_{N-1},
  // buf_{N-1}) would be a redundant op. k = 0 is emitted last as the
  // outermost select.
  for (int k = static_cast<int>(buffers.size()) - 2; k >= 1; --k) {
    Value cond = mkCondEq(k);
    auto newSelectedOp =
        builder.create<arith::SelectOp>(loc, cond, buffers[k], selected);
    setSsbufferTags(newSelectedOp, builder, blockId, transferId);
    selected = newSelectedOp.getResult();
  }
  Value cond0 = mkCondEq(0);
  auto resultOp =
      builder.create<arith::SelectOp>(loc, cond0, buffers[0], selected);
  setSsbufferTags(resultOp, builder, blockId, transferId);
  return resultOp.getResult();
}

/// Insert an arith.select chain before `transferOp` that selects one of
/// `buffers` based on `counter % buffers.size()`, then mutate
/// `transferOp`'s last operand to use the selected buffer. This replaces
/// the previous scf.if/clone approach with a single SSA value that flows
/// directly through the transferOp's existing operand. Existing wait/set
/// ops are left untouched.
///
/// `buffers` may have any size N (1, 2, 3, ...). For N == 1 the transferOp
/// is mutated to use the buffer directly; for N >= 2 an arith.select chain
/// selects among them.
///
/// Cross-core deps: producer transferOps arrive pre-tagged with
/// crossCoreDeps = [tid, 1] from upstream tagging (e.g.
/// AddControlFlowCondition); the same is true for receiver transferOps with
/// [tid, 0]. Since we do not clone or wrap, those attrs are preserved as-is on
/// the transferOp itself — no re-tagging needed.
static Operation *wrapTransferOpWithSelectChain(Operation *transferOp,
                                                Value counter,
                                                ArrayRef<Value> buffers,
                                                int bid, int tid,
                                                OpBuilder &builder) {
  OpBuilder::InsertionGuard guard(builder);
  builder.setInsertionPoint(transferOp);

  Location loc = transferOp->getLoc();

  Value selected =
      buildBufferSelectChain(builder, loc, counter, buffers, bid, tid);

  // Mutate the transferOp's last operand (outs for fixpipe/copy, source for
  // memory_space_cast/convert_layout which have a single operand).
  if (transferOp->getNumOperands() > 0) {
    transferOp->setOperand(transferOp->getNumOperands() - 1, selected);
  }

  return transferOp;
}

/// Receiver-side variant: the cross-core buffer flows through a chain of
/// ops (transferOp → memref.memory_space_cast → bufferization.to_tensor).
/// Since the buffer is now a single SSA Value (selected via arith.select),
/// all of those ops read from it directly without any cloning or
/// scf.if wrapping. The transferOp's existing crossCoreDeps = [tid, 0]
/// is preserved as-is.
static Operation *
wrapReceiverChainWithSelectChain(Operation *transferOp, Value counter,
                                 ArrayRef<Value> buffers, int bid, int tid,
                                 OpBuilder &builder) {
  return wrapTransferOpWithSelectChain(transferOp, counter, buffers, bid, tid,
                                       builder);
}

/// Process polling for a sender or receiver transfer chain.
///
/// With the select-chain refactor, this function no longer wraps any ops
/// in scf.if. Instead it inserts an arith.select chain before the
/// transferOp and mutates the transferOp's last operand to consume the
/// selected buffer. Existing wait/set ops are left in place — they
/// already carry the original (input) flag and the pass no longer
/// generates output-flag wait/set ops (see createOutputBuffersForGroup).
///
/// `buffers` may have any size N (1, 2, 3, ...); the same logic applies
/// for sender and receiver — the trailing ops (memory_space_cast /
/// to_tensor) on the receiver side read from the same SSA value selected
/// by the chain, so no special-casing is needed.
static int processTransferChain(TransferOpChain &chain, Value counter,
                                ArrayRef<Value> buffers,
                                OpBuilder &builder) {
  if (!chain.waitOp) {
    return -1;
  }

  if (chain.transferOp) {
    int bid = getBlockId(chain.transferOp);
    int tid = getTransferId(chain.transferOp);

    LDBG("transferOp: " << chain.transferOp->getName()
                        << ", bufferCount=" << buffers.size()
                        << ", hasToTensor="
                        << (chain.toTensorOp ? "yes" : "no") << ".");

    chain.transferOp = wrapReceiverChainWithSelectChain(
        chain.transferOp, counter, buffers, bid, tid, builder);
  }
  return 0;
}

/// Build a counter Value for a loop op (ForOp or WhileOp) and set the
/// insertion point for subsequent ops. The counter is the per-round
/// iteration counter that the select chain uses to choose between
/// inputBuffer and outputBuffer.
///
/// For scf.for: counter = iterVar / step (matches the previous cond
/// computation, where polling alternated per round rather than per
/// iteration).
///
/// For scf.while: the counter was injected as the last iter-arg by
/// `ensureWhileOpHasCounter` during preprocessing; we just look it up.
///
/// `builderOut` is positioned at the loop's terminator (or, for ForOp,
/// just before waitOp) so subsequent `buildBufferSelectChain` calls
/// emit ops at the right place.
static Value prepareLoopPolling(Operation *loopOp, Operation *waitOp,
                                OpBuilder &builderOut) {
  int bid = getBlockId(waitOp);
  int tid = getTransferId(waitOp);

  if (auto forOp = dyn_cast<scf::ForOp>(loopOp)) {
    OpBuilder counterBuilder(forOp.getBody(), Block::iterator(waitOp));
    Location loc = forOp.getLoc();
    Value iterVar = forOp.getInductionVar();
    Value step = forOp.getStep();
    auto divOp = counterBuilder.create<arith::DivSIOp>(loc, iterVar, step);
    setSsbufferTags(divOp.getOperation(), counterBuilder, bid, tid);
    Value counter = divOp.getResult();

    // Builder for subsequent select-chain ops: just before the forOp
    // terminator (yield).
    builderOut.setInsertionPoint(forOp.getBody()->getTerminator());
    return counter;
  }

  if (auto whileOp = dyn_cast<scf::WhileOp>(loopOp)) {
    Block &after = whileOp.getAfter().front();
    Value counter = after.getArgument(after.getNumArguments() - 1);
    builderOut.setInsertionPoint(after.getTerminator());
    return counter;
  }

  llvm_unreachable("unexpected loop op type");
}

/// Add polling control flow for all transfer groups
static int addPollingControlFlow(DenseMap<int, TransferGroupInfo> &groups) {
  for (auto &p : groups) {
    TransferGroupInfo &g = p.second;

    // Get sender's loop op (ForOp or WhileOp)
    Operation *senderWaitParent = g.senderChain.waitOp->getParentOp();

    // Prepare counter Value and builder for sender loop
    OpBuilder senderBuilder(senderWaitParent->getContext());
    Value senderCounter = prepareLoopPolling(
        senderWaitParent, g.senderChain.waitOp, senderBuilder);

    // Process sender chain
    if (processTransferChain(g.senderChain, senderCounter, g.senderBuffers,
                             senderBuilder) != 0) {
      return -1;
    }

    // Process receiver chain (may use different loop op)
    if (g.receiverChain.waitOp) {
      Operation *receiverWaitParent = g.receiverChain.waitOp->getParentOp();

      if (receiverWaitParent == senderWaitParent) {
        // Use the same counter and builder
        if (processTransferChain(g.receiverChain, senderCounter,
                                 g.receiverBuffers, senderBuilder) != 0) {
          return -1;
        }
      } else {
        // Receiver uses a different loop op, prepare new counter and builder
        OpBuilder receiverBuilder(receiverWaitParent->getContext());
        Value receiverCounter = prepareLoopPolling(
            receiverWaitParent, g.receiverChain.waitOp, receiverBuilder);
        if (processTransferChain(g.receiverChain, receiverCounter,
                                 g.receiverBuffers, receiverBuilder) != 0) {
          return -1;
        }
      }
    }
  }
  return 0;
}

// ============================================================================
// Preprocessing: inject iteration counter into WhileOps with main_loop
// ============================================================================

/// Inject an i32 iteration counter loop-carried variable into every WhileOp
/// that has main_loop and contains transfer_id ops. Must run BEFORE Step 1 so
/// subsequent data collection sees the already-modified IR.
static void preInjectWhileOpToggles(ModuleOp module) {
  SmallVector<scf::WhileOp> whileOps;
  module.walk([&](scf::WhileOp whileOp) {
    if (!CVPipeline::isMainLoopOp(whileOp))
      return;
    bool hasTransferOps = false;
    whileOp.walk([&](Operation *op) {
      if (op->hasAttr(mlir::CVPipeline::kTransferId)) {
        hasTransferOps = true;
        return WalkResult::interrupt();
      }
      return WalkResult::advance();
    });
    if (hasTransferOps)
      whileOps.push_back(whileOp);
  });

  for (auto whileOp : whileOps)
    ensureWhileOpHasCounter(whileOp);

  LDBG("Preprocessed " << whileOps.size()
                       << " WhileOps with toggle injection.");
}

// ============================================================================
// Pass entry point
// ============================================================================

void AddMultiBufferOuterScopePass::runOnOperation() {
  ModuleOp module = getOperation();

  if (CVPipeline::hasFallbackAttr(module)) {
    return;
  }

  LDBG("============================================================");
  LDBG("Enter AddMultiBufferOuterScope pass.");
  LDBG("============================================================");

  // Determine buffer mode early; only inject toggle for double-buffer
  int interCoreBufNum = BufferCountManager(module).getBufferCountByType(
      BufferCountManager::DepType::InterCore);
  bool isDoubleBuf = (interCoreBufNum > 1);
  LDBG("[BufferCount] interCoreBufNum=" << interCoreBufNum
                                        << " doubleBuf=" << isDoubleBuf << ".");

  // Preprocessing: inject iteration counter into WhileOps before data
  // collection (only needed for double-buffer polling)
  if (isDoubleBuf) {
    preInjectWhileOpToggles(module);
  }

  // Step 1: Collect transfer group information
  LDBG("[Step 1/3] Start: transfer group collection.");
  FlagIdManager flagIdMgr(module);
  DenseMap<int, SmallVector<Operation *>> opsByTid;
  collectOpsByTransferId(module, opsByTid);
  DenseMap<int, TransferGroupInfo> groups;
  if (collectTransferGroupData(module, opsByTid, flagIdMgr, groups)) {
    LDBG("FALLBACK: Step 1/3 failed, no valid transfer groups found, rc="
         << CVPipeline::ERRCODE_FAILED << ".");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
    return;
  }
  LDBG("[Step 1/3] Done: " << groups.size() << " transfer groups.");

  // Flag ID budget:
  // 1. Usable ids 0..MAX_FLAG_ID (14); kReservedPipeFlagId (15) is reserved.
  // 2. Final max id = largest output flag acquired in Step 1.
  // 3. Exceed budget -> keep single-buffer mode.
  // 4. Input flag > kReservedPipeFlagId -> fallback rc=2.
  std::set<int> usedFlags;
  module.walk([&](Operation *op) {
    if (isa<hivm::SyncBlockSetOp>(op) || isa<hivm::SyncBlockWaitOp>(op)) {
      int f = getFlagFromSyncOp(op);
      if (f >= 0)
        usedFlags.insert(f);
    }
  });
  // Input flags:
  // 1. Flag id > kReservedPipeFlagId not producible by this pass.
  // 2. kReservedPipeFlagId (pipe) itself is allowed.
  bool inputOverBudget = false;
  for (int f : usedFlags) {
    if (f > kReservedPipeFlagId)
      inputOverBudget = true;
  }
  if (inputOverBudget) {
    LDBG("FALLBACK: FlagBudget, input flag id > "
         << kReservedPipeFlagId << ", rc=" << CVPipeline::ERRCODE_IGNORED
         << ".");
    CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_IGNORED);
    return;
  }
  if (isDoubleBuf) {
    int maxOutputFlag = -1;
    for (auto &p : groups) {
      if (p.second.outputFlag > maxOutputFlag)
        maxOutputFlag = p.second.outputFlag;
    }
    LDBG("[FlagBudget] maxOutputFlag=" << maxOutputFlag
                                       << " (usable flag ids 0.."
                                       << FlagIdManager::MAX_FLAG_ID << ").");
    if (maxOutputFlag > FlagIdManager::MAX_FLAG_ID) {
      LDBG("FALLBACK: FlagBudget, estimated flag id "
           << maxOutputFlag << " exceeds usable range (0.."
           << FlagIdManager::MAX_FLAG_ID
           << "), fallback to single-buffer mode.");
      isDoubleBuf = false;
    }
  }

  if (isDoubleBuf) {
    // Tag llvm.load/store volatile ops with crossDeps
    DenseMap<int, SmallVector<Operation *>> loadStoreByTid;
    collectLoadStoreOpsByTransferId(module, loadStoreByTid);
    tagLoadStoreOpsWithCrossDeps(loadStoreByTid);
  }

  if (isDoubleBuf) {
    LDBG("[Step 2/3] Start: output buffer creation.");
    if (createOutputBuffers(groups, module, interCoreBufNum)) {
      LDBG("FALLBACK: Step 2/3 failed, output buffer creation failed, rc="
           << CVPipeline::ERRCODE_FAILED << ".");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
    LDBG("[Step 2/3] Done.");

    LDBG("[Step 3/3] Start: polling control flow.");
    if (addPollingControlFlow(groups)) {
      LDBG("FALLBACK: Step 3/3 failed, polling control flow failed, rc="
           << CVPipeline::ERRCODE_FAILED << ".");
      CVPipeline::setFallbackAttr(module, CVPipeline::ERRCODE_FAILED);
      return;
    }
    LDBG("[Step 3/3] Done.");
  } else {
    LDBG("[Step 2-3] Skipped (single-buffer mode).");
  }

  LDBG("============================================================");
  LDBG("Exit AddMultiBufferOuterScope pass.");
  LDBG("============================================================");
}

std::unique_ptr<OperationPass<ModuleOp>> createAddMultiBufferOuterScopePass() {
  return std::make_unique<AddMultiBufferOuterScopePass>();
}

void AddMultiBufferOuterScopePass::getDependentDialects(
    DialectRegistry &registry) const {
  registry
      .insert<mlir::annotation::AnnotationDialect, mlir::memref::MemRefDialect,
              mlir::bufferization::BufferizationDialect,
              mlir::arith::ArithDialect, mlir::scf::SCFDialect,
              mlir::hivm::HIVMDialect, mlir::scope::ScopeDialect>();
}

void registerAddMultiBufferOuterScopePasses() {
  registerPass([]() -> std::unique_ptr<mlir::Pass> {
    return createAddMultiBufferOuterScopePass();
  });
}

} // namespace triton
} // namespace mlir
