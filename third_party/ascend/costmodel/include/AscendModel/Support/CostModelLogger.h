//===- CostModelLogger.h - Costmodel execution tracing ----------*- C++ -*-===//
//
// Lightweight tracing for the Ascend SIMD/SIMT cost model.  Every log call
// site goes through costModelSink(), so the current llvm::errs() backend can
// be swapped for another stream (file, diagnostic engine, callback) by
// editing that single function.
//
// Log levels, resolved once from the COSTMODEL_LOG_LEVEL environment
// variable:
//   0 / off      - silent (also the unset default)
//   1 / default  - full detail: interface call chain, key inputs/outputs,
//                  IR snapshots, stage partition detail, and per-stage cost
//                  formulas
//   2 / verbose  - alias of 1 (kept for compatibility)
//
// Usage:
//   void someInterface(ModuleOp module) {
//     COSTMODEL_TRACE("someInterface");            // >>> / <<< call chain
//     costModelLog() << "anchors=" << n << "\n";   // level >= 1
//     costModelLogIR("tag", module);               // level >= 1 IR snapshot
//     COSTMODEL_TRACE_DEBUG("helper");             // level >= 2
//   }
//
//===----------------------------------------------------------------------===//

#ifndef ASCENDMODEL_SUPPORT_COSTMODELLOGGER_H
#define ASCENDMODEL_SUPPORT_COSTMODELLOGGER_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/raw_ostream.h"

#include <string>

namespace mlir {
class Operation;
}

namespace mlir::ascend::costmodel {

/// Resolved log level: 0 = off, 1 = default, 2 = verbose.  Read once from
/// COSTMODEL_LOG_LEVEL; unset or unrecognized values mean 0 (silent) so
/// production compilation output stays clean.
int logLevel();
bool logEnabled();   // level >= 1
bool debugEnabled(); // level >= 2

/// Single owner of the output stream.  Currently llvm::errs(); replace this
/// implementation to redirect logging without touching any call site.
llvm::raw_ostream &costModelSink();

/// Default-level stream (level >= 1).  Writes the "[COSTMODEL]" prefix plus
/// current trace indentation, then returns the sink; returns llvm::nulls()
/// when disabled so call sites need no guard.
llvm::raw_ostream &costModelLog();

/// Verbose-level stream (level >= 2), same prefixing rules as costModelLog().
llvm::raw_ostream &costModelDebug();

/// One-line helper: prefix + text + newline at default level.
void costModelLogLine(llvm::StringRef text);
/// One-line helper at verbose level.
void costModelDebugLine(llvm::StringRef text);

/// Print an IR snapshot between banner separators at default level:
///   ===== IR dump: <tag> =====
///   <op IR>
///   ===== IR dump end =====
void costModelLogIR(llvm::StringRef tag, Operation *op);

/// RAII trace scope.  Prints ">>> name" on entry and "<<< name" on exit,
/// maintaining indentation so the printed log mirrors the call tree.  No-op
/// when inactive.
class CostModelTraceScope {
public:
  CostModelTraceScope(llvm::StringRef name, bool active);
  ~CostModelTraceScope();
  CostModelTraceScope(const CostModelTraceScope &) = delete;
  CostModelTraceScope &operator=(const CostModelTraceScope &) = delete;

private:
  std::string name;
  bool active;
};

} // namespace mlir::ascend::costmodel

// Hoist the logging helpers into mlir::ascend so call sites living in (or
// using) that namespace can invoke them unqualified.
namespace mlir::ascend {
using costmodel::costModelDebug;
using costmodel::costModelDebugLine;
using costmodel::costModelLog;
using costmodel::costModelLogIR;
using costmodel::costModelLogLine;
using costmodel::costModelSink;
using costmodel::debugEnabled;
using costmodel::logEnabled;
using costmodel::logLevel;
} // namespace mlir::ascend

#define COSTMODEL_TRACE_PRIVATE_CONCAT2(a, b) a##b
#define COSTMODEL_TRACE_PRIVATE_CONCAT(a, b)                                    \
  COSTMODEL_TRACE_PRIVATE_CONCAT2(a, b)
#define COSTMODEL_TRACE_PRIVATE_VAR(line)                                       \
  COSTMODEL_TRACE_PRIVATE_CONCAT(costModelTraceScope, line)

/// Interface-level trace, active at log level >= 1.
#define COSTMODEL_TRACE(name)                                                   \
  ::mlir::ascend::costmodel::CostModelTraceScope COSTMODEL_TRACE_PRIVATE_VAR(   \
      __LINE__)(name, ::mlir::ascend::costmodel::logEnabled())

/// Internal helper trace, active at log level >= 2 only.
#define COSTMODEL_TRACE_DEBUG(name)                                             \
  ::mlir::ascend::costmodel::CostModelTraceScope COSTMODEL_TRACE_PRIVATE_VAR(   \
      __LINE__)(name, ::mlir::ascend::costmodel::debugEnabled())

#endif // ASCENDMODEL_SUPPORT_COSTMODELLOGGER_H
