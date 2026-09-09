//===- CostModelLogger.cpp - Costmodel execution tracing --------*- C++ -*-===//

#include "AscendModel/Support/CostModelLogger.h"
#include "mlir/IR/Operation.h"

#include "llvm/Support/raw_ostream.h"

#include <cstdlib>
#include <mutex>

using namespace mlir;
using namespace mlir::ascend::costmodel;

namespace {

// The whole costmodel pipeline runs on one thread per module, but keep the
// trace bookkeeping under a mutex so concurrent compilations cannot
// interleave a single trace scope.
std::mutex &traceMutex() {
  static std::mutex mutex;
  return mutex;
}

int resolveLogLevelFromEnvironment() {
  const char *environment = std::getenv("COSTMODEL_LOG_LEVEL");
  if (!environment || !*environment)
    return 0; // unset: silent, keeps production stderr clean
  llvm::StringRef value(environment);
  if (value == "0" || value.equals_insensitive("off"))
    return 0;
  if (value == "2" || value.equals_insensitive("verbose") ||
      value.equals_insensitive("debug"))
    return 2;
  return 1;
}

int &logLevelRef() {
  static int level = resolveLogLevelFromEnvironment();
  return level;
}

unsigned &traceDepthRef() {
  static unsigned depth = 0;
  return depth;
}

llvm::raw_ostream &writePrefix(llvm::raw_ostream &os) {
  os << "[COSTMODEL] ";
  for (unsigned i = 0; i < traceDepthRef(); ++i)
    os << "  ";
  return os;
}

} // namespace

namespace mlir::ascend::costmodel {

int logLevel() { return logLevelRef(); }
bool logEnabled() { return logLevel() >= 1; }
bool debugEnabled() { return logLevel() >= 1; }

// The one place that owns the log backend.  Swap llvm::errs() here (e.g. for
// a raw_fd_ostream or a captured callback stream) to redirect all costmodel
// logging; no call site needs to change.
llvm::raw_ostream &costModelSink() { return llvm::errs(); }

llvm::raw_ostream &costModelLog() {
  if (!logEnabled())
    return llvm::nulls();
  return writePrefix(costModelSink());
}

llvm::raw_ostream &costModelDebug() {
  if (!debugEnabled())
    return llvm::nulls();
  return writePrefix(costModelSink());
}

void costModelLogLine(llvm::StringRef text) {
  if (!logEnabled())
    return;
  std::lock_guard<std::mutex> lock(traceMutex());
  writePrefix(costModelSink()) << text << "\n";
}

void costModelDebugLine(llvm::StringRef text) {
  if (!debugEnabled())
    return;
  std::lock_guard<std::mutex> lock(traceMutex());
  writePrefix(costModelSink()) << text << "\n";
}

void costModelLogIR(llvm::StringRef tag, Operation *op) {
  if (!logEnabled() || !op)
    return;
  std::lock_guard<std::mutex> lock(traceMutex());
  llvm::raw_ostream &os = costModelSink();
  writePrefix(os) << "===== IR dump: " << tag << " =====\n";
  op->print(os);
  os << "\n";
  writePrefix(os) << "===== IR dump end =====\n";
}

CostModelTraceScope::CostModelTraceScope(llvm::StringRef name, bool active)
    : name(name.str()), active(active && logLevel() >= 1) {
  if (!this->active)
    return;
  std::lock_guard<std::mutex> lock(traceMutex());
  writePrefix(costModelSink()) << ">>> " << this->name << "\n";
  ++traceDepthRef();
}

CostModelTraceScope::~CostModelTraceScope() {
  if (!active)
    return;
  std::lock_guard<std::mutex> lock(traceMutex());
  --traceDepthRef();
  writePrefix(costModelSink()) << "<<< " << name << "\n";
}

} // namespace mlir::ascend::costmodel
