//===- CostModelError.h - Typed cost-model failures ------------*- C++ -*-===//

#ifndef ASCENDMODEL_SUPPORT_COSTMODELERROR_H
#define ASCENDMODEL_SUPPORT_COSTMODELERROR_H

#include "llvm/ADT/StringRef.h"
#include "llvm/Support/Error.h"
#include "llvm/Support/raw_ostream.h"

#include <string>
#include <system_error>

namespace mlir::ascend {

/// An input kernel that is valid TTIR but is outside the model/materializer
/// contract.  The selector may recover from this by using backend_default.
class UnsupportedCostModelIR final
    : public llvm::ErrorInfo<UnsupportedCostModelIR> {
public:
  static inline char ID = 0;

  explicit UnsupportedCostModelIR(std::string message)
      : message(std::move(message)) {}

  void log(llvm::raw_ostream &os) const override { os << message; }
  std::error_code convertToErrorCode() const override {
    return std::make_error_code(std::errc::not_supported);
  }
  llvm::StringRef getMessage() const { return message; }

private:
  std::string message;
};

} // namespace mlir::ascend

#endif // ASCENDMODEL_SUPPORT_COSTMODELERROR_H
