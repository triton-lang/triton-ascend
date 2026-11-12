#include "SpecializedMemoryPublicCases.h"
#include <iostream>
int main() {
  const auto checks = mlir::ascend::specialized_test::publicChecks();
  std::cout << checks << " public checks passed\n";
  return checks == 34 ? 0 : 1;
}
