// RUN: triton-opt --help | FileCheck %s --check-prefix=REGISTERED
// RUN: ! triton-opt --analyze-name %s

// REGISTERED: --analyze-name

module {
  func.func @bmm_kernel() {
    return
  }
}
