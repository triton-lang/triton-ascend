// RUN: triton-opt %s | FileCheck %s
// CHECK-LABEL: func.func @inline_asm
// CHECK: llvm.inline_asm "nop", "~{dirflag}" : () -> ()
// CHECK: return
// CHECK-NOT: tail_call_kind

func.func @inline_asm() {
  llvm.inline_asm "nop", "~{dirflag}" : () -> ()
  return
}
