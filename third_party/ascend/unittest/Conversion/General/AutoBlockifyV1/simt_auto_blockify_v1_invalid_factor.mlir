// RUN: not triton-opt --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=3" %s 2>&1 | FileCheck %s --check-prefix=F3
// RUN: not triton-opt --ta-simt-auto-blockify-v1="physical-vector-core-count=64 superblock-factor=64" %s 2>&1 | FileCheck %s --check-prefix=F64

// F3: superblock-factor must be one of 1, 2, 4, 8, 16 or 32, got 3
// F64: superblock-factor must be one of 1, 2, 4, 8, 16 or 32, got 64

module {
  tt.func public @invalid_factor(%arg0: !tt.ptr<f32>) {
    %pid = tt.get_program_id x : i32
    %ptr = tt.addptr %arg0, %pid : !tt.ptr<f32>, i32
    %zero = arith.constant 0.0 : f32
    tt.store %ptr, %zero : !tt.ptr<f32>
    tt.return
  }
}
