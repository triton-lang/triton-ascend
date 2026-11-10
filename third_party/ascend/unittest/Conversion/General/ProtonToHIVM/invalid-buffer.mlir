// REQUIRES: proton
// RUN: not triton-opt %s --lower-ascend-proton-to-hivm='data-segment-bytes=7' 2>&1 | FileCheck %s

module {}

// CHECK: error: Ascend Proton data-segment-bytes must be >= 8 and divisible by 8
