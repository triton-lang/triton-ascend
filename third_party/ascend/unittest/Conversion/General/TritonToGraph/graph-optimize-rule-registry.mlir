// RUN: triton-opt --verify-each %s -graph-optimize='rule-mask=0' -o - | FileCheck %s --check-prefix=NOOP
// RUN: triton-opt --verify-each %s -graph-optimize='rule-mask=65535' -o - | FileCheck %s --check-prefix=NOOP
// RUN: triton-opt --debug-only=graph-optimize --verify-each %s -graph-optimize -o /dev/null 2>&1 | FileCheck %s --check-prefix=DEFAULT-OFF --allow-empty
// RUN: not triton-opt --verify-each %s -graph-optimize='rule-mask=65536' -o - 2>&1 | FileCheck %s --check-prefix=UNKNOWN
// RUN: not triton-opt --verify-each %s -graph-optimize='rule-mask=4294967296' -o - 2>&1 | FileCheck %s --check-prefix=OUT-OF-RANGE

// The aggregate mask validates the complete registered-bit set, while zero
// keeps the legacy no-op path covered.
// NOOP-LABEL: tt.func @registry_noop(
// NOOP: tt.return
// DEFAULT-OFF-NOT: IndependentAxisTensorizeRule
// DEFAULT-OFF-NOT: StaticProgramAxisFusionRule
// DEFAULT-OFF-NOT: PersistentTaskStripMiningRule
// DEFAULT-OFF-NOT: ResidentLoadForwardingRule
// DEFAULT-OFF-NOT: IntermediatePrecisionBoundaryElisionRule
// DEFAULT-OFF-NOT: StoreCoveragePlanningRule
// DEFAULT-OFF-NOT: ContiguousBlockAccessFormationRule
tt.func @registry_noop(%value: i32) -> i32 {
  tt.return %value : i32
}

// UNKNOWN: graph-optimize rule-mask contains unknown or out-of-range bits: 65536
// OUT-OF-RANGE: graph-optimize rule-mask contains unknown or out-of-range bits: 4294967296
