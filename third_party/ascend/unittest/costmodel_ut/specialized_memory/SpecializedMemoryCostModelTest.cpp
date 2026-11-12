#include "SpecializedMemoryMainCases.h"
#include "SpecializedMemoryPublicCases.h"
#include "gtest/gtest.h"
TEST(SpecializedMemory, PublicGeometryContract) {
  EXPECT_EQ(mlir::ascend::specialized_test::publicChecks(), 34u);
}
TEST(SpecializedMemory, CompleteFragmentCompilerProof) {
  EXPECT_TRUE(mlir::ascend::specialized_test::compilerChecks());
}
