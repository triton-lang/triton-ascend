#include "AscendModel/RouteModel/Reduction/ReductionCostModel.h"
#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "mlir/Dialect/Arith/IR/Arith.h"
#include "mlir/Dialect/Func/IR/FuncOps.h"
#include "mlir/Parser/Parser.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/raw_ostream.h"
#include <gtest/gtest.h>
using namespace mlir::ascend;

namespace {
class TemporaryReductionProfile {
public:
  explicit TemporaryReductionProfile(llvm::StringRef contents) {
    int fd = -1;
    auto error = llvm::sys::fs::createTemporaryFile("reduction-profile", "json",
                                                    fd, path);
    if (error)
      llvm::report_fatal_error("cannot create profile test file");
    llvm::raw_fd_ostream stream(fd, true);
    stream << contents;
  }
  ~TemporaryReductionProfile() { llvm::sys::fs::remove(path); }
  llvm::StringRef name() const { return path; }

private:
  llvm::SmallString<128> path;
};
} // namespace

TEST(ReductionProfileTest, DedicatedProfilePricesScalarReduction) {
  auto loaded =
      ReductionProfile::loadFromFile(TRITON_ASCEND_REDUCTION_TEST_PROFILE_PATH);
  ASSERT_TRUE(bool(loaded));
  ReductionWorkload work;
  work.kind = "and";
  work.dataType = "i32";
  work.shape = {64, 128};
  work.axis = 1;
  auto cycles = estimateTailAxisReductionCycles(
      work, loaded->getModeProfile(ReductionMode::SIMD), ReductionMode::SIMD,
      4);
  ASSERT_TRUE(cycles.has_value());
  EXPECT_DOUBLE_EQ(*cycles, 12.0 * 64 * 127);
  EXPECT_FALSE(loaded->getContentHash().empty());
}

TEST(ReductionProfileTest, RejectsWrongUnitAndMalformedCandidate) {
  TemporaryReductionProfile wrongUnit(
      R"({"schema_version":1,"profile_version":"test","target":"test","unit":"AI_core_cycle","families":{"tail_axis":{"simd":{},"simt":{}}}})");
  auto unit = ReductionProfile::loadFromFile(wrongUnit.name());
  EXPECT_FALSE(bool(unit));
  llvm::consumeError(unit.takeError());
  auto value = llvm::json::parse(R"({"rn_candidate_sum_f32":[0,1,2]})");
  ASSERT_TRUE(bool(value));
  auto candidate = parseReductionParameters(*value->getAsObject());
  EXPECT_FALSE(bool(candidate));
  llvm::consumeError(candidate.takeError());
}

TEST(ReductionProfileTest, MissingParametersKeepLegacyFallbackAvailable) {
  TemporaryReductionProfile empty(
      R"({"schema_version":1,"profile_version":"test","target":"test","unit":"syscnt_cycle","families":{"tail_axis":{"simd":{},"simt":{}}}})");
  auto loaded = ReductionProfile::loadFromFile(empty.name());
  ASSERT_TRUE(bool(loaded));
  ReductionWorkload work;
  work.kind = "sum";
  work.dataType = "f32";
  work.shape = {4, 64};
  work.axis = 1;
  EXPECT_FALSE(estimateTailAxisReductionCycles(
      work, loaded->getModeProfile(ReductionMode::SIMD), ReductionMode::SIMD,
      4));
  EXPECT_DOUBLE_EQ(legacyReductionShuffleCycles(work, 32), 4.0 * 64 * 6 / 32);
}

TEST(ReductionProfileTest, ContentHashIncludesCoefficientChanges) {
  TemporaryReductionProfile first(
      R"({"schema_version":1,"profile_version":"test","target":"test","unit":"syscnt_cycle","families":{"tail_axis":{"simd":{"rn_linear_and_i32":[24,12]},"simt":{}}}})");
  TemporaryReductionProfile second(
      R"({"schema_version":1,"profile_version":"test","target":"test","unit":"syscnt_cycle","families":{"tail_axis":{"simd":{"rn_linear_and_i32":[24,13]},"simt":{}}}})");
  auto a = ReductionProfile::loadFromFile(first.name());
  auto b = ReductionProfile::loadFromFile(second.name());
  ASSERT_TRUE(bool(a));
  ASSERT_TRUE(bool(b));
  EXPECT_NE(a->getContentHash(), b->getContentHash());
}

TEST(ReductionProfileTest, SelectionLoadsDedicatedParameters) {
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      "module { func.func @add(%x: i32, %y: i32) -> i32 { %z = arith.addi %x, "
      "%y : i32 return %z : i32 } }",
      &context);
  ASSERT_TRUE(bool(module));
  auto report = analyzeSimdSimtCandidates(*module);
  if (!report)
    FAIL() << llvm::toString(report.takeError());
  EXPECT_EQ(report->reductionProfileStatus, "loaded");
  EXPECT_FALSE(report->reductionProfileContentSha256.empty());
}

TEST(ReductionProfileTest, MissingExternalFileDoesNotRejectSelection) {
  auto owner = getDefaultSimdSimtProfilePath();
  auto file = llvm::MemoryBuffer::getFile(owner);
  ASSERT_TRUE(bool(file));
  auto json = llvm::json::parse(file.get()->getBuffer());
  ASSERT_TRUE(bool(json));
  auto *root = json->getAsObject();
  ASSERT_NE(root, nullptr);
  llvm::SmallString<256> microbenchmark(owner);
  llvm::sys::path::remove_filename(microbenchmark);
  llvm::sys::path::append(microbenchmark,
                          *root->getString("microbenchmark_profile"));
  (*root)["microbenchmark_profile"] = microbenchmark.str().str();
  (*root)["reduction_profile"] = "missing-reduction.json";
  std::string text;
  llvm::raw_string_ostream stream(text);
  stream << *json;
  stream.flush();
  TemporaryReductionProfile modified(text);
  mlir::MLIRContext context;
  context.loadDialect<mlir::func::FuncDialect, mlir::arith::ArithDialect>();
  auto module = mlir::parseSourceString<mlir::ModuleOp>(
      "module { func.func @add(%x: i32, %y: i32) -> i32 { %z = arith.addi %x, "
      "%y : i32 return %z : i32 } }",
      &context);
  ASSERT_TRUE(bool(module));
  SimdSimtCostModelOptions options;
  options.profilePath = modified.name().str();
  auto report = analyzeSimdSimtCandidates(*module, options);
  if (!report)
    FAIL() << llvm::toString(report.takeError());
  EXPECT_TRUE(llvm::StringRef(report->reductionProfileStatus)
                  .starts_with("unavailable:"));
  EXPECT_TRUE(report->reductionProfileContentSha256.empty());
}
