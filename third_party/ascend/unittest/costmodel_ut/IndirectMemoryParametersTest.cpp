#include "AscendModel/RouteModel/Models/IndirectGatherMemoryCostModel.h"
#include "AscendModel/RouteModel/SimdSimtCostModel.h"
#include "IndirectMemoryTestUtils.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/FileUtilities.h"
#include "llvm/Support/JSON.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/Path.h"
#include <gtest/gtest.h>

using namespace mlir;
using namespace mlir::ascend;

TEST(IndirectMemoryParametersTest, RejectsInvalidAssets) {
  llvm::SmallString<256> source(getDefaultSimdSimtProfilePath()), temporary;
  llvm::sys::path::remove_filename(source);
  llvm::sys::path::append(source, "../indirect_memory/ascend_950pr_v1.json");
  auto buffer = llvm::MemoryBuffer::getFile(source);
  ASSERT_TRUE(bool(buffer));
  int fd;
  ASSERT_FALSE(llvm::sys::fs::createTemporaryFile("indirect-parameters", "json",
                                                  fd, temporary));
  llvm::sys::fs::closeFile(fd);
  llvm::FileRemover cleanup(temporary);
  for (int variant = 0; variant < 9; ++variant) {
    auto value = llvm::cantFail(llvm::json::parse((*buffer)->getBuffer()));
    auto &root = *value.getAsObject();
    auto &parameters = *root.getObject("parameters");
    switch (variant) {
    case 0:
      root["schema_version"] = 2;
      break;
    case 1:
      root["unit"] = "aicore_cycle";
      break;
    case 2:
      parameters.erase("native_chain_load");
      break;
    case 3:
      parameters.getArray("native_chain_load")->pop_back();
      break;
    case 4:
      (*parameters.getArray("native_chain_load"))[0] = "not-a-number";
      break;
    case 5:
      (*parameters.getArray("mask_basis"))[0] = 2;
      break;
    case 6:
      (*parameters.getArray("completeLoad8Version"))[0] = 5;
      break;
    case 7:
      (*parameters.getArray("completeLoad16Roots"))[0] = -1;
      break;
    case 8:
      (*parameters.getArray("completeLoad16Nodes"))[2] = 0;
      break;
    }
    std::error_code error;
    {
      llvm::raw_fd_ostream out(temporary, error);
      out << value;
    }
    ASSERT_FALSE(error);
    auto result = IndirectMemoryParameters::load(temporary);
    EXPECT_FALSE(bool(result)) << variant;
    if (!result)
      llvm::consumeError(result.takeError());
  }
}

TEST(IndirectMemoryParametersTest, DefaultSimdSelection) {
  std::string root = __FILE__;
  root.resize(root.rfind("/unittest/"));
  for (const char *file :
       {"david_v100_simd_simt_v1.json", "ascend_950pr_simd_simt_v1.json"}) {
    auto buffer = llvm::MemoryBuffer::getFile(
        root + "/costmodel/profiles/simd_simt/" + file);
    ASSERT_TRUE(bool(buffer));
    auto parsed = llvm::json::parse((*buffer)->getBuffer());
    ASSERT_TRUE(bool(parsed));
    auto *profile = parsed->getAsObject();
    ASSERT_NE(profile, nullptr);
    for (bool store : {false, true})
      for (int mask : {0, 1, 2}) {
        Memory t(store, 1, "f32", mask);
        ASSERT_TRUE(t.module);
        (store ? t.stage.workload.indirectStoreTransactions
               : t.stage.workload.indirectLoadTransactions) = 128;
        auto selector = profile->getString(store ? "simd_indirect_store_model"
                                                 : "simd_indirect_load_model");
        ASSERT_TRUE(selector);
        (store ? t.profile.simdIndirectStoreModel
               : t.profile.simdIndirectLoadModel) = selector->str();
        auto result = IndirectGatherMemoryCostModel().cost(
            t.stage, t.profile, {StageMode::SIMD, 1});
        EXPECT_EQ(result.nativeSimdEnvelope, mask != 0);
        EXPECT_STREQ(store ? result.storePricing : result.loadPricing,
                     mask    ? "simd_mask_envelope_20261019"
                     : store ? "random_store_no_fill_first_20261008"
                             : "random_dtype_matched_ab_20261008");
      }
  }
}
