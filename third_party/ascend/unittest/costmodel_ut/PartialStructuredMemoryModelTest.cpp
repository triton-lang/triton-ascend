#include "AscendModel/RouteModel/Models/PartialStructuredMemoryModel.h"
#include "AscendModel/RouteModel/Models/PartialStructuredSimtLoadModel.h"
#include "AscendModel/RouteModel/Models/PartialStructuredStaticModel.h"
#include <gtest/gtest.h>

using namespace mlir::ascend::partial_memory;

TEST(PartialStructuredMemoryModelTest, StaticModelsMatchFrozenReferences) {
  // Independent E/W and the true full-group F2/F4 predictions. These are not
  // obtained by multiplying the F1 price. Masked rows use a tail half-prefix.
  FactorLoadGeometry c{
      {{{8, 46, true}, {16, 3, false}}, 32, 1, true}, 1, 1472, {0, 0}, {8, 16}};
  struct Reference {
    Model model;
    int factor;
    bool masked;
    double ticks;
  };
  for (const auto &r :
       {Reference{Model::SimdLoad, 1, false, 735.4462395047153},
        Reference{Model::SimdLoad, 1, true, 665.5191911728514},
        Reference{Model::SimdStore, 1, false, 4824.885842617609},
        Reference{Model::SimdStore, 1, true, 5806.000000000003},
        Reference{Model::SimtLoad, 1, false, 150.97310520707447},
        Reference{Model::SimtLoad, 2, false, 173.38050333732676},
        Reference{Model::SimtLoad, 4, false, 222.4674727315751},
        Reference{Model::SimtLoad, 1, true, 138.12692343167586},
        Reference{Model::SimtLoad, 2, true, 177.32168591623153},
        Reference{Model::SimtLoad, 4, true, 181.39361035187534},
        Reference{Model::SimtStore, 1, false, 229.4340747053534},
        Reference{Model::SimtStore, 2, false, 620.8297160760349},
        Reference{Model::SimtStore, 4, false, 542.6056212341609},
        Reference{Model::SimtStore, 1, true, 368.46656099838117},
        Reference{Model::SimtStore, 2, true, 376.7140884698134},
        Reference{Model::SimtStore, 4, true, 417.97612056594676}}) {
    c.factor = r.factor;
    c.upper[1] = r.masked ? 8 : 16;
    auto value = predictStatic(c, r.model);
    ASSERT_TRUE(value);
    EXPECT_NEAR(*value, r.ticks, 1e-8);
  }
}

TEST(PartialStructuredMemoryModelTest, StaticModelsRejectUnsupportedDomain) {
  FactorLoadGeometry c{
      {{{8, 46, true}, {16, 3, false}}, 32, 1, true}, 4, 1472, {0, 0}, {8, 16}};
  ASSERT_TRUE(predictStatic(c, Model::SimtStore));
  EXPECT_FALSE(predictStatic(c, Model::SimdStore));
  c.geometry.warps = 32;
  EXPECT_FALSE(predictStatic(c, Model::SimtLoad));
  c.geometry.warps = 1;
  c.programStride = 0;
  EXPECT_FALSE(predictStatic(c, Model::SimtStore));
  c.factor = 1;
  c.upper[1] = 3;
  EXPECT_FALSE(predictStatic(c, Model::SimdLoad));
  c.upper[1] = 16;
  c.geometry.axes[1].pitch = 0;
  EXPECT_FALSE(predictStatic(c, Model::SimtLoad));
}

TEST(PartialStructuredMemoryModelTest, FactorLoadUsesIndependentEAndWarps) {
  FactorLoadGeometry c{
      {{{4, 32, true}, {8, 3, false}}, 32, 2, true}, 2, 512, {0, 0}, {4, 8}};
  ASSERT_TRUE(predictFactorLoad(c)); // E=32, W=2: no q=1 restriction.
  auto masked = c;
  masked.upper[0] = 2;
  ASSERT_TRUE(predictFactorLoad(masked));
  EXPECT_LT(extractFactorLoadFeatures(masked).at("K"),
            extractFactorLoadFeatures(c).at("K"));
  for (int64_t factor : {2, 4}) {
    c.factor = factor;
    ASSERT_TRUE(predictFactorLoad(c));
  }
  c.factor = 1;
  EXPECT_FALSE(predictFactorLoad(c));
}

TEST(PartialStructuredMemoryModelTest, FactorLoadRejectsUnmeasuredGeometry) {
  FactorLoadGeometry c{
      {{{4, 32, true}, {8, 3, false}}, 32, 2, true}, 4, 512, {0, 0}, {4, 8}};
  ASSERT_TRUE(predictFactorLoad(c));
  auto bad = c;
  bad.programStride = 0;
  EXPECT_FALSE(predictFactorLoad(bad));
  bad = c;
  bad.programStride = 1;
  EXPECT_FALSE(predictFactorLoad(bad));
  bad = c;
  bad.geometry.warps = 32;
  EXPECT_FALSE(predictFactorLoad(bad)); // 4096 physical threads.
  bad = c;
  bad.upper[1] = 3;
  EXPECT_FALSE(predictFactorLoad(bad)); // Rectangle outside measured families.
  bad = c;
  bad.upper[0] = 0;
  EXPECT_FALSE(predictFactorLoad(bad));
  bad = c;
  bad.geometry.floatingPoint = false;
  EXPECT_FALSE(predictFactorLoad(bad)); // 32-bit representative is f32.
  bad = c;
  bad.geometry.axes.back().discrete = true;
  EXPECT_FALSE(predictFactorLoad(bad));
}

TEST(PartialStructuredMemoryModelTest, ShortFormulaReferencePredictions) {
  // References from frozen Python inference; asymmetric shape and large odd
  // stride exercise byte geometry and affine formula selection.
  Geometry ds{{{2, 15346, true}, {16, 1023, false}}, 32, 1, true};
  for (auto item : {std::pair{Model::SimtLoad, 170.99446226275214},
                    std::pair{Model::SimtStore, 180.61070864903908},
                    std::pair{Model::SimdLoad, 166.24302465065506}}) {
    auto result = predict(ds, item.first);
    ASSERT_TRUE(result);
    EXPECT_NEAR(*result, item.second, 1e-8);
  }
  Geometry sd{{{2, 49088, false}, {16, 1, true}}, 32, 1, true};
  auto result = predict(sd, Model::SimdStore);
  ASSERT_TRUE(result);
  EXPECT_NEAR(*result, 2895.7006393137017, 1e-8);
}

TEST(PartialStructuredMemoryModelTest, UnsupportedDomainFallsBack) {
  Geometry valid{{{4, 8, true}, {8, 1, false}}, 32, 1, true};
  ASSERT_TRUE(predict(valid, Model::SimtLoad));
  auto q2 = valid;
  q2.warps = 2;
  EXPECT_FALSE(predict(q2, Model::SimtLoad));
  auto zero = valid;
  zero.axes.back().pitch = 0;
  EXPECT_FALSE(predict(zero, Model::SimtLoad));
  auto negative = valid;
  negative.axes.back().pitch = -1;
  EXPECT_FALSE(predict(negative, Model::SimdStore));
  auto allD = valid;
  allD.axes.back().discrete = true;
  EXPECT_FALSE(predict(allD, Model::SimtLoad));
  auto allS = valid;
  allS.axes.front().discrete = false;
  EXPECT_FALSE(predict(allS, Model::SimdLoad));
  auto boolean = valid;
  boolean.bits = 1;
  EXPECT_FALSE(predict(boolean, Model::SimdStore));
  auto nonPower = valid;
  nonPower.warps = 3;
  EXPECT_FALSE(predict(nonPower, Model::SimtStore));
  EXPECT_FALSE(predict(valid, static_cast<Model>(4)));
}

TEST(PartialStructuredMemoryModelTest,
     DiscretePitchIsNotRuntimeAddressCoverage) {
  Geometry g{{{4, 32, true}, {8, 4, false}}, 16, 1, true};
  auto f = extractFeatures(g);
  EXPECT_DOUBLE_EQ(f.at("log_D_pitch"), 5);
  EXPECT_DOUBLE_EQ(f.at("log_step"), 3);
  EXPECT_DOUBLE_EQ(f.at("static_work"), 32);
  EXPECT_EQ(f.count("U"), 0u);
  EXPECT_EQ(f.count("V"), 0u);
  // D coordinates here are counterfactual logical positions 0..3. These
  // counts describe known coefficients and do not estimate random occupancy.
  EXPECT_DOUBLE_EQ(f.at("basis_bins128"), 2);
  EXPECT_DOUBLE_EQ(f.at("S_basis_bins128"), 1);
  EXPECT_DOUBLE_EQ(f.at("S_basis_work128"), 4);
  EXPECT_DOUBLE_EQ(f.at("basis_waves128"), 2);
}
