#include "AscendModel/HardwareConfig.h"
#include "AscendModel/Profile/MicrobenchmarkProfile.h"

#include <gtest/gtest.h>

#include <filesystem>
#include <string>

namespace {

std::string get910BConfigPath() {
  namespace fs = std::filesystem;
  fs::path src = fs::path(__FILE__).parent_path().parent_path().parent_path();
  fs::path cfg = src / "costmodel" / "configs" / "ascend_910b.json";
  return cfg.string();
}

std::string getDavidConfigPath() {
  namespace fs = std::filesystem;
  fs::path src = fs::path(__FILE__).parent_path().parent_path().parent_path();
  fs::path cfg = src / "costmodel" / "configs" / "ascend_davidv100.json";
  return cfg.string();
}

std::string getDavidMicrobenchmarkProfilePath() {
  namespace fs = std::filesystem;
  fs::path src = fs::path(__FILE__).parent_path().parent_path().parent_path();
  fs::path profile = src / "costmodel" / "profiles" / "microbench" /
                     "ascend_davidv100_v1.json";
  return profile.string();
}

std::unique_ptr<mlir::ascend::HardwareConfig>
loadConfigFromText(llvm::StringRef text) {
  auto parsed = llvm::json::parse(text);
  if (!parsed)
    return nullptr;
  return mlir::ascend::HardwareConfig::loadFromJSON(*parsed);
}

} // namespace

TEST(CostModelHardwareConfigTest, Default910BHasExpectedBasics) {
  auto cfg = mlir::ascend::HardwareConfig::getDefault910B();
  ASSERT_NE(cfg, nullptr);

  EXPECT_GT(cfg->getClockFrequencyGHz(), 0.0);
  EXPECT_GT(cfg->getCyclesPerMicrosecond(), 0);
  EXPECT_GT(cfg->getCubeTFLOPS(), 0.0);
  EXPECT_GT(cfg->getVectorTFLOPS(), 0.0);

  const auto *hbm = cfg->getMemorySpace("hbm");
  ASSERT_NE(hbm, nullptr);
  EXPECT_GT(hbm->sizeBytes, 0u);
  EXPECT_GT(hbm->bandwidthBytesPerCycle, 0.0);
}

TEST(CostModelHardwareConfigTest, LoadFromCommitConfigFile) {
  const std::string path = get910BConfigPath();

  auto cfg = mlir::ascend::HardwareConfig::loadFromFile(path);
  ASSERT_NE(cfg, nullptr) << "failed to load config from: " << path;

  std::string error;
  EXPECT_TRUE(cfg->validate(error)) << error;
  EXPECT_GT(cfg->getCubeTFLOPS(), 0.0);
  EXPECT_GT(cfg->getVectorTFLOPS(), 0.0);

  int m = 0, n = 0, k = 0;
  cfg->getCubeFractalSize(/*elementBits=*/16, m, n, k);
  EXPECT_GT(m, 0);
  EXPECT_GT(n, 0);
  EXPECT_GT(k, 0);
}

TEST(CostModelHardwareConfigTest, LoadGlobalConfigFromFile) {
  const std::string path = get910BConfigPath();
  std::string error;
  ASSERT_TRUE(mlir::ascend::loadHardwareConfigFromFile(path, error)) << error;

  mlir::ascend::HardwareConfig &global = mlir::ascend::getHardwareConfig();
  EXPECT_GT(global.getClockFrequencyGHz(), 0.0);
  EXPECT_GT(global.getHBMBandwidthGBs(), 0.0);
}

TEST(CostModelHardwareConfigTest,
     SharedMicrobenchmarkProfilePreservesMetadataAndUnits) {
  auto profile = mlir::ascend::MicrobenchmarkProfile::loadFromFile(
      getDavidMicrobenchmarkProfilePath());
  if (!profile)
    FAIL() << llvm::toString(profile.takeError());

  EXPECT_EQ(profile->getProfileVersion(),
            "david-v100-shared-microbench-20260730-v2");
  EXPECT_EQ(profile->getTarget(), "Ascend950PR/dav-c310");
  EXPECT_FALSE(profile->getContentSha256().empty());

  const auto *measurement = profile->getMeasurement("simd.f32.add.throughput");
  ASSERT_NE(measurement, nullptr);
  EXPECT_DOUBLE_EQ(measurement->value, 3.30);
  EXPECT_EQ(measurement->unit, "vector_instruction/system_cycle");
  EXPECT_EQ(measurement->cycleDomain, "SYS_CNT");
  EXPECT_EQ(measurement->scope,
            "single_aiv_runtime_loop_ilp4_or_more_effective_source_vadd");
  EXPECT_EQ(measurement->sourceKind, "isolated_microbenchmark");
  EXPECT_EQ(measurement->confidence, "high");

  const auto *serializedSetup =
      profile->getMeasurement("simt.setup.empty_with_barrier");
  ASSERT_NE(serializedSetup, nullptr);
  EXPECT_DOUBLE_EQ(serializedSetup->value, 141.0);
  EXPECT_EQ(serializedSetup->confidence, "medium");

  const auto *gmLoad = profile->getMeasurement("simt.gm.load.throughput");
  ASSERT_NE(gmLoad, nullptr);
  EXPECT_DOUBLE_EQ(gmLoad->value, 0.1761917890625);
  const auto *gmStore = profile->getMeasurement("simt.gm.store.throughput");
  ASSERT_NE(gmStore, nullptr);
  EXPECT_DOUBLE_EQ(gmStore->value, 0.12914221875);

  auto converted = profile->convertRatePerCycle(
      measurement->value, "clock.sys_cnt.frequency_mhz",
      "clock.device_compute.frequency_mhz");
  if (!converted)
    FAIL() << llvm::toString(converted.takeError());
  EXPECT_NEAR(*converted, 3.30 * 988.9 / 1650.0, 1.0e-12);

  auto wrongUnit =
      profile->requireValue("simd.f32.add.throughput", "byte/system_cycle");
  EXPECT_FALSE(static_cast<bool>(wrongUnit));
  if (!wrongUnit)
    llvm::consumeError(wrongUnit.takeError());

  auto wrongDomain = profile->requireValue("simd.f32.add.throughput",
                                           "vector_instruction/system_cycle",
                                           "device_compute");
  EXPECT_FALSE(static_cast<bool>(wrongDomain));
  if (!wrongDomain)
    llvm::consumeError(wrongDomain.takeError());
}

TEST(CostModelHardwareConfigTest,
     DavidConfigLoadsAndConsumesReferencedSharedProfile) {
  auto cfg = mlir::ascend::HardwareConfig::loadFromFile(getDavidConfigPath());
  ASSERT_NE(cfg, nullptr);
  ASSERT_NE(cfg->getMicrobenchmarkProfile(), nullptr);
  EXPECT_EQ(cfg->getTarget(), "Ascend950PR/dav-c310");
  EXPECT_EQ(cfg->getMicrobenchmarkProfile()->getProfileVersion(),
            "david-v100-shared-microbench-20260730-v2");
  EXPECT_NEAR(
      cfg->getMicrobenchmarkRatePerDeviceCycle("simd.f32.add.throughput"),
      3.30 * 988.9 / 1650.0, 1.0e-12);
}

TEST(CostModelHardwareConfigTest, EstimationAPIsReturnNonNegative) {
  auto cfg = mlir::ascend::HardwareConfig::getDefault910B();
  ASSERT_NE(cfg, nullptr);

  EXPECT_GE(cfg->estimateCubeCycles(/*M=*/128, /*N=*/128, /*K=*/64), 0);
  EXPECT_GE(cfg->estimateVectorCycles(/*numElements=*/4096), 0);
  EXPECT_GE(cfg->estimateMemoryCyclesWithLatency("hbm", /*bytes=*/8192), 0);
}

TEST(CostModelHardwareConfigTest, ParsesCustomJsonAndExposesCoreParameters) {
  auto cfg = loadConfigFromText(R"json(
{
  "name": "Tiny Test NPU",
  "vendor": "UnitTest",
  "version": "1.0",
  "clock": {"frequency_ghz": 2.0},
  "memory_spaces": {
    "hbm": {"type": "off_chip", "size_gb": 1, "bandwidth_gbps": 2000, "latency_cycles": 11},
    "ub": {"type": "on_chip_local", "size_kb": 64, "bandwidth_gbps": 1000, "latency_cycles": 3},
    "l0c": {"type": "register_file", "size_kb": 32, "bandwidth_gbps": 500, "latency_cycles": 1}
  },
  "compute_units": {
    "cube": {
      "type": "matrix_engine",
      "tflops_fp16": 64,
      "tflops_fp32": 32,
      "tflops_int8": 128,
      "tile_m": 8,
      "tile_n": 16,
      "tile_k": 32,
      "fractal_sizes": {
        "fp16": {"m": 16, "n": 16, "k": 16},
        "fp32": {"m": 16, "n": 16, "k": 8},
        "int8": {"m": 16, "n": 16, "k": 32}
      },
      "input_spaces": ["hbm", "ub"],
      "output_space": "l0c"
    },
    "vector": {
      "type": "simd_engine",
      "tflops_fp32": 8,
      "width_elements": 64,
      "width_bytes": 256,
      "compute_space": "ub",
      "supported_ops": ["add", "exp"],
      "supported_dtypes": ["fp32"]
    }
  },
  "data_movers": {
    "vector_mte2": {
      "src_space": "hbm",
      "dst_spaces": ["ub"],
      "bandwidth_gbps": 1000,
      "max_burst_bytes": 1024,
      "alignment_bytes": 32
    }
  },
  "pipeline": {
    "vector_path": {"stages": ["vector_mte2", "vector"]},
    "parallelism": {"cube_path_and_vector_path": true}
  },
  "calibration": {
    "vector_op_cycles_per_vec_instruction": {
      "simple_ops_add_sub_mul_etc": 2,
      "exp": 7
    }
  }
}
)json");
  ASSERT_NE(cfg, nullptr);

  std::string error;
  EXPECT_TRUE(cfg->validate(error)) << error;
  EXPECT_EQ(cfg->getName(), "Tiny Test NPU");
  EXPECT_EQ(cfg->getVendor(), "UnitTest");
  EXPECT_DOUBLE_EQ(cfg->getClockFrequencyGHz(), 2.0);
  EXPECT_EQ(cfg->getCyclesPerMicrosecond(), 2000);

  const auto *hbm = cfg->getMemorySpace("hbm");
  ASSERT_NE(hbm, nullptr);
  EXPECT_EQ(hbm->sizeBytes, 1ULL * 1024 * 1024 * 1024);
  EXPECT_DOUBLE_EQ(hbm->bandwidthBytesPerCycle, 1000.0);
  EXPECT_EQ(hbm->latencyCycles, 11);

  int m = 0, n = 0, k = 0;
  cfg->getCubeTileSize(m, n, k);
  EXPECT_EQ(m, 8);
  EXPECT_EQ(n, 16);
  EXPECT_EQ(k, 32);
  cfg->getCubeFractalSize(/*elementBits=*/32, m, n, k);
  EXPECT_EQ(m, 16);
  EXPECT_EQ(n, 16);
  EXPECT_EQ(k, 8);
  cfg->getCubeFractalSize(/*elementBits=*/8, m, n, k);
  EXPECT_EQ(k, 32);

  EXPECT_EQ(cfg->getVectorWidthElements(), 64);
  EXPECT_EQ(cfg->getVectorOpCyclesPerInstruction("vadd"), 2);
  EXPECT_EQ(cfg->getVectorOpCyclesPerInstruction("vexp"), 7);
  EXPECT_TRUE(cfg->canRunInParallel("cube_path", "vector_path"));
  EXPECT_EQ(cfg->estimateVectorCycles(/*numElements=*/129), 3);
  EXPECT_EQ(cfg->estimateMemoryCycles("vector_mte2", /*bytes=*/2001), 5);
  EXPECT_EQ(cfg->estimateMemoryCyclesWithLatency("hbm", /*bytes=*/2001), 14);
}

TEST(CostModelHardwareConfigTest, RejectsInvalidConfigReferences) {
  auto cfg = loadConfigFromText(R"json(
{
  "clock": {"frequency_ghz": 1.0},
  "memory_spaces": {
    "hbm": {"type": "off_chip", "size_gb": 1, "bandwidth_gbps": 1000}
  },
  "compute_units": {
    "vector": {"type": "simd_engine", "tflops_fp32": 1, "width_elements": 16}
  },
  "data_movers": {
    "bad_mover": {
      "src_space": "hbm",
      "dst_spaces": ["missing_space"],
      "bandwidth_gbps": 100
    }
  }
}
)json");
  ASSERT_NE(cfg, nullptr);

  std::string error;
  EXPECT_FALSE(cfg->validate(error));
  EXPECT_NE(error.find("bad_mover"), std::string::npos);
  EXPECT_NE(error.find("missing_space"), std::string::npos);
}

TEST(CostModelHardwareConfigTest,
     RejectsNonStringMicrobenchmarkProfileReference) {
  auto cfg = loadConfigFromText(R"json(
{
  "microbenchmark_profile": 17,
  "clock": {"frequency_ghz": 1.0}
}
)json");
  EXPECT_EQ(cfg, nullptr);
}
