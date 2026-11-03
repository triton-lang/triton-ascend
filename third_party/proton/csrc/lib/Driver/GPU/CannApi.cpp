#include "Driver/GPU/CannApi.h"
#include "pybind11/pybind11.h"

namespace proton::cann {

Device getDevice(uint64_t index) {
  pybind11::gil_scoped_acquire gil;
  auto utils = pybind11::module_::import("triton.runtime")
                   .attr("driver")
                   .attr("active")
                   .attr("utils");
  auto arch = utils.attr("get_arch")().cast<std::string>();
  auto vectorCoreCount = utils.attr("get_aivector_core_num")().cast<uint64_t>();
  auto vectorClockRate =
      utils.attr("get_vector_clock_rate")(index).cast<uint64_t>();
  auto hbmClockRate = utils.attr("get_hbm_clock_rate")(index).cast<uint64_t>();
  return Device(DeviceType::NPU, index, vectorClockRate, hbmClockRate,
                /*busWidth=*/0, vectorCoreCount, arch);
}

} // namespace proton::cann
