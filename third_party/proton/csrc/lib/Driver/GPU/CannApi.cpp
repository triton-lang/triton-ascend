#include "Driver/GPU/CannApi.h"
#include "pybind11/pybind11.h"

namespace proton::cann {

Device getDevice(uint64_t index) {
  pybind11::gil_scoped_acquire gil;
  auto utils = pybind11::module_::import("triton.runtime")
                   .attr("driver")
                   .attr("active")
                   .attr("utils");
  auto properties = utils.attr("get_profiling_device_properties")(index);
  return Device(DeviceType::NPU, index,
                properties["clock_rate"].cast<uint64_t>(),
                properties["memory_clock_rate"].cast<uint64_t>(),
                /*busWidth=*/0, properties["num_vectorcore"].cast<uint64_t>(),
                properties["arch"].cast<std::string>());
}

uint64_t getDeviceId() {
  pybind11::gil_scoped_acquire gil;
  auto driver =
      pybind11::module_::import("triton.runtime").attr("driver").attr("active");
  return driver.attr("utils")
      .attr("get_profiling_device_id")(driver.attr("get_current_device")())
      .cast<uint64_t>();
}

} // namespace proton::cann
