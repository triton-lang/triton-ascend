set(PROTON_DRIVER_SOURCES
  Device.cpp
  GPU/CannApi.cpp
)

if(PROTON_BUILD_CUDA)
  list(APPEND PROTON_DRIVER_SOURCES
    GPU/CudaApi.cpp
    GPU/CuptiApi.cpp
    GPU/NvtxApi.cpp
  )
endif()

if(PROTON_BUILD_HIP)
  list(APPEND PROTON_DRIVER_SOURCES
    GPU/HipApi.cpp
    GPU/HsaApi.cpp
    GPU/RoctracerApi.cpp
  )
endif()

if(PROTON_BUILD_MSPTI)
  list(APPEND PROTON_DRIVER_SOURCES GPU/MsptiApi.cpp)
endif()

add_proton_library(ProtonDriver ${PROTON_DRIVER_SOURCES})
