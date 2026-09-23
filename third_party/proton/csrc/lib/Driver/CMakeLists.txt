add_proton_library(ProtonDriver
	Device.cpp
	GPU/CudaApi.cpp
	GPU/CuptiApi.cpp
	GPU/HipApi.cpp
	GPU/HsaApi.cpp
	GPU/RoctracerApi.cpp
	GPU/NvtxApi.cpp
)

if(PROTON_ENABLE_NPU)
	target_sources(ProtonDriver PRIVATE NPU/AscendApi.cpp)
endif()
