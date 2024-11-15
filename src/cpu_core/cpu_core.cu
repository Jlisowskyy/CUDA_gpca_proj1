//
// Created by Jlisowskyy on 14/11/24.
//

#include "cpu_core.cuh"

#include "../cuda_core/helpers.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <format>
#include <string_view>
#include <string>
#include <cassert>

cpu_core::cpu_core() = default;

cpu_core::~cpu_core() = default;

void cpu_core::runCVC() {
    _runSimpleMoveGen();
}

void cpu_core::runPVC() {
    _runSimpleMoveGen();
}

void cpu_core::init() {
    const auto [bestDeviceIdx, deviceHardwareThreads] = _pickGpu();
    CUDA_ASSERT_SUCCESS(cudaSetDevice(bestDeviceIdx));
    m_deviceThreads = deviceHardwareThreads;
}

std::pair<int, int> cpu_core::_pickGpu() {
    int deviceCount;

    std::cout << "Processing CUDA devices..." << std::endl;

    CUDA_ASSERT_SUCCESS(cudaGetDeviceCount(&deviceCount));

    std::cout << std::format("Found {} CUDA devices", deviceCount) << std::endl;

    if (deviceCount == 0) {
        std::cerr << "No CUDA devices found" << std::endl;
        exit(EXIT_FAILURE);
    }

    int bestDeviceIdx = 0;
    int bestDeviceScore = 0;
    std::string bestDeviceName{};

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop{};

        if (CUDA_TRACE_ERROR(cudaGetDeviceProperties(&prop, i))) {
            continue;
        }

        _dumpGPUInfo(i, prop);

        const int MaxThreadsPerBlock = prop.maxThreadsPerBlock * prop.multiProcessorCount;

        if (MaxThreadsPerBlock > bestDeviceScore) {
            bestDeviceScore = MaxThreadsPerBlock;
            bestDeviceIdx = i;
            bestDeviceName = prop.name;
        }
    }

    std::cout << std::format("Device chosen for computation: {} ({}) with {} threads available.", bestDeviceName,
                             bestDeviceIdx, bestDeviceScore) << std::endl;

    return {bestDeviceIdx, bestDeviceScore};
}

void cpu_core::_dumpGPUInfo(const int idx, const cudaDeviceProp &prop) {
    static constexpr std::string_view INFO_FORMAT =
            R"(Device {}: "{}"
          Compute Capability: {}.{}
          Total Global Memory: {:.2f} GB
          Max Threads per Block: {}
          Max Threads Dimensions: [{}, {}, {}]
          Max Grid Size: [{}, {}, {}]
          Max threads per SM: {}
          Number of SMs: {}
          Warp Size: {}
          Memory Clock Rate: {:.0f} MHz
          Memory Bus Width: {} bits
          L2 Cache Size: {} KB
          Max Shared Memory Per Block: {} KB
          Total Constant Memory: {} KB
          Compute Mode: {}
          Concurrent Kernels: {}
          ECC Enabled: {}
          Multi-GPU Board: {}
          Unified Addressing: {}
)";

    std::cout << std::format(INFO_FORMAT,
                             idx, prop.name,
                             prop.major, prop.minor,
                             static_cast<float>(prop.totalGlobalMem) / (1024.0f * 1024.0f * 1024.0f),
                             prop.maxThreadsPerBlock,
                             prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2],
                             prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2],
                             prop.maxThreadsPerMultiProcessor,
                             prop.multiProcessorCount,
                             prop.warpSize,
                             prop.memoryClockRate * 1e-3f,
                             prop.memoryBusWidth,
                             prop.l2CacheSize / 1024,
                             prop.sharedMemPerBlock / 1024,
                             prop.totalConstMem / 1024,
                             prop.computeMode,
                             prop.concurrentKernels ? "Yes" : "No",
                             prop.ECCEnabled ? "Yes" : "No",
                             prop.isMultiGpuBoard ? "Yes" : "No",
                             prop.unifiedAddressing ? "Yes" : "No"
    );
}

void cpu_core::_runSimpleMoveGen() {
    assert(m_deviceThreads > 0 && "No device threads available");
}

