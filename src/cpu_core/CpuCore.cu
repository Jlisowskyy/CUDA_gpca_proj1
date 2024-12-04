//
// Created by Jlisowskyy on 14/11/24.
//

#include "CpuCore.cuh"

#include "../cuda_core/Helpers.cuh"
#include "../cuda_core/RookMap.cuh"

#include "MctsEngine.cuh"

#include <cuda_runtime.h>

#include <iostream>
#include <format>
#include <string_view>
#include <string>
#include <cassert>

void initializeRookMap() {
    FancyMagicRookMap hostMap{
            false}; /* WORKAROUND: This is a workaround for the fact that the constructor is not constexpr */
    CUDA_ASSERT_SUCCESS(cudaMemcpyToSymbol(G_ROOK_FANCY_MAP_INSTANCE, &hostMap, sizeof(FancyMagicRookMap)));
}

CpuCore::CpuCore() = default;

CpuCore::~CpuCore() {
    delete m_deviceProps;
}

void CpuCore::runCVC(const __uint32_t moveTime) {

}

void CpuCore::runPVC(const __uint32_t moveTime) {

}

void CpuCore::init() {
    const auto [bestDeviceIdx, deviceHardwareThreads, deviceProps] = _pickGpu();

    assert(deviceProps != nullptr && "Device properties cannot be nullptr");
    CUDA_ASSERT_SUCCESS(cudaSetDevice(bestDeviceIdx));

    size_t stackSize;
    cudaDeviceGetLimit(&stackSize, cudaLimitStackSize);
    std::cout << "Current stack size: " << stackSize << " bytes" << std::endl;

    m_deviceThreads = deviceHardwareThreads;
    m_deviceProps = deviceProps;

    initializeRookMap();
}

std::tuple<int, int, cudaDeviceProp *> CpuCore::_pickGpu() {
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
    cudaDeviceProp *bestDeviceProps = nullptr;
    bestDeviceProps = new cudaDeviceProp;

    for (int i = 0; i < deviceCount; i++) {
        cudaDeviceProp prop{};

        if (CUDA_TRACE_ERROR(cudaGetDeviceProperties(&prop, i))) {
            continue;
        }

        _dumpGPUInfo(i, prop);

        const int MaxThreadsPerSM = prop.maxThreadsPerMultiProcessor * prop.multiProcessorCount;

        if (MaxThreadsPerSM > bestDeviceScore) {
            bestDeviceScore = MaxThreadsPerSM;
            bestDeviceIdx = i;
            bestDeviceName = prop.name;
        }
    }

    std::cout << std::format("Device chosen for computation: {} ({}) with {} threads available.", bestDeviceName,
                             bestDeviceIdx, bestDeviceScore) << std::endl;

    CUDA_ASSERT_SUCCESS(cudaGetDeviceProperties(bestDeviceProps, bestDeviceIdx));
    return {bestDeviceIdx, bestDeviceScore, bestDeviceProps};
}

void CpuCore::_dumpGPUInfo(const int idx, const cudaDeviceProp &prop) {
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
          Multi-GPU cpu_Board: {}
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
    ) << std::endl;
}

void CpuCore::setBoard(cuda_Board *board) {
    assert(board != nullptr && "cpu_Board cannot be nullptr");
    m_board = board;
}


