//
// Created by Jlisowskyy on 16/11/24.
//

#include "CudaTests.cuh"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <string_view>

#include "../cuda_core/RookMap.cuh"
#include "../cuda_core/BishopMap.cuh"

#include "cpu/CpuTests.h"

__device__ static constexpr unsigned TEST_SIZE = 1'000'000;

__device__ __forceinline__ __uint64_t simpleRand(__uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

template<typename MapT>
__global__ void FancyMagicKernel(const __uint64_t *seeds, __uint64_t *results) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const __uint64_t seed = seeds[idx];

    unsigned randomPos = idx % 64;
    __uint64_t board = seed;

    for (unsigned i = 0; i < TEST_SIZE; ++i) {
        board = simpleRand(board);
        randomPos = (randomPos + (board & 63)) & 64;

        const auto result = MapT::GetMoves(randomPos, board);

        results[idx] += result;
    }
}

template<typename MapT>
void PerformTestOnMap_(unsigned blocks, unsigned threads, thrust::device_vector<__uint64_t> &dSeeds,
                       thrust::device_vector<__uint64_t> &dResults, std::string_view title = "") {
    const double threadsUtilized = blocks * threads;

    std::cout << "Starting run for " << title << "..." << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    FancyMagicKernel<MapT><<<blocks, threads>>>(thrust::raw_pointer_cast(dSeeds.data()),
                                                thrust::raw_pointer_cast(dResults.data()));
    CUDA_ASSERT_SUCCESS(cudaDeviceSynchronize());
    auto t2 = std::chrono::high_resolution_clock::now();

    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("Fancy Magic Test failed inside the kernel!");
    }

    const double seconds = std::chrono::duration<double>(t2 - t1).count();
    const double milliseconds = seconds * 1000.0;
    const double readsPerSecond = (TEST_SIZE * threadsUtilized) / seconds;
    const double readsPerMillisecond = (TEST_SIZE * threadsUtilized) / milliseconds;

    std::cout << std::format("Fancy Magic Test for {} took {} seconds, {} milliseconds,\n"
                             "Average performance was:\n"
                             "reads per second: {}\n"
                             "reads per millisecond: {}",
                             title, seconds, milliseconds, readsPerSecond, readsPerMillisecond) << std::endl;

    const auto hResults = dResults;
    const auto sum = std::accumulate(hResults.begin(), hResults.end(), 0ull);

    std::cout << "Control sum of the results: " << sum << std::endl;
}

void FancyMagicTest_(int threadsAvailable, const cudaDeviceProp &deviceProps) {
    std::cout << "Fancy Magic Test" << std::endl;
    std::cout << "Received " << threadsAvailable << " available threads..." << std::endl;

    //    const unsigned blocks = std::ceil(static_cast<double>(threadsAvailable) / deviceProps.maxThreadsPerBlock);
    //    const unsigned threads = deviceProps.maxThreadsPerBlock;

    const unsigned blocks = deviceProps.multiProcessorCount * 2;
    const unsigned threads = deviceProps.maxThreadsPerMultiProcessor / 2;
    const double threadsUtilized = blocks * threads;;
    const auto sizeThreads = static_cast<size_t>(threadsUtilized);

    std::cout << "Utilizing " << blocks << " blocks..." << std::endl;
    std::cout << "Utilizing " << threads << " threads per block..." << std::endl;
    std::cout << "Totally utilizing " << threadsUtilized << " threads..." << std::endl;

    std::vector<__uint64_t> vSeeds{};
    vSeeds.reserve(sizeThreads);

    std::mt19937_64 rng{std::random_device{}()};
    for (size_t i = 0; i < sizeThreads; ++i) {
        vSeeds.push_back(rng());
    }

    thrust::device_vector<__uint64_t> dSeeds(vSeeds);
    thrust::device_vector<__uint64_t> dResults(sizeThreads, 0);

//    PerformTestOnMap_<RookMap>(blocks, threads, dSeeds, dResults, "RookMap");
    PerformTestOnMap_<BishopMap>(blocks, threads, dSeeds, dResults, "BishopMap");

    std::cout << "Fancy Magic Test finished!" << std::endl;
}

void FancyMagicTest(int threadsAvailable, const cudaDeviceProp &deviceProps) {
    try {
//        cpu::FancyMagicTest();
        FancyMagicTest_(threadsAvailable, deviceProps);
    } catch (const std::exception &e) {
        std::cerr << "Fancy Magic Test failed with exception: " << e.what() << std::endl;
    }
}