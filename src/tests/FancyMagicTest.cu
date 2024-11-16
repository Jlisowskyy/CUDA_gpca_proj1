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

__global__ void FancyMagicKernel(const __uint64_t *seeds, __uint64_t *results, const unsigned size) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;

    if (idx < gridDim.x * blockDim.x) {
        results[idx] = seeds[idx] * seeds[idx];
    }
}

void FancyMagicTest_(int threadsAvailable, const cudaDeviceProp &deviceProps) {
    static constexpr unsigned TEST_SIZE = 1'0;

    std::cout << "Fancy Magic Test" << std::endl;
    std::cout << "Received " << threadsAvailable << " available threads..." << std::endl;

    std::vector<__uint64_t> vSeeds(TEST_SIZE, 0);

    std::mt19937_64 rng{std::random_device{}()};
    for (auto &seed: vSeeds) {
        seed = rng();
    }

    thrust::device_vector<__uint64_t> dSeeds(vSeeds);
    thrust::device_vector<__uint64_t> dResults(TEST_SIZE, 0);

//    const unsigned blocks = std::ceil(static_cast<double>(threadsAvailable) / deviceProps.maxThreadsPerBlock);
//    const unsigned threads = deviceProps.maxThreadsPerBlock;

    const unsigned blocks = deviceProps.multiProcessorCount * 2;
    const unsigned threads = deviceProps.maxThreadsPerMultiProcessor / 2;
    const double threadsUtilized = blocks * threads;;

    std::cout << "Utilizing " << blocks << " blocks..." << std::endl;
    std::cout << "Utilizing " << threads << " threads per block..." << std::endl;
    std::cout << "Totally utilizing " << threadsUtilized << " threads..." << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    FancyMagicKernel<<<blocks, threads>>>(thrust::raw_pointer_cast(dSeeds.data()),
                                          thrust::raw_pointer_cast(dResults.data()), TEST_SIZE);
    cudaDeviceSynchronize();
    auto t2 = std::chrono::high_resolution_clock::now();

    if (cudaGetLastError() != cudaSuccess) {
        throw std::runtime_error("Fancy Magic Test failed inside the kernel!");
    }

    const double seconds = std::chrono::duration<double>(t2 - t1).count() / 1'000'000'000.0;
    const double milliseconds = seconds * 1'000.0;
    const double readsPerSecond = (TEST_SIZE * threadsUtilized) / seconds;
    const double readsPerMillisecond = (TEST_SIZE * threadsUtilized) / milliseconds;

    std::cout << std::format("Fancy Magic Test took {} seconds, {} miliseconds,\n"
                             "Average performance was:\n"
                             "reads per second: {}\n"
                             "reads per millisecond: {}",
                             seconds, milliseconds, readsPerSecond, readsPerMillisecond) << std::endl;

    const auto hResults = dResults;
    const auto sum = std::accumulate(hResults.begin(), hResults.end(), 0ull);

    std::cout << "Control sum of the results: " << sum << std::endl;
}

void FancyMagicTest(int threadsAvailable, const cudaDeviceProp &deviceProps) {
    try {
        FancyMagicTest_(threadsAvailable, deviceProps);
    } catch (const std::exception &e) {
        std::cerr << "Fancy Magic Test failed with exception: " << e.what() << std::endl;
    }
}