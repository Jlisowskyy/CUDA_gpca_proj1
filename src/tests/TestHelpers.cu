//
// Created by Jlisowskyy on 01/12/24.
//

#include "CudaTests.cuh"

#include "../ported/CpuTests.h"
#include "../cuda_core/ComputeKernels.cuh"

#include <iostream>
#include <filesystem>
#include <random>
#include <tuple>

#include <thrust/device_vector.h>

bool G_USE_DEFINED_SEED = false;

void PolluteCache() {
    static constexpr uint32_t POLLUTE_SIZE = 1'000'000'0;
    static constexpr uint32_t ROUNDS = 50;

    std::cout << "Polluting caches..." << std::endl;

    const uint32_t blocks = POLLUTE_SIZE / 1024;
    const auto seeds = GenSeeds(POLLUTE_SIZE);
    const auto data = GenSeeds(POLLUTE_SIZE);

    thrust::device_vector<uint32_t> dSeeds = seeds;
    thrust::device_vector<uint32_t> dData = data;
    thrust::device_vector<uint32_t> dResult(POLLUTE_SIZE);

    PolluteCache<<<blocks, 1024>>>(
            thrust::raw_pointer_cast(dData.data()),
            thrust::raw_pointer_cast(dSeeds.data()),
            thrust::raw_pointer_cast(dResult.data()),
            ROUNDS
    );
    CUDA_ASSERT_SUCCESS(cudaGetLastError());
    GUARDED_SYNC();
}

std::vector<std::string> LoadFenDb() {
    const auto fenDbPath = "./test_data/fen_db.txt";

    const auto fenDb = cpu::LoadFenDb(fenDbPath);
    std::cout << "Loaded " << fenDb.size() << " positions" << std::endl;

    return fenDb;
}

std::vector<uint32_t> GenSeeds(const uint32_t size) {
    std::vector<uint32_t> seeds{};
    seeds.reserve(size);

    std::mt19937 rng{G_USE_DEFINED_SEED ? DEFAULT_TEST_SEED : std::random_device{}()};
    for (uint32_t i = 0; i < size; ++i) {
        seeds.push_back(rng());
    }
    return seeds;
}

std::tuple<uint32_t, uint32_t> GetDims(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    //    const unsigned blocks = std::ceil(static_cast<double>(threadsAvailable) / deviceProps.maxThreadsPerBlock);
    //    const unsigned threads = deviceProps.maxThreadsPerBlock;

    const uint32_t blocks = deviceProps.multiProcessorCount * 2;
    const uint32_t threads = deviceProps.maxThreadsPerMultiProcessor / 2;
    const uint32_t threadsUtilized = blocks * threads;

    std::cout << "Utilizing " << blocks << " blocks..." << std::endl;
    std::cout << "Utilizing " << threads << " threads per block..." << std::endl;
    std::cout << "Totally utilizing " << threadsUtilized << " / " << threadsAvailable << " threads..." << std::endl;

    return {blocks, threads};
}
