//
// Created by Jlisowskyy on 16/11/24.
//

#include "CudaTests.cuh"

#include <cuda_runtime.h>
#include <thrust/device_vector.h>
#include <thrust/host_vector.h>

#include <iostream>
#include <vector>
#include <random>
#include <chrono>
#include <numeric>
#include <string_view>

#include "../cuda_core/RookMap.cuh"
#include "../cuda_core/BishopMap.cuh"
#include "../cuda_core/RookMapRuntime.cuh"
#include "../cuda_core/BishopMapRuntime.cuh"

#include "../ported/CpuTests.h"

__device__ static constexpr uint32_t TEST_SIZE = 1'000'000;

/**
 * @brief CUDA kernel for testing Fancy Magic mapping performance.
 *
 * Executes a large number of move generation tests for a given mapping type,
 * using randomly generated seeds and board positions.
 *
 * @tparam MapT Move generation mapping type to test (e.g., RookMap, BishopMap)
 * @param seeds Array of random 64-bit seed values for board generation
 * @param results Array to store the sum of generate bitboards
 */
template<typename MapT>
__global__ void FancyMagicKernel(const uint64_t *seeds, uint64_t *results) {
    const unsigned idx = blockIdx.x * blockDim.x + threadIdx.x;
    const uint64_t seed = seeds[idx];

    unsigned randomPos = idx % 64;
    uint64_t board = seed;

    uint64_t sum{};
    for (unsigned i = 0; i < TEST_SIZE; ++i) {
        simpleRand(board);
        randomPos = (randomPos + (board & 63)) & 63;

        const auto result = MapT::GetMoves(randomPos, board);

        sum += static_cast<unsigned>(result > 0);
    }

    results[idx] = sum;
}

/**
 * @brief Performs performance testing for a specific move generation map type.
 *
 * Launches a CUDA kernel to measure access times and performance characteristics
 * of a given move generation mapping strategy.
 *
 * @tparam MapT Move generation mapping type to test
 * @param blocks Number of CUDA thread blocks to launch
 * @param threads Number of threads per block
 * @param dSeeds Device vector of random seed values
 * @param dResults Device vector to store test results
 * @param title Optional descriptive title for the test
 *
 * @throws std::runtime_error if kernel execution fails
 */
template<typename MapT>
void PerformTestOnMap_(uint32_t blocks, uint32_t threads, thrust::device_vector<uint64_t> &dSeeds,
                       thrust::device_vector<uint64_t> &dResults, std::string_view title = "") {
    const double threadsUtilized = blocks * threads;

    std::cout << "Starting run for " << title << "..." << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();
    FancyMagicKernel<MapT><<<blocks, threads>>>(thrust::raw_pointer_cast(dSeeds.data()),
                                                thrust::raw_pointer_cast(dResults.data()));
    CUDA_TRACE_ERROR(cudaGetLastError());

    const auto rc = cudaDeviceSynchronize();
    CUDA_TRACE_ERROR(rc);

    if (rc != cudaSuccess) {
        throw std::runtime_error("Fancy Magic Test failed inside the kernel!");
    }

    auto t2 = std::chrono::high_resolution_clock::now();

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

/**
 * @brief Internal implementation of Fancy Magic performance test.
 *
 * Prepares test data, launches performance tests for different mapping types,
 * and manages cache pollution between tests.
 *
 * @param threadsAvailable Number of CUDA threads available
 * @param deviceProps CUDA device properties for kernel configuration
 */
void FancyMagicTest_(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    std::cout << "Fancy Magic Test" << std::endl;
    std::cout << "Received " << threadsAvailable << " available threads..." << std::endl;

    const auto [blocks, threads] = GetDims(threadsAvailable, deviceProps);
    const auto sizeThreads = static_cast<size_t>(blocks * threads);

    std::vector<uint64_t> vSeeds{};
    vSeeds.reserve(sizeThreads);

    std::mt19937_64 rng{std::random_device{}()};
    for (size_t i = 0; i < sizeThreads; ++i) {
        vSeeds.push_back(rng());
    }

    thrust::device_vector<uint64_t> dSeeds(vSeeds);
    thrust::device_vector<uint64_t> dResults(sizeThreads, 0);

    PolluteCache();
    std::cout << std::string(80, '-') << std::endl;
    PerformTestOnMap_<RookMap>(blocks, threads, dSeeds, dResults, "RookMap");
    PolluteCache();
    std::cout << std::string(80, '-') << std::endl;
    PerformTestOnMap_<BishopMap>(blocks, threads, dSeeds, dResults, "BishopMap");
    PolluteCache();
    std::cout << std::string(80, '-') << std::endl;
    PerformTestOnMap_<RookMapRuntime>(blocks, threads, dSeeds, dResults, "RookMapRuntime");
    PolluteCache();
    std::cout << std::string(80, '-') << std::endl;
    PerformTestOnMap_<BishopMapRuntime>(blocks, threads, dSeeds, dResults, "BishopMapRuntime");
    PolluteCache();

    std::cout << "Fancy Magic Test finished!" << std::endl;
}

/**
 * @brief CUDA kernel for accessing and testing move generation mappings.
 *
 * Generates moves for figure positions across multiple board states,
 * storing the results for comparison with CPU implementation.
 *
 * @tparam MapT Move generation mapping type to test
 * @param results Array to store generated moves
 * @param count Number of board states to test
 * @param fullMaps Full board state for each test case
 * @param figureMaps Figure (piece) positions for each test case
 */
template<class MapT>
__global__ void
AccessMapping(uint64_t *results, const uint64_t count, const uint64_t *fullMaps, const uint64_t *figureMaps) {
    uint64_t moveCount{};

    for (uint64_t idx = 0; idx < count; ++idx) {
        const uint64_t fullMap = fullMaps[idx];
        uint64_t figureMap = figureMaps[idx];

        while (figureMap != 0) {
            const int msbPos = ExtractMsbPos(figureMap);

            const auto result = MapT::GetMoves(msbPos, fullMap);
            results[moveCount++] = result;

            figureMap ^= cuda_MaxMsbPossible >> msbPos;
        }
    }
}

/**
 * @brief CPU reference implementation for move mapping access.
 *
 * Generates moves using a CPU-based mapping function for comparison
 * with GPU implementation.
 *
 * @param func CPU move generation function
 * @param count Number of board states to test
 * @param fullMaps Full board states
 * @param figureMaps Figure (piece) positions
 * @return std::vector of generated moves
 */
std::vector<uint64_t>
AccessMappingCPU(uint64_t (*func)(int, uint64_t), const uint64_t count, const std::vector<uint64_t> &fullMaps,
                 const std::vector<uint64_t> &figureMaps) {
    std::vector<uint64_t> results{};

    for (uint64_t idx = 0; idx < count; ++idx) {
        const uint64_t fullMap = fullMaps[idx];
        uint64_t figureMap = figureMaps[idx];

        while (figureMap != 0) {
            const int msbPos = cpu::ExtractMsbPosCPU(figureMap);

            const auto result = func(msbPos, fullMap);
            results.push_back(result);

            figureMap ^= cuda_MaxMsbPossible >> msbPos;
        }
    }

    return results;
}


/**
 * @brief Runs correctness test for a specific move generation mapping.
 *
 * Compares GPU move generation results with a CPU reference implementation
 * to verify correctness of the mapping strategy.
 *
 * @tparam MapT Move generation mapping type to test
 * @param func CPU reference move generation function
 * @param records Test case data containing board states
 * @param title Descriptive name of the mapping being tested
 */
template<typename MapT>
void RunCorrectnessTestOnMap(uint64_t (*func)(int, uint64_t), const cpu::MapCorrecntessRecordsPack &records,
                             const std::string &title) {
    static constexpr int MAX_DISPLAYS = 5;

    std::cout << "Running correctness test on << " << title << "..." << std::endl;

    auto [recordCount, fullMaps, figureMaps, correctnessMap] = records;
    thrust::device_vector<uint64_t> dFullMaps(fullMaps);
    thrust::device_vector<uint64_t> dFigureMaps(figureMaps);

    uint64_t mCount{};
    for (size_t i = 0; i < recordCount; ++i) {
        mCount += correctnessMap[i].size();
    }

    thrust::device_vector<uint64_t> dResults(mCount, 0);

    AccessMapping<MapT><<<1, 1>>>(thrust::raw_pointer_cast(dResults.data()), recordCount,
                                  thrust::raw_pointer_cast(dFullMaps.data()),
                                  thrust::raw_pointer_cast(dFigureMaps.data()));
    CUDA_TRACE_ERROR(cudaGetLastError());

    const auto cpuResults = AccessMappingCPU(func, recordCount, fullMaps, figureMaps);

    const thrust::host_vector<uint64_t> hResults = dResults;


    if (cpuResults.size() != hResults.size()) {
        std::cerr << "Results size mismatch!" << std::endl;
    }

    const auto size = std::min(cpuResults.size(), hResults.size());

    uint32_t displays = 0;
    uint64_t errors = 0;
    for (size_t i = 0; i < size; ++i) {
        if (displays < MAX_DISPLAYS && cpuResults[i] != hResults[i]) {
            std::cerr << "Results mismatch at index " << i << std::endl;
            displays++;

            std::cout << "Correct CPU map: " << std::endl;
            cpu::DisplayBitBoardCPU(cpuResults[i]);

            std::cout << "Calculated GPU map: " << std::endl;
            cpu::DisplayBitBoardCPU(hResults[i]);
        }

        errors += cpuResults[i] != hResults[i];
    }

    std::cout << std::format("Correctness test for {} finished with {} errors, where visited {} / {} moves", title,
                             errors, size, mCount) << std::endl;
}

/**
 * @brief Attempts to read a file path for correctness test data, with user interaction fallback.
 *
 * Tries to read a default file path, and if unsuccessful, prompts the user to
 * provide an alternative file path for the test data.
 *
 * @param defaultPath Initial file path to attempt reading
 * @param prompt Descriptive prompt to display if user input is required
 * @return cpu::MapCorrecntessRecordsPack containing the test data
 * @throws std::runtime_error if no valid file path is provided
 */
cpu::MapCorrecntessRecordsPack TryReadingFilePath(const std::string &defaultPath, const std::string &prompt) {
    try {
        return cpu::ReadMagicCorrectnessTestFile(defaultPath);
    } catch (const std::exception &e) {}

    std::string line;
    std::cout << "Enter the path to the file with the correctness test data: " << prompt << std::endl;
    std::cout << "Or press enter to exit" << std::endl;

    if (line.empty()) {
        throw std::runtime_error("No file path provided, exiting...");
    }

    std::getline(std::cin, line);
    return cpu::ReadMagicCorrectnessTestFile(line);
}

/**
 * @brief Internal function for running Fancy Magic correctness tests.
 *
 * Executes correctness tests for different move generation mapping types
 * using predefined test data files.
 */
void FancyMagicTestCorrectness_() {
    static constexpr std::string_view BISHOP_PATH = "./test_data/corr2";
    static constexpr std::string_view ROOK_PATH = "./test_data/corr4";

    try {
        const auto records = TryReadingFilePath(std::string(BISHOP_PATH), " for the BishopMap");
        RunCorrectnessTestOnMap<BishopMap>(cpu::AccessCpuBishopMap, records, "BishopMap");
        std::cout << std::string(80, '-') << std::endl;
        RunCorrectnessTestOnMap<BishopMapRuntime>(cpu::AccessCpuBishopMap, records, "BishopMapRuntime");
    } catch (const std::exception &e) {
        std::cout << e.what();
    }

    std::cout << std::string(80, '-') << std::endl;

    try {
        const auto records = TryReadingFilePath(std::string(ROOK_PATH), " for the RookMap");
        RunCorrectnessTestOnMap<RookMap>(cpu::AccessCpuRookMap, records, "RookMap");
        std::cout << std::string(80, '-') << std::endl;
        RunCorrectnessTestOnMap<RookMapRuntime>(cpu::AccessCpuRookMap, records, "RookMapRuntime");
    } catch (const std::exception &e) {
        std::cout << e.what();
    }
}

void FancyMagicTest(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    try {
        std::cout << std::string(80, '-') << std::endl;
        cpu::FancyMagicTest();
        std::cout << std::string(80, '-') << std::endl;
        FancyMagicTest_(threadsAvailable, deviceProps);
        std::cout << std::string(80, '-') << std::endl;
        FancyMagicTestCorrectness_();
        std::cout << std::string(80, '-') << std::endl;
    } catch (const std::exception &e) {
        std::cerr << "Fancy Magic Test failed with exception: " << e.what() << std::endl;
    }
}
