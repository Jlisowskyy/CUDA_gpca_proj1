//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_CUDATESTS_CUH
#define SRC_CUDATESTS_CUH

#include "../cuda_core/Helpers.cuh"

#include <unordered_map>
#include <string>
#include <tuple>
#include <vector>

// ------------------------------
// Test defines
// ------------------------------

struct cudaDeviceProp;

using TestFunc = void (*)(__uint32_t, const cudaDeviceProp &deviceProps);

// ------------------------------
// Test functions
// ------------------------------

/**
 * @brief Performs a comprehensive test of "Fancy Magic" bitboard mapping techniques.
 *
 * Conducts performance and correctness tests for different chess piece move generation
 * mapping strategies on the GPU, including RookMap, BishopMap, and their runtime variants.
 *
 * @param threadsAvailable Number of CUDA threads available for the test
 * @param deviceProps CUDA device properties for kernel launch configuration
 *
 * @note Includes cache pollution between tests to minimize performance interference
 */
void FancyMagicTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

void MoveGenTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

void MoveGenPerfTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

inline void MoveGenCorPerfTest(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    MoveGenTest(threadsAvailable, deviceProps);
    MoveGenPerfTest(threadsAvailable, deviceProps);
}

/* test mapping */
static const std::unordered_map<std::string, std::tuple<std::string, std::string, TestFunc>> CudaTestsMap = {
        {
                "magic_test",
                std::make_tuple(
                        "Fancy Magic Test",
                        "Measures average access times of Fancy mappings on the GPU as well as correctness",
                        &FancyMagicTest
                )
        },
        {
                "move_gen",
                std::make_tuple(
                        "MoveGen Test",
                        "Tests the correctness of the move generation on the GPU",
                        &MoveGenTest
                )
        },
        {
                "move_perf",
                std::make_tuple(
                        "MoveGen Performance Test",
                        "Tests the performance of the move generation on the GPU",
                        &MoveGenPerfTest
                )
        },
        {
                "move_gen_perf",
                std::make_tuple(
                        "MoveGen PerfCor Test",
                        "Tests first the correctness of the move generation and performance after it",
                        &MoveGenCorPerfTest
                )
        },
};

// ------------------------------
// Helper functions
// ------------------------------

/**
 * @brief Calculates optimal CUDA kernel launch dimensions based on device properties.
 *
 * Determines the number of thread blocks and threads per block to maximize
 * GPU utilization, considering the device's multiprocessor characteristics.
 *
 * @param threadsAvailable Total number of threads available for computation
 * @param deviceProps CUDA device properties used for dimensioning
 * @return std::tuple containing {number of blocks, threads per block}
 *
 * @note Prints diagnostic information about thread and block utilization
 */
std::tuple<__uint32_t, __uint32_t> GetDims(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

/**
 * @brief Loads a chess position database from a predefined file path.
 *
 * Locates and reads a FEN database file relative to the
 * current source file's directory. Useful for providing test positions for chess-related
 * CUDA kernel tests.
 *
 * @return std::vector containing loaded chess positions
 * @throws std::filesystem::filesystem_error if file cannot be found or read
 */
std::vector<std::string> LoadFenDb();

/**
 * @brief Generates a vector of random 32-bit unsigned integer seeds.
 *
 * Creates a collection of cryptographically unpredictable random seeds using
 * a Mersenne Twister random number generator seeded by std::random_device.
 *
 * @param size Number of random seeds to generate
 * @return std::vector<__uint32_t> containing randomly generated seeds
 */
std::vector<__uint32_t> GenSeeds(__uint32_t size);

/**
 * @brief Pollutes the CPU and GPU caches to minimize performance interference in benchmarking.
 *
 * This function generates random data and performs a computation to effectively flush
 * existing cache contents, helping to create more consistent performance measurements
 * in subsequent tests by reducing cache-related variability.
 *
 * @note Uses a predefined number of rounds and data size to ensure thorough cache pollution.
 */
void PolluteCache();

#endif //SRC_CUDATESTS_CUH