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

using TestFunc = void (*)(uint32_t, const cudaDeviceProp &deviceProps);

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
void FancyMagicTest(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

/**
* @brief Performs correctness testing of move generation algorithms on the GPU.
*
* Validates the accuracy of chess move generation implemented on CUDA devices
* by comparing generated moves against expected results. Generated by CPU libs.
*
* @param threadsAvailable Number of CUDA threads available for the test
* @param deviceProps CUDA device properties used for kernel launch configuration
*/
void MoveGenTest(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

/**
 * @brief Executes performance benchmarking for move generation on the GPU.
 *
 * Measures the computational efficiency and throughput of move generation
 * algorithms on CUDA-enabled devices.
 *
 * @param threadsAvailable Number of CUDA threads available for the test
 * @param deviceProps CUDA device properties used for kernel launch configuration
 */
void MoveGenPerfTest(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

/**
 * @brief Combines correctness and performance testing for move generation.
 *
 * Sequentially runs move generation correctness test followed by
 * performance benchmarking to provide comprehensive algorithm evaluation.
 *
 * @param threadsAvailable Number of CUDA threads available for the test
 * @param deviceProps CUDA device properties used for kernel launch configuration
 */
inline void MoveGenCorPerfTest(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    MoveGenTest(threadsAvailable, deviceProps);
    MoveGenPerfTest(threadsAvailable, deviceProps);
}

void TestMCTSEngines(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

void TestRandomGen(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

void TestMctsCorrectness(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

inline void TestFullCorrectness(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    TestMctsCorrectness(threadsAvailable, deviceProps);
    MoveGenTest(threadsAvailable, deviceProps);
    TestRandomGen(threadsAvailable, deviceProps);
    FancyMagicTest(threadsAvailable, deviceProps);
}

// ------------------------------
// Test map
// ------------------------------

/**
 * @brief Mapping of available CUDA performance tests for CLI interaction.
 *
 * This static const unordered_map provides a centralized registry of CUDA tests,
 * allowing dynamic test selection and information retrieval via command-line interface.
 *
 * Map Structure:
 * - Key (std::string): Unique test identifier used for CLI test selection
 *   - Used to specify which test to run when invoking the test suite
 *
 * - Value (std::tuple<std::string, std::string, TestFunc>): Test metadata and execution function
 *   1. First element: Human-readable test title
 *   2. Second element: Detailed test description
 *   3. Third element: Function pointer to the test implementation
 *
 * @example
 * - "magic_test" runs FancyMagicTest()
 * - "move_gen" runs MoveGenTest()
 */
static const std::unordered_map<std::string, std::tuple<std::string, std::string, TestFunc> > g_CudaTestsMap = {
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
            "MoveGen Performance and Correctness Test",
            "Tests first the correctness of the move generation and performance after it",
            &MoveGenCorPerfTest
        )
    },
    {
        "mcts_perf",
        std::make_tuple(
            "Mcts Engine Performance Test",
            "Tests number of simulation played inside MCTS search of various implementations",
            &TestMCTSEngines
        )
    },
    {
        "random_gen",
        std::make_tuple(
            "Random Generator Test",
            "Tests the correctness of the random generator on the GPU",
            &TestRandomGen
        )
    },
    {
        "mcts_correctness",
        std::make_tuple(
            "MCTS Correctness Test",
            "Tests the correctness of the MCTS implementation",
            &TestMctsCorrectness
        )
    },
    {
        "full_correctness",
        std::make_tuple(
            "Full Correctness Test",
            "Tests the correctness of all implemented algorithms",
            &TestFullCorrectness
        )
    }
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
std::tuple<uint32_t, uint32_t> GetDims(uint32_t threadsAvailable, const cudaDeviceProp &deviceProps);

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
 * @return std::vector<uint32_t> containing randomly generated seeds
 */
std::vector<uint32_t> GenSeeds(uint32_t size);

void GenSeeds(uint32_t *out, uint32_t size);

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

// ------------------------------
// Test constants
// ------------------------------

static constexpr uint64_t DEFAULT_TEST_SEED = 0xDEADC0D3;
extern bool G_USE_DEFINED_SEED;

#endif //SRC_CUDATESTS_CUH
