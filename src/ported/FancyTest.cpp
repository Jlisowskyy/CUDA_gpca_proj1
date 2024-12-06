//
// Created by Jlisowskyy on 17/11/24.
//

#include "CpuTests.h"

#include <iostream>
#include <random>
#include <thread>
#include <format>

#include "../../engine/include/MoveGeneration/BishopMap.h"
#include "../../engine/include/MoveGeneration/RookMap.h"
#include "../../engine/include/TestsAndDebugging/MapCorrectnessTest.h"
#include "../../engine/include/TestsAndDebugging/MapPerformanceTest.h"

static constexpr unsigned TEST_SIZE = 1'000'000;

/**
 * @brief Simple pseudo-random number generator with XOR-shift algorithm.
 *
 * Modifies the input state using bitwise XOR operations to generate the next random value.
 *
 * @param state Reference to the random state, which is modified in-place
 */
static uint64_t simpleRand(uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}


/**
 * @brief Perform a performance test on magic bitboard move generation.
 *
 * Runs a high-performance test generating moves for a specific piece type map
 * (BishopMap or RookMap) using a random seed and multiple iterations.
 *
 * @tparam MapT Type of map to test (BishopMap or RookMap)
 * @param title Descriptive name of the test for logging
 * @param seed Initial random seed for move generation
 */
template<class MapT>
void runTest_(const char *title, const uint64_t seed) {
    const uint64_t idx = 0;

    std::cout << "Running fancy magic " << title << " test on the CPU!" << std::endl;

    auto t1 = std::chrono::high_resolution_clock::now();

    unsigned randomPos = idx;
    uint64_t board = seed;
    uint64_t control = 0;
    for (unsigned i = 0; i < TEST_SIZE; ++i) {
        board = simpleRand(board);
        randomPos = (randomPos + (board & 63)) & 63;

        const uint64_t moves = MapT::GetMoves(static_cast<int>(randomPos), board);
        control += moves;
    }

    auto t2 = std::chrono::high_resolution_clock::now();
    const unsigned threadsUtilized = std::thread::hardware_concurrency();

    const double seconds = std::chrono::duration<double>(t2 - t1).count();
    const double milliseconds = seconds * 1000.0;
    const double readsPerSecond = (TEST_SIZE * threadsUtilized) / seconds;
    const double readsPerMillisecond = (TEST_SIZE * threadsUtilized) / milliseconds;

    std::cout << std::format("Fancy Magic Test for {} took {} seconds, {} milliseconds,\n"
                             "Average performance was:\n"
                             "reads per second: {}\n"
                             "reads per millisecond: {}",
                             title, seconds, milliseconds, readsPerSecond, readsPerMillisecond) << std::endl;

    std::cout << "Control sum of the results: " << control << std::endl;
}

namespace cpu {

    void FancyMagicTest() {
        std::mt19937_64 rng(std::random_device{}());
        const unsigned seed = rng();

        runTest_<BishopMap>("BishopMap", seed);
        std::cout << std::string(80, '-') << std::endl;
        runTest_<RookMap>("RookMap", seed);
    }

    MapCorrecntessRecordsPack ReadMagicCorrectnessTestFile(const std::string &filename) {
        return MapCorrectnessTester::readTestFile(filename);
    }

    MapPerformanceRecordsPack ReadMagicPerformanceTestFile(const std::string &filename) {
        return MapPerformanceTester::readTestFile(filename);
    }

    void DisplayBitBoardCPU(uint64_t board) {
        DisplayMask(board);
    }

} // namespace cpu