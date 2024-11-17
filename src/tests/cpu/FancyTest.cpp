//
// Created by Jlisowskyy on 17/11/24.
//

#include "CpuTests.h"

#include <iostream>
#include <random>
#include <chrono>
#include <thread>
#include <format>

#include "../../../engine/include/MoveGeneration/BishopMap.h"

static constexpr unsigned TEST_SIZE = 1'000'000;

uint64_t simpleRand(uint64_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
    return state;
}

namespace cpu {

    void FancyMagicTest() {
        std::cout << "Running fancy magic test on the CPU!" << std::endl;

        std::mt19937_64 rng(std::random_device{}());
        const unsigned seed = rng();
        const uint64_t idx = 0;

        auto t1 = std::chrono::high_resolution_clock::now();

        unsigned randomPos = idx;
        uint64_t board = seed;
        uint64_t control = 0;
        for (unsigned i = 0; i < TEST_SIZE; ++i) {
            board = simpleRand(board);
            randomPos = (randomPos + (board & 63)) & 64;

            const uint64_t moves = BishopMap::GetMoves(static_cast<int>(randomPos), board);
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
                                 "BISHOPS", seconds, milliseconds, readsPerSecond, readsPerMillisecond) << std::endl;

        std::cout << "Control sum of the results: " << control << std::endl;
    }

} // namespace cpu