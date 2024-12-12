//
// Created by Jlisowskyy on 12/12/24.
//

#include "CudaTests.cuh"
#include "../cpu_core/GlobalState.cuh"
#include "../cpu_core/ThreadPool.cuh"

#include <iostream>
#include <thread>
#include <mutex>
#include <tuple>

/**
 * @brief Pre-generated set of totally random numbers for testing purposes.
 */
static constexpr uint32_t RandomNumbers[]{
    2063823207,
    1182548513,
    4057511361,
    2153812279,
    768812903,
    3339868802,
    1167018722,
    1308166594,
    820605297,
    2076623162,
    3742730924,
    741708029,
    3255519063,
    2313708676,
    1966528379,
    3129001482,
    2141941530,
    3611520571,
    3058975218,
    1282892888,
    2008421873,
    1816075310,
    2859363133,
    528953541,
    408420833,
    397434091,
    680424058,
    2799847613,
    124557351,
    747609092,
    2284323046,
    3359022812,
    3620451222,
    2064002864,
    3426784128,
    3306674905,
    3163243787,
    3798635324,
    1277835062,
    234703502,
    2724394409,
    3200777071,
    3368564977,
    2688312130,
    4217473380,
    51758553,
    3852236456,
    471580846,
    1359970170,
    763185670,
    2252720665,
    4150747897,
    377508426,
    45717194,
    433487071,
    110102635,
    1184634085,
    3260866623,
    1534691670,
    1146313058,
    3447270171,
    2395614373,
    2195343791,
    3631149583,
    2339946860,
    3436745484,
    3442348483,
    4143796618,
    664337465,
    1038902433,
    1666520730,
    1456096185,
    422892648,
    2208488383,
    1016671267,
    3592720118,
    2323602253,
    43045522,
    1797806454,
    2219511305,
    2228145070,
    4114742732,
    783227404,
    2427935326,
    1296083206,
    2473336364,
    3124898665,
    2872682665,
    331480460,
    2628679589,
    449771748,
    4193214750,
    606091058,
    1500737839,
    1688923783,
    1591665343,
    943725959,
    3499891146,
    3928636713,
    1602383769,
    183770071,
    2353129719,
    3160361471,
    373610396,
    346956788,
    2614768451,
    1818035877,
    1084585244,
    1955012025,
    75511072,
    3817686422,
    2778449695,
    347725055,
    1691002597,
    3229031211,
    120142241,
    1846923576,
    678333044,
    346439010,
    3310474164,
    2015632321,
    3230516235,
    3904876383,
    2378519384,
    180712088,
    55698817,
    437157683,
    1200921089,
    3374980082,
    1813268035,
    1364726672,
    1671961832,
    3676728784,
    2145403348,
    3075980517,
    1776290707,
    1071031766,
    265548057,
    4046791371,
    3548482465,
    280726541,
    4011378603,
    538431526,
    2074015175,
    1110003059,
    2038455250,
    3733934215,
    19820952,
    3968945436,
    196398884,
    1331124936,
    3166901754,
    1374882342,
    2415366749,
    1330745944,
    3111015074,
    2595959586,
    769962667,
    3843726808,
    2096934626,
    3149062658,
    598784041,
    327927359,
    2732866665,
    4395800,
    4204876796,
    3972663807,
    2192167496,
    3899496283,
    1759630068,
    1749968379,
    4125043534,
    2374783619,
    754518526,
    642134128,
    3969636843,
    1247931032,
    1887077070,
    894799970,
    3822283113,
    502701362,
    2084794143,
    652967625,
    2354439812,
    544491279,
    3150164819,
    503857363,
    1417814573,
    2170409133,
    2740587685,
    2889457902,
    1551883817,
    689688196,
    818814462,
    1971093583,
    1638847107,
    2640804270,
    3664188662,
    617863102,
    4255706527,
};

static constexpr uint64_t NUM_NUMBERS = std::size(RandomNumbers);
static constexpr uint64_t NUM_TRIES = 1'000'000;
static constexpr double MAX_DEVIATION = 1.1;
static constexpr uint64_t MAX_NUM_MOVES = 80;

std::tuple<bool, double> RunSingleTest_(const uint32_t seed, std::mutex &mut) {
    bool result{};
    double totalMaxDeviation{1.0};

    if (g_GlobalState.WriteExtensiveInfo) {
        std::cout << "Running test with seed: " << seed << std::endl;
    }

    for (uint64_t numMoves = 2; numMoves < MAX_NUM_MOVES; ++numMoves) {
        if (g_GlobalState.WriteExtensiveInfo) {
            std::cout << "Running test with numMoves: " << numMoves << std::endl;
        }

        uint32_t curSeed = seed;
        std::array<uint64_t, MAX_NUM_MOVES> samples{};

        /* Generate samples */
        for (uint64_t retries = 0; retries < NUM_TRIES; ++retries) {
            simpleRand(curSeed);

            ++samples[curSeed % numMoves];
        }

        if (g_GlobalState.WriteExtensiveInfo) {
            std::cout << "Probabilities acquired:" << std::endl;
        }

        /* calculate probabilities */
        std::array<double, MAX_NUM_MOVES> probabilities{};
        for (uint64_t i = 0; i < numMoves; ++i) {
            probabilities[i] = double(samples[i]) / double(NUM_TRIES);
        }

        /* calculate max deviation */
        double maxDeviation = 0.0;
        uint32_t maxDeviationMoves = 0;
        const double expectedProb = 1.0 / double(numMoves);

        for (uint64_t i = 0; i < numMoves; ++i) {
            const double upperDev = probabilities[i] / expectedProb;
            const double lowerDev = expectedProb / probabilities[i];

            const double localMaxDev = std::max(upperDev, lowerDev);

            if (localMaxDev > maxDeviation) {
                maxDeviation = localMaxDev;
                maxDeviationMoves = i;
            }
        }

        if (g_GlobalState.WriteExtensiveInfo) {
            std::cout << "Max deviation: " << maxDeviation << std::endl;
        }

        if (maxDeviation > MAX_DEVIATION) {
            std::lock_guard lock(mut);

            std::cout << "Test failed with seed: " << seed << " and numMoves: " << numMoves << std::endl;
            std::cout << "Max deviation: " << maxDeviation << std::endl;
            std::cout << "Max deviation moves: " << maxDeviationMoves << std::endl;

            std::cout << "Probabilities and samples:" << std::endl;
            for (uint64_t i = 0; i < numMoves; ++i) {
                std::cout << "{" << i << ", " << samples[i] << ", " << probabilities[i] << "} ";
            }
            std::cout << std::endl;

            result |= true;
        }

        totalMaxDeviation = std::max(totalMaxDeviation, maxDeviation);
    }

    return {!result, totalMaxDeviation};
}

static void TestRandomGen_() {
    const uint32_t numThreads = g_GlobalState.WriteExtensiveInfo ? 1 : std::thread::hardware_concurrency();
    ThreadPool pool(numThreads);

    bool testResult{};
    double maxDeviation{};
    std::mutex mut{};

    pool.RunThreads([&](const uint32_t tIdx) {
        for (uint32_t idx = tIdx; idx < NUM_NUMBERS; idx += numThreads) {
            const auto [result,dev] = RunSingleTest_(RandomNumbers[idx], mut);

            mut.lock();
            testResult |= result;
            maxDeviation = std::max(maxDeviation, dev);
            mut.unlock();
        }
    });

    pool.Wait();
    std::cout << "RandomGen test finished" << std::endl;
    std::cout << "Test result: " << (testResult ? "FAILED" : "PASSED") << std::endl;
    std::cout << "Max deviation: " << maxDeviation << std::endl;
}

void TestRandomGen([[maybe_unused]] uint32_t threadsAvailable, [[maybe_unused]] const cudaDeviceProp &deviceProps) {
    try {
        TestRandomGen_();
    } catch (const std::exception &e) {
        std::cerr << "TestRandomGen failed with exception: " << e.what() << std::endl;
    }
}
