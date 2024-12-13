//
// Created by Jlisowskyy on 05/12/24.
//

#include "CudaTests.cuh"

#include "../cuda_core/cuda_Board.cuh"
#include "../cpu_core/ProgressBar.cuh"
#include "../cpu_core/Mcts.cuh"
#include "../ported/CpuUtils.h"
#include "../cpu_core/MctsEngine.cuh"

#include <iostream>
#include <thread>
#include <iomanip>

static constexpr std::array TestFEN{
    "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
    "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
    "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
    "7k/r2q1ppp/1p1p4/p1bPrPPb/P1PNPR1P/1PQ5/2B5/R5K1 w - - 23 16",
};

//static constexpr uint32_t TEST_TIME = 5000;
static constexpr uint32_t TEST_TIME_EXPECTED_MOVE = 20;

static void RunProcessingAnim(const uint32_t moveTime) {
    static constexpr uint32_t PROG_BAR_STEP_MS = 50;
    uint32_t timeLeft = moveTime;

    ProgressBar bar(moveTime, 50);
    while (timeLeft) {
        const uint32_t curStep = std::min(PROG_BAR_STEP_MS, timeLeft);

        std::this_thread::sleep_for(std::chrono::milliseconds(curStep));

        bar.Increment(curStep);
        timeLeft -= curStep;
    }
}

template<class ENGINE_T1>
static double RunTestOnEngineOnce(const uint32_t moveTime, const std::string &fen) {
    /* prepare components */
    auto board = cuda_Board(cpu::TranslateFromFen(fen));

    std::cout << "Running test on engine: " << ENGINE_T1::GetName() << " on position: " << fen << std::endl;
    cpu::DisplayBoard(board.DumpToExternal());

    G_USE_DEFINED_SEED = true;

    ENGINE_T1 engine{board, ENGINE_T1::GetPreferredThreadsCount()};
    engine.MoveSearchStart();
    RunProcessingAnim(moveTime);
    [[maybe_unused]] const auto move = engine.MoveSearchWait();
    engine.DisplayResults(0);

    const uint64_t simulations = mcts::g_SimulationCounter.load();
    const double scoreMS = double(simulations) / double(moveTime);
    const double scoreS = scoreMS * 1000.0;

    std::cout << "ENGINE: " << ENGINE_T1::GetName() << " made: " << simulations << " simulations in " << moveTime
            << "ms that gives us: " << scoreS << " simulations per S and " << scoreMS << " simulations per MS"
            << std::endl;


    G_USE_DEFINED_SEED = false;

    return scoreMS;
}

template<class ENGINE_T1>
static std::vector<double> RunTestsGroup(const uint32_t moveTime) {
    std::vector<double> rv{};
    for (const std::string &fen: TestFEN) {
        const double result = RunTestOnEngineOnce<ENGINE_T1>(moveTime, fen);
        PolluteCache();
        rv.push_back(result);
    }
    return rv;
}

static void TestMCTSEngines_() {
    const auto resultGPU1 = RunTestsGroup<MctsEngine<EngineType::GPU1, true> >(TEST_TIME_EXPECTED_MOVE);
    const auto resultGPU0 = RunTestsGroup<MctsEngine<EngineType::GPU0, true> >(TEST_TIME_EXPECTED_MOVE);
    const auto resultCPU = RunTestsGroup<MctsEngine<EngineType::CPU, true> >(TEST_TIME_EXPECTED_MOVE);

    // Pretty print results as a table
    std::cout << std::left << std::setw(30) << "FEN Position"
            << std::setw(20) << "GPU0 (sim/ms)"
            << std::setw(20) << "GPU1 (sim/ms)"
            << std::setw(20) << "CPU (sim/ms)"
            << std::endl;

    std::cout << std::string(90, '-') << std::endl;

    for (size_t i = 0; i < TestFEN.size(); ++i) {
        std::cout << std::left
                << std::setw(30) << (std::string(TestFEN[i]).substr(0, 27) + "...")
                << std::setw(20) << std::fixed << std::setprecision(2) << resultGPU0[i]
                << std::setw(20) << std::fixed << std::setprecision(2) << resultGPU1[i]
                << std::setw(20) << std::fixed << std::setprecision(2) << resultCPU[i]
                << std::endl;
    }

    // Calculate and print averages
    std::cout << std::string(90, '=') << std::endl;

    double avgGPU0 = std::accumulate(resultGPU0.begin(), resultGPU0.end(), 0.0) / resultGPU0.size();
    double avgGPU1 = std::accumulate(resultGPU1.begin(), resultGPU1.end(), 0.0) / resultGPU1.size();
    double avgCPU = std::accumulate(resultCPU.begin(), resultCPU.end(), 0.0) / resultCPU.size();

    std::cout << std::left
            << std::setw(30) << "Average"
            << std::setw(20) << std::fixed << std::setprecision(2) << avgGPU0
            << std::setw(20) << std::fixed << std::setprecision(2) << avgGPU1
            << std::setw(20) << std::fixed << std::setprecision(2) << avgCPU
            << std::endl;
}

void TestMCTSEngines([[maybe_unused]] uint32_t threadsAvailable, [[maybe_unused]] const cudaDeviceProp &deviceProps) {
    try {
        TestMCTSEngines_();
    } catch (const std::exception &e) {
        std::cerr << "MCTS Test failed with exception: " << e.what() << std::endl;
    }
}
