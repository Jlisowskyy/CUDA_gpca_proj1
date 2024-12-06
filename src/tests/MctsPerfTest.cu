//
// Created by Jlisowskyy on 05/12/24.
//

#include "CudaTests.cuh"

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/cuda_Board.cuh"
#include "cpu_core/ProgressBar.cuh"
#include "../cpu_core/Mcts.cuh"
#include "../ported/CpuUtils.h"
#include "../cpu_core/MctsEngine.cuh"

#include <iostream>
#include <thread>

static constexpr std::array TestFEN{
        "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1",
        "r3k2r/p1ppqpb1/bn2pnp1/3PN3/1p2P3/2N2Q1p/PPPBBPPP/R3K2R w KQkq - 0 1",
        "8/2p5/3p4/KP5r/1R3p1k/8/4P1P1/8 w - - 0 1",
        "7k/r2q1ppp/1p1p4/p1bPrPPb/P1PNPR1P/1PQ5/2B5/R5K1 w - - 23 16",
};

static constexpr __uint32_t TEST_TIME = 5000;

static void RunProcessingAnim(__uint32_t moveTime) {
    static constexpr __uint32_t PROG_BAR_STEP_MS = 50;
    __uint32_t timeLeft = moveTime;

    ProgressBar bar(moveTime, 50);
    while (timeLeft) {
        const __uint32_t curStep = std::min(PROG_BAR_STEP_MS, timeLeft);

        std::this_thread::sleep_for(std::chrono::milliseconds(curStep));

        bar.Increment(curStep);
        timeLeft -= curStep;
    }
}

template<class ENGINE_T1>
void RunTestOnEngineOnce(__uint32_t moveTime, const std::string &fen) {

    /* prepare components */
    cuda_Board board = cuda_Board(cpu::TranslateFromFen(fen));

    std::cout << "Running test on engine: " << ENGINE_T1::GetName() << " on position: " << fen << std::endl;
    cpu::DisplayBoard(board.DumpToExternal());

    G_USE_DEFINED_SEED = true;

    ENGINE_T1 engine{board, ENGINE_T1::GetPreferredThreadsCount()};
    engine.MoveSearchStart();
    RunProcessingAnim(moveTime);
    [[maybe_unused]] const auto move = engine.MoveSearchWait();
    engine.DisplayResults();

    const __uint64_t simulations = mcts::g_SimulationCounter.load();
    const double scoreMS = double(simulations) / double(moveTime);
    const double scoreS = scoreMS * 1000.0;

    std::cout << "ENGINE: " << ENGINE_T1::GetName() << " made: " << simulations << " simulations in " << moveTime
              << "ms that gives us: " << scoreS << " simulations per S and " << scoreMS << " simulations per MS"
              << std::endl;


    G_USE_DEFINED_SEED = false;
}

template<class ENGINE_T1>
void RunTestsGroup(__uint32_t moveTime) {
    for (const std::string &fen: TestFEN) {
        RunTestOnEngineOnce<ENGINE_T1>(moveTime, fen);
    }
}

void TestMCTSEngines_() {
    RunTestsGroup<MctsEngine<EngineType::CPU>>(TEST_TIME);
    RunTestsGroup<MctsEngine<EngineType::GPU0>>(TEST_TIME);
    RunTestsGroup<MctsEngine<EngineType::GPU1>>(TEST_TIME);
}

void TestMCTSEngines(__uint32_t threadsAvailable, const cudaDeviceProp &deviceProps) {
    try {
        TestMCTSEngines_();
    } catch (const std::exception &e) {
        std::cerr << "Fancy Magic Test failed with exception: " << e.what() << std::endl;
    }
}
