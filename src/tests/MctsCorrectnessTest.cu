//
// Created by Jlisowskyy on 12/12/24.
//

#include "CudaTests.cuh"
#include "../cpu_core/ProgressBar.cuh"
#include "../cpu_core/MctsEngine.cuh"
#include "../ported/CpuUtils.h"
#include "../cuda_core/Move.cuh"

#include <tuple>
#include <string>
#include <iostream>

static std::tuple<std::string, std::string, std::string> TEST_FEN_EXPECTED_MOVE_MAP[]{
    {"8/P7/8/8/8/8/8/k6K w - - 0 1", "a7a8Q", "simplest promo possible"},
    {"8/8/8/4pP2/8/8/8/k6K w - e6 0 1", "f5e6", "winning en passant"},
    {"2bnkn2/3ppp2/2p5/7b/7P/6PP/1q3PBQ/2R1K2R w K - 0 1", "e1g1", "enforced castling"},
    {"3NN2k/4Q3/6K1/8/8/8/8/8 w - - 0 1", "e7f8", "stalemate check"},
    {"r3k2r/ppp2ppp/2n5/3q4/3P4/2N5/PPP2PPP/R3K2R w KQkq - 0 1", "c3d5", "queen for free 1"},
    {"rnb1kbnr/pppppppp/8/8/3qR3/8/PPPPPPPP/RNBQKBN1 b Qkq - 0 1", "e4d5", "queen for free 2"},
    {"r4rk1/1pp2ppp/p1np1n2/2b1p1B1/q1B1P1b1/P1NP1N2/1PP1QPPP/R4RK1 w - - 0 10", "c3a4", "queen for free 3"},
    {"rnb1kbqr/pppp1p1p/4p1p1/8/4N3/2B5/PPPPPPPP/R2QKBNR w KQkq - 0 1", "e4f6", "free fork"},
};

static constexpr uint32_t NUM_POS = std::size(TEST_FEN_EXPECTED_MOVE_MAP);
static constexpr uint32_t TEST_TIME = 1000;
static constexpr uint32_t BAR_WIDTH = 80;

template<class ENGINE_T>
static bool RunTestOnEngineOnce(const std::string &fen, const std::string &expectedResult, const std::string &desc) {
    /* prepare components */
    auto board = cuda_Board(cpu::TranslateFromFen(fen));

    if (g_GlobalState.WriteExtensiveInfo) {
        std::cout << "Running test on engine: " << ENGINE_T::GetName() << " on position: " << fen << std::endl;
        cpu::DisplayBoard(board.DumpToExternal());
    }

    G_USE_DEFINED_SEED = true;

    ENGINE_T engine{board, ENGINE_T::GetPreferredThreadsCount()};
    engine.MoveSearchStart();
    std::this_thread::sleep_for(std::chrono::milliseconds(TEST_TIME));
    const cuda_Move move = engine.MoveSearchWait();

    G_USE_DEFINED_SEED = false;

    if (g_GlobalState.WriteExtensiveInfo) {
        engine.DisplayResults(0);
    }

    const bool result = expectedResult == move.GetPackedMove().GetLongAlgebraicNotationCPU();

    if (!result) {
        std::cout << "Mcts Correctness test failed on position: " << fen << std::endl;
        std::cout << "Expected move: " << expectedResult << std::endl;
        std::cout << "Got move: " << move.GetPackedMove().GetLongAlgebraicNotationCPU() << std::endl;
    }

    return result;
}

static void TestMctsCorrectness_() {
    std::cout << "Running MctsCorrectness test...\n" << std::endl;

    bool result{};
    ProgressBar bar(NUM_POS, BAR_WIDTH);
    bar.Start();
    for (const auto &[fen, expectedMove, desc]: TEST_FEN_EXPECTED_MOVE_MAP) {
        result |= RunTestOnEngineOnce<MctsEngine<EngineType::GPU0, true> >(fen, expectedMove, desc);
        bar.Increment();
        result |= RunTestOnEngineOnce<MctsEngine<EngineType::GPU1, true> >(fen, expectedMove, desc);
        bar.Increment();
    }

    result = !result;
    std::cout << "Mcts Correctness test finished with result: " << (result ? "SUCCESS" : "FAILURE") << std::endl;
}

void TestMctsCorrectness([[maybe_unused]] uint32_t threadsAvailable,
                         [[maybe_unused]] const cudaDeviceProp &deviceProps) {
    try {
        TestMctsCorrectness_();
    } catch (const std::exception &e) {
        std::cout << "TestMctsCorrectness failed: " << e.what() << std::endl;
    }
}
