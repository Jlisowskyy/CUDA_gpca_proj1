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

static constexpr uint32_t NUM_POS_EXPECTED_MOVE = std::size(TEST_FEN_EXPECTED_MOVE_MAP);
static constexpr uint32_t TEST_TIME_EXPECTED_MOVE = 1000;
static constexpr uint32_t TEST_TIME_ASSERT_RUN = 10;
static constexpr uint32_t NUM_BOARDS_ASSERT_TEST = 1000;
static constexpr uint32_t BAR_WIDTH = 80;

template<class ENGINE_T>
static bool RunExpectedMoveTestOnEngineOnce(const std::string &fen, const std::string &expectedResult,
                                            const std::string &desc, ProgressBar &bar) {
    /* prepare components */
    auto board = cuda_Board(cpu::TranslateFromFen(fen));

    if (g_GlobalState.WriteExtensiveInfo) {
        const std::string msg = "Running expected move test on engine: " + std::string(ENGINE_T::GetName()) +
                                " on position: " + fen + " with desc: " + desc;
        bar.WriteLine(msg);
    }

    G_USE_DEFINED_SEED = true;

    ENGINE_T engine{board, ENGINE_T::GetPreferredThreadsCount()};
    engine.MoveSearchStart();
    std::this_thread::sleep_for(std::chrono::milliseconds(TEST_TIME_EXPECTED_MOVE));
    const cuda_Move move = engine.MoveSearchWait();

    G_USE_DEFINED_SEED = false;

    if (g_GlobalState.WriteExtensiveInfo) {
        engine.DisplayResults(0);
    }

    const bool result = expectedResult == move.GetPackedMove().GetLongAlgebraicNotationCPU();

    if (!result) {
        const std::string msg = "Mcts Correctness test failed on position: " + fen + ", desc: " + desc + "\n" +
                                "Expected move: " + expectedResult + "\n" +
                                "Got move: " + move.GetPackedMove().GetLongAlgebraicNotationCPU();
        bar.WriteLine(msg);
    }

    return result;
}

template<class ENGINE_T>
static void RunAssertTestOnEngineOnce(const std::string &fen, ProgressBar &bar) {
    /* prepare components */
    auto board = cuda_Board(cpu::TranslateFromFen(fen));

    if (g_GlobalState.WriteExtensiveInfo) {
        const std::string msg = "Running assert test on engine: " + std::string(ENGINE_T::GetName()) + " on position: "
                                + fen;
        bar.WriteLine(msg);
    }

    G_USE_DEFINED_SEED = true;

    ENGINE_T engine{board, ENGINE_T::GetPreferredThreadsCount()};
    engine.MoveSearchStart();
    std::this_thread::sleep_for(std::chrono::milliseconds(TEST_TIME_ASSERT_RUN));
    [[maybe_unused]] const auto move = engine.MoveSearchWait();

    G_USE_DEFINED_SEED = false;

    if (g_GlobalState.WriteExtensiveInfo) {
        bar.WriteLine("Finished assert test on engine: " + std::string(ENGINE_T::GetName()) + " on position: " + fen);
    }
}

static void TestMctsCorrectnessExpectedMove_() {
    std::cout << "Running MctsCorrectness test on move == expected move case...\n" << std::endl;

    bool result{};
    ProgressBar bar(NUM_POS_EXPECTED_MOVE, BAR_WIDTH);
    bar.Start();
    for (const auto &[fen, expectedMove, desc]: TEST_FEN_EXPECTED_MOVE_MAP) {
        result |= RunExpectedMoveTestOnEngineOnce<MctsEngine<EngineType::GPU0> >(fen, expectedMove, desc, bar);
        bar.Increment();
        result |= RunExpectedMoveTestOnEngineOnce<MctsEngine<EngineType::GPU1> >(fen, expectedMove, desc, bar);
        bar.Increment();
    }

    result = !result;
    std::cout << "Mcts Correctness test finished with result: " << (result ? "SUCCESS" : "FAILURE") << std::endl;
}

static void TestMctsCorrectnessAssertBigRun_() {
    std::cout << "Running MctsCorrectness test on assert test case...\n" << std::endl;

    auto fenDb = LoadFenDb();
    fenDb.resize(NUM_BOARDS_ASSERT_TEST);

    ProgressBar bar(fenDb.size() * 2, BAR_WIDTH);
    bar.Start();
    for (const auto &fen: fenDb) {
        RunAssertTestOnEngineOnce<MctsEngine<EngineType::GPU0> >(fen, bar);
        bar.Increment();
        RunAssertTestOnEngineOnce<MctsEngine<EngineType::GPU1> >(fen, bar);
        bar.Increment();
    }
}

void TestMctsCorrectness([[maybe_unused]] uint32_t threadsAvailable,
                         [[maybe_unused]] const cudaDeviceProp &deviceProps) {
    try {
        // TestMctsCorrectnessAssertBigRun_();
        TestMctsCorrectnessExpectedMove_();
    } catch (const std::exception &e) {
        std::cout << "TestMctsCorrectness failed: " << e.what() << std::endl;
    }
}