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

static std::tuple<std::string, std::string> TEST_FEN_EXPECTED_MOVE_MAP[]{
    {"", ""}
};

static constexpr uint32_t NUM_POS = std::size(TEST_FEN_EXPECTED_MOVE_MAP);
static constexpr uint32_t TEST_TIME = 1000;
static constexpr uint32_t BAR_WIDTH = 80;

template<class ENGINE_T>
static bool RunTestOnEngineOnce(const std::string &fen, const std::string &expectedResult) {
    /* prepare components */
    auto board = cuda_Board(cpu::TranslateFromFen(fen));

    if (g_GlobalState.WriteExtensiveInfo) {
        std::cout << "Running test on engine: " << ENGINE_T::GetName() << " on position: " << fen << std::endl;
        cpu::DisplayBoard(board.DumpToExternal());
    }

    G_USE_DEFINED_SEED = true;

    ENGINE_T engine{board, ENGINE_T::GetPreferredThreadsCount()};
    engine.MoveSearchStart();
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
    std::cout << "Running MctsCorrectness test" << std::endl;

    bool result{};
    ProgressBar bar(NUM_POS, BAR_WIDTH);
    bar.Start();
    for (const auto &[fen, expectedMove]: TEST_FEN_EXPECTED_MOVE_MAP) {
        result |= RunTestOnEngineOnce<MctsEngine<EngineType::GPU0, true> >(fen, expectedMove);
        bar.Increment();
        result |= RunTestOnEngineOnce<MctsEngine<EngineType::GPU1, true> >(fen, expectedMove);
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
