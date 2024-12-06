//
// Created by Jlisowskyy on 04/12/24.
//

#include "CpuMoveGen.h"

#include "../../engine/include/MoveGeneration/BishopMap.h"
#include "../../engine/include/MoveGeneration/RookMap.h"
#include "../../engine/include/MoveGeneration/MoveGenerator.h"
#include "../../engine/include/BitOperations.h"

#include <random>

/**
 * @brief Simple pseudo-random number generator with XOR-shift algorithm.
 *
 * Modifies the input state using bitwise XOR operations to generate the next random value.
 *
 * @param state Reference to the random state, which is modified in-place
 */
void simpleRand(uint32_t &state) {
    state ^= state << 13;
    state ^= state >> 7;
    state ^= state << 17;
}


namespace cpu {
    uint64_t AccessCpuRookMap(int msbInd, uint64_t fullMap) {
        return RookMap::GetMoves(msbInd, fullMap);
    }

    uint64_t AccessCpuBishopMap(int msbInd, uint64_t fullMap) {
        return BishopMap::GetMoves(msbInd, fullMap);
    }

    int ExtractMsbPosCPU(uint64_t map) {
        return ExtractMsbPos(map);
    }

    std::vector<external_move> GenerateMoves(const external_board &board) {
        Board bd = TranslateToInternalBoard(board);

        Stack<Move, DEFAULT_STACK_SIZE> s;
        MoveGenerator mech{bd, s};
        auto moves = mech.GetMovesFast();

        std::vector<external_move> result;
        result.reserve(moves.size);
        for (size_t i = 0; i < moves.size; ++i) {
            const auto &mv = moves[i];
            result.push_back({mv.GetPackedMove().DumpContent(), mv.DumpIndexes(), mv.DumpMisc()});
        }

        s.PopAggregate(moves);
        return result;
    }

    bool IsCheck(const external_board &board) {
        Board bd = TranslateToInternalBoard(board);
        ChessMechanics mech{bd};
        return mech.IsCheck();
    }

    Board TranslateToInternalBoard(const external_board &exBoard) {
        Board bd{};
        for (int i = 0; i < 12; ++i) {
            bd.BitBoards[i] = exBoard[i];
        }
        bd.ElPassantField = exBoard[12];
        bd.Castlings = exBoard[13];
        bd.MovingColor = static_cast<int>(exBoard[14]);

        return bd;
    }

    static constexpr __int32_t FIG_VALUES_CPU[Board::BitBoardsCount + 1]{
            100, 330, 330, 500, 900, 10000, -100, -330, -330, -500, -900, -10000, 0
    };

    __uint32_t SimulateGame(const external_board &board) {
        static constexpr __uint32_t MAX_DEPTH = 100;
        static constexpr __uint32_t DRAW = 2;
        static constexpr __uint32_t NUM_ROUNDS_IN_MATERIAL_ADVANTAGE_TO_WIN = 5;
        static constexpr __uint32_t MATERIAL_ADVANTAGE_TO_WIN = 500;

        __uint32_t evalCounters[2]{};
        __uint32_t seed = std::mt19937{
                static_cast<__uint64_t>(std::chrono::steady_clock::now().time_since_epoch().count())}();
        Stack<Move, DEFAULT_STACK_SIZE> s;

        Board bd = TranslateToInternalBoard(board);
        MoveGenerator mech{bd, s};
        __uint32_t eval{};

        for (__uint32_t bIdx = 0; bIdx < Board::BitBoardsCount; ++bIdx) {
            eval += CountOnesInBoard(bd.BitBoards[bIdx]) * FIG_VALUES_CPU[bIdx];
        }

        for (__uint32_t depth = 0; depth < MAX_DEPTH; ++depth) {
            auto moves = mech.GetMovesFast();

            if (moves.size == 0) {
                return mech.IsCheck() ? SwapColor(bd.MovingColor) : DRAW;
            }

            __uint32_t correctedEval = bd.MovingColor == BLACK ? -eval : eval;
            const bool isInWinningRange = correctedEval >= MATERIAL_ADVANTAGE_TO_WIN;
            evalCounters[bd.MovingColor] = isInWinningRange ? evalCounters[bd.MovingColor] + 1 : 0;

            if (evalCounters[bd.MovingColor] >= NUM_ROUNDS_IN_MATERIAL_ADVANTAGE_TO_WIN) {
                return bd.MovingColor;
            }

            const auto nextMove = moves[seed % moves.size];
            Move::MakeMove(nextMove, bd);
            simpleRand(seed);
        }

        return DRAW;
    }
}
