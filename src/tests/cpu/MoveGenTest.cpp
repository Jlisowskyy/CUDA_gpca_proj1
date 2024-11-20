//
// Created by Jlisowskyy on 19/11/24.
//

#include "CpuTests.h"

#include "../../../engine/include/MoveGeneration/MoveGenerator.h"

namespace cpu {

    std::vector<external_move> GenerateMoves(const external_board &board) {
        Board bd{};
        for (int i = 0; i < 12; ++i) {
            bd.BitBoards[i] = board[i];
        }
        bd.ElPassantField = board[12];
        bd.Castlings = board[13];
        bd.MovingColor = static_cast<int>(board[14]);

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

    uint64_t CountMoves(const external_board &board, int depth) {
        Board bd{};
        for (int i = 0; i < 12; ++i) {
            bd.BitBoards[i] = board[i];
        }
        bd.ElPassantField = board[12];
        bd.Castlings = board[13];
        bd.MovingColor = static_cast<int>(board[14]);

        Stack<Move, DEFAULT_STACK_SIZE> s;
        MoveGenerator mech{bd, s};

        return mech.CountMoves(bd, depth);
    }

}
