//
// Created by Jlisowskyy on 04/12/24.
//

#include "CpuMoveGen.h"

#include "../../engine/include/MoveGeneration/BishopMap.h"
#include "../../engine/include/MoveGeneration/RookMap.h"
#include "../../engine/include/MoveGeneration/MoveGenerator.h"
#include "../../engine/include/BitOperations.h"


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
}
