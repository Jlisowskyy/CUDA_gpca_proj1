//
// Created by Jlisowskyy on 04/12/24.
//

#ifndef SRC_CPU_MOVEGEN_CUH
#define SRC_CPU_MOVEGEN_CUH

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/Move.cuh"

#include "../ported/CpuMoveGen.h"

#include <vector>

namespace ported_translation {

    [[nodiscard]] std::vector<cuda_Move> GenMoves(const cuda_Board &board) {
        const auto moves = cpu::GenerateMoves(board.DumpToExternal());

        std::vector<cuda_Move> rv{};
        rv.reserve(moves.size());

        for (const auto &move: moves) {
            rv.emplace_back(move);
        }

        return rv;
    }

    [[nodiscard]] bool IsCheck(const cuda_Board &board) {
        return cpu::IsCheck(board.DumpToExternal());
    }

}

#endif //SRC_CPU_MOVEGEN_CUH
