//
// Created by Jlisowskyy on 04/12/24.
//

#ifndef SRC_CPU_MOVEGEN_CUH
#define SRC_CPU_MOVEGEN_CUH

#include "../cuda_core/cuda_Board.cuh"
#include "../cuda_core/Move.cuh"

#include "../ported/CpuMoveGen.h"

#include <vector>

/**
 * @namespace ported_translation
 * @brief Provides translation layer between Checkmate-Chariot library and this engine
 *
 * Translates move generation and check detection methods from
 * CPU-based implementations to representations used within this project
 */
namespace ported_translation {

    /**
     * @brief Generates legal moves for a given board state
     *
     * Converts CPU-generated moves to CUDA move representation
     *
     * @param board Current chess board state
     * @return std::vector<cuda_Move> List of legal moves
     */
    [[nodiscard]] inline std::vector<cuda_Move> GenMoves(const cuda_Board &board) {
        const auto moves = cpu::GenerateMoves(board.DumpToExternal());

        std::vector<cuda_Move> rv{};
        rv.reserve(moves.size());

        for (const auto &move: moves) {
            rv.emplace_back(move);
        }

        return rv;
    }

    /**
     * @brief Determines if the current board is in check
     *
     * @param board Current chess board state
     * @return bool True if the current side is in check, false otherwise
     */
    [[nodiscard]] inline bool IsCheck(const cuda_Board &board) {
        return cpu::IsCheck(board.DumpToExternal());
    }

}

#endif //SRC_CPU_MOVEGEN_CUH
