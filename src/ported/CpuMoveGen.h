//
// Created by Jlisowskyy on 04/12/24.
//

#ifndef SRC_CPUMOVEGEN_H
#define SRC_CPUMOVEGEN_H

#include "CpuDefines.h"

#include <vector>

class Board;

namespace cpu {

    /**
     * @brief Translate an external board representation to an internal board.
     *
     * Converts the external board's bit representations, en passant field,
     * castling rights, and moving color to an internal Board object.
     *
     * @param exBoard External board representation
     * @return Board Translated internal board representation
     */
    [[nodiscard]] Board TranslateToInternalBoard(const external_board& exBoard);

    /**
     * @brief Retrieve rook moves for a given board position using CPU magic bitboard method.
     *
     * @param msbInd Most significant bit index of the piece
     * @param fullMap Full board occupancy map
     * @return uint64_t Bitboard representing all possible rook moves
     */
    [[nodiscard]] uint64_t AccessCpuRookMap(int msbInd, uint64_t fullMap);

    /**
     * @brief Retrieve bishop moves for a given board position using CPU magic bitboard method.
     *
     * @param msbInd Most significant bit index of the piece
     * @param fullMap Full board occupancy map
     * @return uint64_t Bitboard representing all possible bishop moves
     */
    [[nodiscard]] uint64_t AccessCpuBishopMap(int msbInd, uint64_t fullMap);

    /**
     * @brief Extract the most significant bit position from a bitboard.
     *
     * @param map Input bitboard
     * @return int Index of the most significant bit
     */
    [[nodiscard]] int ExtractMsbPosCPU(uint64_t map);

    /**
     * @brief Generate all legal moves for a given external board representation.
     *
     * Translates the external board to an internal board and uses MoveGenerator
     * to produce a vector of moves in external move format.
     *
     * @param board External board representation
     * @return std::vector<external_move> Vector of generated moves
     */
    [[nodiscard]] std::vector<external_move> GenerateMoves(const external_board &board);

    /**
     * @brief Determine if the given board is in check.
     *
     * Translates the external board to an internal board and checks
     * whether the current side to move is in check.
     *
     * @param board External board representation
     * @return bool True if the current side is in check, false otherwise
     */
    [[nodiscard]] bool IsCheck(const external_board &board);

    /**
     * @brief ONLY ONE ENGINE IN A TIME CAN USE THIS FUNCTION
     *
     * @param index
     * @param board
     * @return
     */
    [[nodiscard]] uint32_t SimulateGame(const external_board &board);

    void AllocateStacks(uint32_t count);

    void DeallocStacks(uint32_t count);
}

#endif //SRC_CPUMOVEGEN_H
