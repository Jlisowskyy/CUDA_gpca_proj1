//
// Created by Jlisowskyy on 19/11/24.
//

#ifndef SRC_CPUUTILS_H
#define SRC_CPUUTILS_H

#include "CpuDefines.h"

#include <string>

namespace cpu {
    /**
     * @brief Translate a FEN string to an external board representation.
     *
     * Converts a FEN string to an internal Board, then transfers
     * bit representations and game state to an external board.
     *
     * @param fen FEN string representing the chess board state
     * @return external_board Converted board representation
     */
    [[nodiscard]] external_board TranslateFromFen(const std::string &fen);

    /**
     * @brief Convert an external board representation to a FEN string.
     *
     * Translates an external board to an internal Board, then generates its FEN representation.
     *
     * @param board External board to convert
     * @return std::string FEN string representing the board state
     */
    [[nodiscard]] std::string TranslateToFen(const external_board& board);

    /**
     * @brief Remove leading and trailing whitespace from a string.
     *
     * Uses ParseTools from Checkmate-Chariot to trim whitespace from the input string.
     *
     * @param str Reference to string to be trimmed (modified in-place)
     */
    void Trim(std::string& str);

    /**
     * @brief Retrieve the default starting chess board position.
     *
     * Creates an external board representing the standard initial chess board setup.
     *
     * @return external_board Default chess board representation
     */
    [[nodiscard]] external_board GetDefaultBoard();

    /**
     * @brief Attempt to translate a FEN string to an external board.
     *
     * Tries to convert a FEN string to an internal Board, then to an external board.
     *
     * @param fen FEN string to translate
     * @param board Reference to external board to be populated
     * @return bool True if translation is successful, false otherwise
     */
    [[nodiscard]] bool Translate(const std::string &fen, external_board &board);

    /**
     * @brief Display the contents of an external board representation.
     *
     * Translates the external board to an internal board and displays its state.
     *
     * @param board External board to display
     */
    void DisplayBoard(const external_board& board);
}

#endif //SRC_CPUUTILS_H
