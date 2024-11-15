//
// Created by Jlisowskyy on 3/4/24.
//

#ifndef BOARD_HPP
#define BOARD_HPP

#include "../../../../../../../usr/lib/gcc/x86_64-pc-linux-gnu/13.3.0/include/c++/array"
#include "../../../../../../../usr/lib/gcc/x86_64-pc-linux-gnu/13.3.0/include/c++/bitset"
#include "../../../../../../../usr/lib/gcc/x86_64-pc-linux-gnu/13.3.0/include/c++/cinttypes"
#include "../../../../../../../usr/lib/gcc/x86_64-pc-linux-gnu/13.3.0/include/c++/unordered_map"

#include "../utilities/BitOperations.hpp"
#include "../utilities/CompilationConstants.hpp"

/*
 *  The most important class used around the project.
 *  It defines representation of the board state.
 *  Currently, it consists of:
 *      - BitBoards: 12 bitboards representing all pieces of both colors with one additional sentinel board at the end.
 *        Such representation allows to easily iterate over all pieces of given color and perform operation with very
 * fast bit operations. Additionally, sentinel allows to unconditionally treats all move types without any additional
 * checks.
 *      - Single ElPassantField: 64-bit integer representing field where en passant is possible.
 *      - Single MovingColor: integer representing color of the player who is currently moving.
 *      - Castlings: bitset representing all castling possibilities for both colors with one additional sentinel field
 * at the end.
 *
 * */

struct cpu_Board {
    // ------------------------------
    // Class creation
    // ------------------------------

    cpu_Board() = default;

    ~cpu_Board() = default;

    cpu_Board(const cpu_Board &) = default;

    cpu_Board &operator=(const cpu_Board &) = default;

    // ------------------------------
    // class interaction
    // ------------------------------

    void ChangePlayingColor() { MovingColor ^= 1; }

    [[nodiscard]] constexpr int GetKingMsbPos(const int col) const {
        return ExtractMsbPos(BitBoards[col * BitBoardsPerCol + kingIndex]);
    }

    [[nodiscard]] constexpr uint64_t GetFigBoard(int col, size_t figDesc) const {
        return BitBoards[col * BitBoardsPerCol + figDesc];
    }

    void SetCastlingRight(size_t castlingIndex, bool value) {
        Castlings = (value << castlingIndex) & (Castlings & ~(MinMsbPossible << castlingIndex));
    }

    [[nodiscard]] bool GetCastlingRight(size_t castlingIndex) const {
        return Castlings & (MinMsbPossible << castlingIndex);
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    static constexpr size_t BitBoardsCount = 12;
    static constexpr size_t CastlingCount = 4;
    static constexpr size_t BitBoardFields = 64;
    static constexpr size_t BitBoardsPerCol = 6;
    static constexpr size_t KingPosCount = 2;
    static constexpr size_t CastlingsPerColor = 2;
    static constexpr uint64_t InvalidElPassantField = 1;
    static constexpr uint64_t InvalidElPassantBitBoard = MaxMsbPossible >> InvalidElPassantField;
    static constexpr size_t SentinelBoardIndex = 12;
    static constexpr size_t SentinelCastlingIndex = 4;

    static constexpr std::array<uint64_t, KingPosCount> DefaultKingBoards{
            MaxMsbPossible >> ConvertToReversedPos(4), MaxMsbPossible >> ConvertToReversedPos(60)
    };
    static constexpr std::array<int, CastlingCount> CastlingNewKingPos{
            ConvertToReversedPos(6), ConvertToReversedPos(2), ConvertToReversedPos(62), ConvertToReversedPos(58)
    };

    static constexpr std::array<uint64_t, CastlingCount> CastlingsRookMaps{
            MinMsbPossible << 7, MinMsbPossible, MinMsbPossible << 63, MinMsbPossible << 56
    };

    static constexpr std::array<uint64_t, CastlingCount> CastlingNewRookMaps{
            MinMsbPossible << 5, MinMsbPossible << 3, MinMsbPossible << 61, MinMsbPossible << 59
    };

    static constexpr std::array<uint64_t, CastlingCount> CastlingSensitiveFields{
            MinMsbPossible << 6 | MinMsbPossible << 5, MinMsbPossible << 2 | MinMsbPossible << 3,
            MinMsbPossible << 61 | MinMsbPossible << 62, MinMsbPossible << 58 | MinMsbPossible << 59
    };

    static constexpr std::array<uint64_t, CastlingCount> CastlingTouchedFields{
            MinMsbPossible << 6 | MinMsbPossible << 5,
            MinMsbPossible << 2 | MinMsbPossible << 3 | MinMsbPossible << 1,
            MinMsbPossible << 61 | MinMsbPossible << 62,
            MinMsbPossible << 58 | MinMsbPossible << 59 | MinMsbPossible << 57
    };

    // --------------------------------
    // Main processing components
    // --------------------------------

    uint64_t BitBoards[BitBoardsCount + 1]{}; // additional sentinel board
    uint64_t ElPassantField{MaxMsbPossible >> InvalidElPassantField};
    uint32_t Castlings{0}; // additional sentinel field
    uint32_t MovingColor{WHITE};// additional sentinel board
};

#endif // BOARD_HPP
