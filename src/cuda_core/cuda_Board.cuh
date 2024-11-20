//
// Created by Jlisowskyy on 3/4/24.
//

#ifndef CUDA_BOARD_CUH
#define CUDA_BOARD_CUH

#include <cassert>

#include "cuda_BitOperations.cuh"
#include "Helpers.cuh"

#include "../ported/CpuDefines.h"

/*
 * Given enum defines values and order of both colors. All indexing schemes used across the projects follows given
 * order. It is important to keep it consistent across the project. It is also very useful when performing some indexing
 * and color switching operations.
 * */

enum Color : int {
    WHITE,
    BLACK,
};

/*
 * Given enum defines indexing order of all BitBoards inside the board.
 * It is important to keep this order consistent across the project.
 * Again very useful when performing some indexing and color switching operations.
 * */

enum ColorlessDescriptors : __uint32_t {
    pawnsIndex,
    knightsIndex,
    bishopsIndex,
    rooksIndex,
    queensIndex,
    kingIndex,
};

/*
 * Given enum defines indexing order of all (color, piece) BitBoards inside the board.
 * It is important to keep it consistent across the project.
 * Used rather less frequently than previous ones but still defines order of all bitboards.
 * */

enum Descriptors : __uint32_t {
    wPawnsIndex,
    wKnightsIndex,
    wBishopsIndex,
    wRooksIndex,
    wQueensIndex,
    wKingIndex,
    bPawnsIndex,
    bKnightsIndex,
    bBishopsIndex,
    bRooksIndex,
    bQueensIndex,
    bKingIndex,
};

/*
 * Defines the order of castling indexes for given color.
 * */

enum CastlingIndexes : __uint32_t {
    KingCastlingIndex,
    QueenCastlingIndex,
};

/*
 * Defines indexes of all castling possibilities.
 * */

enum CastlingPossibilities : __uint32_t {
    WhiteKingSide,
    WhiteQueenSide,
    BlackKingSide,
    BlackQueenSide,
};

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

__device__ static constexpr __uint32_t BitBoardsCount = 12;
__device__ static constexpr __uint32_t CastlingCount = 4;
__device__ static constexpr __uint32_t BitBoardFields = 64;
__device__ static constexpr __uint32_t BitBoardsPerCol = 6;
__device__ static constexpr __uint32_t KingPosCount = 2;
__device__ static constexpr __uint32_t CastlingsPerColor = 2;
__device__ static constexpr __uint64_t InvalidElPassantField = 1;
__device__ static constexpr __uint64_t InvalidElPassantBitBoard = cuda_MaxMsbPossible >> InvalidElPassantField;
__device__ static constexpr __uint32_t SentinelBoardIndex = 12;
__device__ static constexpr __uint32_t SentinelCastlingIndex = 4;

__device__ static constexpr __uint64_t DefaultKingBoards[KingPosCount]{
        cuda_MaxMsbPossible >> ConvertToReversedPos(4), cuda_MaxMsbPossible >> ConvertToReversedPos(60)
};

__device__ static constexpr __int32_t CastlingNewKingPos[CastlingCount]{
        ConvertToReversedPos(6), ConvertToReversedPos(2), ConvertToReversedPos(62), ConvertToReversedPos(58)
};

__device__ static constexpr __uint64_t CastlingsRookMaps[CastlingCount]{
        cuda_MinMsbPossible << 7, cuda_MinMsbPossible, cuda_MinMsbPossible << 63, cuda_MinMsbPossible << 56
};

__device__ static constexpr __uint64_t CastlingNewRookMaps[CastlingCount]{
        cuda_MinMsbPossible << 5, cuda_MinMsbPossible << 3, cuda_MinMsbPossible << 61, cuda_MinMsbPossible << 59
};

__device__ static constexpr __uint64_t CastlingSensitiveFields[CastlingCount]{
        cuda_MinMsbPossible << 6 | cuda_MinMsbPossible << 5, cuda_MinMsbPossible << 2 | cuda_MinMsbPossible << 3,
        cuda_MinMsbPossible << 61 | cuda_MinMsbPossible << 62, cuda_MinMsbPossible << 58 | cuda_MinMsbPossible << 59
};

__device__ static constexpr __uint64_t CastlingTouchedFields[CastlingCount]{
        cuda_MinMsbPossible << 6 | cuda_MinMsbPossible << 5, cuda_MinMsbPossible << 2 | cuda_MinMsbPossible << 3 | cuda_MinMsbPossible << 1,
        cuda_MinMsbPossible << 61 | cuda_MinMsbPossible << 62,
        cuda_MinMsbPossible << 58 | cuda_MinMsbPossible << 59 | cuda_MinMsbPossible << 57
};

class cuda_Board final {
public:
    // ------------------------------
    // Class creation
    // ------------------------------

    cuda_Board() = default;

    ~cuda_Board() = default;

    cuda_Board(const cuda_Board &) = default;

    cuda_Board &operator=(const cuda_Board &) = default;

    explicit cuda_Board(const cpu::external_board &board) {
        for (size_t i = 0; i < 12; ++i)
            BitBoards[i] = board[i];

        ElPassantField = board[12];
        Castlings = board[13];
        MovingColor = board[14];

        assert(MovingColor == WHITE || MovingColor == BLACK);
        assert(Castlings <= (1 << (CastlingCount + 1)));
    }

    // ------------------------------
    // class interaction
    // ------------------------------

    FAST_CALL void ChangePlayingColor() { MovingColor ^= 1; }

    [[nodiscard]] __device__ INLINE __uint32_t GetKingMsbPos(const __uint32_t col) const {
        return ExtractMsbPos(BitBoards[col * BitBoardsPerCol + kingIndex]);
    }

    [[nodiscard]] FAST_CALL __uint64_t GetFigBoard(__uint32_t col, __uint32_t figDesc) const {
        return BitBoards[col * BitBoardsPerCol + figDesc];
    }

    FAST_CALL void SetCastlingRight(size_t castlingIndex, bool value) {
        SetBitBoardBit(Castlings, castlingIndex, value);
    }

    [[nodiscard]] FAST_CALL bool GetCastlingRight(size_t castlingIndex) const {
        return Castlings & (cuda_MinMsbPossible << castlingIndex);
    }

    // --------------------------------
    // Main processing components
    // --------------------------------

    __uint64_t BitBoards[BitBoardsCount + 1]{}; // additional sentinel board
    __uint64_t ElPassantField{cuda_MaxMsbPossible >> InvalidElPassantField};
    __uint32_t Castlings{0}; // additional sentinel field
    __uint32_t MovingColor{WHITE};
};

#endif // CUDA_BOARD_CUH
