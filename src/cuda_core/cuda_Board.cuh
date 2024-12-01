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
    PAWN_INDEX,
    KNIGHT_INDEX,
    BISHOP_INDEX,
    ROOK_INDEX,
    QUEEN_INDEX,
    KING_INDEX,
};

/*
 * Given enum defines indexing order of all (color, piece) BitBoards inside the board.
 * It is important to keep it consistent across the project.
 * Used rather less frequently than previous ones but still defines order of all bitboards.
 * */

enum Descriptors : __uint32_t {
    W_PAWN_INDEX,
    W_KNIGHT_INDEX,
    W_BISHOP_INDEX,
    W_ROOK_INDEX,
    W_QUEEN_INDEX,
    W_KING_INDEX,
    B_PAWN_INDEX,
    B_KNIGHT_INDEX,
    B_BISHOP_INDEX,
    B_ROOK_INDEX,
    B_QUEEN_INDEX,
    B_KING_INDEX,
};

/*
 * Defines the order of castling indexes for given color.
 * */

enum CastlingIndexes : __uint32_t {
    KING_CASTLING_INDEX,
    QUEEN_CASTLING_INDEX,
};

/*
 * Defines indexes of all castling possibilities.
 * */

enum CastlingPossibilities : __uint32_t {
    W_KING_CASTLING_INDEX,
    W_QUEEN_CASTLING_INDEX,
    B_KING_CASTLING_INDEX,
    B_QUEEN_CASTLING_INDEX,
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

__device__ __constant__ static constexpr __uint32_t BIT_BOARDS_COUNT = 12;
__device__ __constant__ static constexpr __uint32_t BIT_BOARDS_GUARDED_COUNT = BIT_BOARDS_COUNT + 1;
__device__ __constant__ static constexpr __uint32_t CASTLING_COUNT = 4;
__device__ __constant__ static constexpr __uint32_t BIT_BOARD_FIELDS = 64;
__device__ __constant__ static constexpr __uint32_t BIT_BOARDS_PER_COLOR = 6;
__device__ __constant__ static constexpr __uint32_t KING_POS_COUNT = 2;
__device__ __constant__ static constexpr __uint32_t CASTLINGS_PER_COLOR = 2;
__device__ __constant__ static constexpr __uint64_t INVALID_EL_PASSANT_FIELD = 1;
__device__ __constant__ static constexpr __uint64_t INVALID_EL_PASSANT_BIT_BOARD = cuda_MaxMsbPossible >> INVALID_EL_PASSANT_FIELD;
__device__ __constant__ static constexpr __uint32_t SENTINEL_BOARD_INDEX = 12;
__device__ __constant__ static constexpr __uint32_t SENTINEL_CASTLING_INDEX = 4;

__device__ __constant__ static constexpr __uint64_t DEFAULT_KING_BOARDS[KING_POS_COUNT]{
        cuda_MaxMsbPossible >> ConvertToReversedPos(4), cuda_MaxMsbPossible >> ConvertToReversedPos(60)
};

__device__ __constant__ static constexpr __int32_t CASTLING_NEW_KING_POS[CASTLING_COUNT]{
        ConvertToReversedPos(6), ConvertToReversedPos(2), ConvertToReversedPos(62), ConvertToReversedPos(58)
};

__device__ __constant__ static constexpr __uint64_t CASTLING_ROOK_MAPS[CASTLING_COUNT]{
        cuda_MinMsbPossible << 7, cuda_MinMsbPossible, cuda_MinMsbPossible << 63, cuda_MinMsbPossible << 56
};

__device__ __constant__ static constexpr __uint64_t CASTLING_NEW_ROOK_MAPS[CASTLING_COUNT]{
        cuda_MinMsbPossible << 5, cuda_MinMsbPossible << 3, cuda_MinMsbPossible << 61, cuda_MinMsbPossible << 59
};

__device__ __constant__ static constexpr __uint64_t CASTLING_SENSITIVE_FIELDS[CASTLING_COUNT]{
        cuda_MinMsbPossible << 6 | cuda_MinMsbPossible << 5, cuda_MinMsbPossible << 2 | cuda_MinMsbPossible << 3,
        cuda_MinMsbPossible << 61 | cuda_MinMsbPossible << 62, cuda_MinMsbPossible << 58 | cuda_MinMsbPossible << 59
};

__device__ __constant__ static constexpr __uint64_t CASTLING_TOUCHED_FIELDS[CASTLING_COUNT]{
        cuda_MinMsbPossible << 6 | cuda_MinMsbPossible << 5,
        cuda_MinMsbPossible << 2 | cuda_MinMsbPossible << 3 | cuda_MinMsbPossible << 1,
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
        for (__uint32_t i = 0; i < 12; ++i) {
            BitBoards[i] = board[i];
        }

        ElPassantField = board[12];
        Castlings = board[13];
        MovingColor = board[14];

        assert(MovingColor == WHITE || MovingColor == BLACK);
        assert(Castlings <= (1 << (CASTLING_COUNT + 1)));
    }

    // ------------------------------
    // class interaction
    // ------------------------------

    FAST_DCALL_ALWAYS void ChangePlayingColor() { MovingColor ^= 1; }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint32_t GetKingMsbPos(const __uint32_t col) const {
        return ExtractMsbPos(BitBoards[col * BIT_BOARDS_PER_COLOR + KING_INDEX]);
    }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint64_t GetFigBoard(__uint32_t col, __uint32_t figDesc) const {
        return BitBoards[col * BIT_BOARDS_PER_COLOR + figDesc];
    }

    FAST_DCALL_ALWAYS void SetCastlingRight(__uint32_t castlingIndex, bool value) {
        SetBitBoardBit(Castlings, castlingIndex, value);
    }

    [[nodiscard]] FAST_CALL_ALWAYS bool GetCastlingRight(__uint32_t castlingIndex) const {
        return Castlings & (cuda_MinMsbPossible << castlingIndex);
    }

    // --------------------------------
    // Main processing components
    // --------------------------------

    __uint64_t BitBoards[BIT_BOARDS_GUARDED_COUNT]; // additional sentinel board
    __uint64_t ElPassantField;
    __uint32_t Castlings; // additional sentinel field
    __uint32_t MovingColor;
};

#endif // CUDA_BOARD_CUH
