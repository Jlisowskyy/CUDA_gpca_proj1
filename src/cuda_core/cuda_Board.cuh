//
// Created by Jlisowskyy on 3/4/24.
//

#ifndef CUDA_BOARD_CUH
#define CUDA_BOARD_CUH

#include "cuda_BitOperations.cuh"
#include "Helpers.cuh"
#include "../utilities/BoardDefs.hpp"

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

    // ------------------------------
    // class interaction
    // ------------------------------

    FAST_CALL void ChangePlayingColor() { MovingColor ^= 1; }

    [[nodiscard]] __device__ INLINE __uint32_t GetKingMsbPos(const __uint32_t col) const {
        return ExtractMsbPos(BitBoards[col * BitBoardsPerCol + kingIndex]);
    }

    [[nodiscard]] FAST_CALL __uint64_t GetFigBoard(int col, __uint32_t figDesc) const {
        return BitBoards[col * BitBoardsPerCol + figDesc];
    }

    FAST_CALL void SetCastlingRight(size_t castlingIndex, bool value) {
        Castlings = (value << castlingIndex) & (Castlings & ~(cuda_MinMsbPossible << castlingIndex));
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
