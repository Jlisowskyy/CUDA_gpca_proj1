//
// Created by Jlisowskyy on 3/4/24.
//

#ifndef CUDA_BOARD_CUH
#define CUDA_BOARD_CUH

#include "cuda_BitOperations.cuh"
#include "Helpers.cuh"

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

class alignas(16) cuda_Board {
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

    __device__ void ChangePlayingColor() { MovingColor ^= 1; }

    __device__ int GetKingMsbPos(const int col) const {
        return ExtractMsbPos(BitBoards[col * BitBoardsPerCol + kingIndex]);
    }

    __device__ uint64_t GetFigBoard(int col, uint32_t figDesc) const {
        return BitBoards[col * BitBoardsPerCol + figDesc];
    }

    __device__ void SetCastlingRight(size_t castlingIndex, bool value) {
        Castlings = (value << castlingIndex) & (Castlings & ~(MinMsbPossible << castlingIndex));
    }

    __device__ bool GetCastlingRight(size_t castlingIndex) const {
        return Castlings & (MinMsbPossible << castlingIndex);
    }

    // ------------------------------
    // Class fields
    // ------------------------------

    static constexpr uint32_t BitBoardsCount = 12;
    static constexpr uint32_t CastlingCount = 4;
    static constexpr uint32_t BitBoardFields = 64;
    static constexpr uint32_t BitBoardsPerCol = 6;
    static constexpr uint32_t KingPosCount = 2;
    static constexpr uint32_t CastlingsPerColor = 2;
    static constexpr uint64_t InvalidElPassantField = 1;
    static constexpr uint64_t InvalidElPassantBitBoard = MaxMsbPossible >> InvalidElPassantField;
    static constexpr uint32_t SentinelBoardIndex = 12;
    static constexpr uint32_t SentinelCastlingIndex = 4;

    static inline uint64_t DefaultKingBoards[KingPosCount]{
            MaxMsbPossible >> ConvertToReversedPos(4), MaxMsbPossible >> ConvertToReversedPos(60)
    };
    static inline int32_t CastlingNewKingPos[CastlingCount]{
            ConvertToReversedPos(6), ConvertToReversedPos(2), ConvertToReversedPos(62), ConvertToReversedPos(58)
    };

    static constexpr uint64_t CastlingsRookMaps[CastlingCount]{
            MinMsbPossible << 7, MinMsbPossible, MinMsbPossible << 63, MinMsbPossible << 56
    };

    static constexpr uint64_t CastlingNewRookMaps[CastlingCount]{
            MinMsbPossible << 5, MinMsbPossible << 3, MinMsbPossible << 61, MinMsbPossible << 59
    };

    static constexpr uint64_t CastlingSensitiveFields[CastlingCount]{
            MinMsbPossible << 6 | MinMsbPossible << 5, MinMsbPossible << 2 | MinMsbPossible << 3,
            MinMsbPossible << 61 | MinMsbPossible << 62, MinMsbPossible << 58 | MinMsbPossible << 59
    };

    static constexpr uint64_t CastlingTouchedFields[CastlingCount]{
            MinMsbPossible << 6 | MinMsbPossible << 5, MinMsbPossible << 2 | MinMsbPossible << 3 | MinMsbPossible << 1,
            MinMsbPossible << 61 | MinMsbPossible << 62,
            MinMsbPossible << 58 | MinMsbPossible << 59 | MinMsbPossible << 57
    };

    // --------------------------------
    // Main processing components
    // --------------------------------

    uint64_t BitBoards[BitBoardsCount + 1]{}; // additional sentinel board
    uint64_t ElPassantField{MaxMsbPossible >> InvalidElPassantField};
    uint32_t Castlings{0}; // additional sentinel field
    uint32_t MovingColor{WHITE};
};

#endif // CUDA_BOARD_CUH
