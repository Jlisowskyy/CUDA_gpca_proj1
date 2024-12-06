#ifndef CUDA_BOARD_CUH
#define CUDA_BOARD_CUH

#include <cassert>
#include <bit>

#include "cuda_BitOperations.cuh"
#include "Helpers.cuh"

#include "../ported/CpuDefines.h"

enum Color : uint32_t {
    WHITE,
    BLACK,
};

enum EVAL_RESULTS : uint32_t {
    WHITE_WIN = WHITE,
    BLACK_WIN = BLACK,
    DRAW = 2,
};
static constexpr uint32_t NUM_EVAL_RESULTS = 3;

static_assert((uint32_t) DRAW != (uint32_t) WHITE && (uint32_t) DRAW != (uint32_t) BLACK);

enum ColorlessDescriptors : uint32_t {
    PAWN_INDEX,
    KNIGHT_INDEX,
    BISHOP_INDEX,
    ROOK_INDEX,
    QUEEN_INDEX,
    KING_INDEX,
};

enum Descriptors : uint32_t {
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

enum CastlingIndexes : uint32_t {
    KING_CASTLING_INDEX,
    QUEEN_CASTLING_INDEX,
};

enum CastlingPossibilities : uint32_t {
    W_KING_CASTLING_INDEX,
    W_QUEEN_CASTLING_INDEX,
    B_KING_CASTLING_INDEX,
    B_QUEEN_CASTLING_INDEX,
};

__device__ __constant__ static constexpr uint32_t BIT_BOARDS_COUNT = 12;
__device__ __constant__ static constexpr uint32_t BIT_BOARDS_GUARDED_COUNT = BIT_BOARDS_COUNT + 1;
__device__ __constant__ static constexpr uint32_t CASTLING_COUNT = 4;
__device__ __constant__ static constexpr uint32_t BIT_BOARD_FIELDS = 64;
__device__ __constant__ static constexpr uint32_t BIT_BOARDS_PER_COLOR = 6;
__device__ __constant__ static constexpr uint32_t KING_POS_COUNT = 2;
__device__ __constant__ static constexpr uint32_t CASTLINGS_PER_COLOR = 2;
__device__ __constant__ static constexpr uint64_t INVALID_EL_PASSANT_FIELD = 1;
__device__ __constant__ static constexpr uint64_t INVALID_EL_PASSANT_BIT_BOARD = cuda_MaxMsbPossible >> INVALID_EL_PASSANT_FIELD;
__device__ __constant__ static constexpr uint32_t SENTINEL_BOARD_INDEX = 12;
__device__ __constant__ static constexpr uint32_t SENTINEL_CASTLING_INDEX = 4;

__device__ __constant__ static constexpr uint64_t DEFAULT_KING_BOARDS[KING_POS_COUNT]{
        cuda_MaxMsbPossible >> ConvertToReversedPos(4), cuda_MaxMsbPossible >> ConvertToReversedPos(60)
};

__device__ __constant__ static constexpr int32_t CASTLING_NEW_KING_POS[CASTLING_COUNT]{
        ConvertToReversedPos(6), ConvertToReversedPos(2), ConvertToReversedPos(62), ConvertToReversedPos(58)
};

__device__ __constant__ static constexpr uint64_t CASTLING_ROOK_MAPS[CASTLING_COUNT]{
        cuda_MinMsbPossible << 7, cuda_MinMsbPossible, cuda_MinMsbPossible << 63, cuda_MinMsbPossible << 56
};

__device__ __constant__ static constexpr uint64_t CASTLING_NEW_ROOK_MAPS[CASTLING_COUNT]{
        cuda_MinMsbPossible << 5, cuda_MinMsbPossible << 3, cuda_MinMsbPossible << 61, cuda_MinMsbPossible << 59
};

__device__ __constant__ static constexpr uint64_t CASTLING_SENSITIVE_FIELDS[CASTLING_COUNT]{
        cuda_MinMsbPossible << 6 | cuda_MinMsbPossible << 5, cuda_MinMsbPossible << 2 | cuda_MinMsbPossible << 3,
        cuda_MinMsbPossible << 61 | cuda_MinMsbPossible << 62, cuda_MinMsbPossible << 58 | cuda_MinMsbPossible << 59
};

__device__ __constant__ static constexpr uint64_t CASTLING_TOUCHED_FIELDS[CASTLING_COUNT]{
        cuda_MinMsbPossible << 6 | cuda_MinMsbPossible << 5,
        cuda_MinMsbPossible << 2 | cuda_MinMsbPossible << 3 | cuda_MinMsbPossible << 1,
        cuda_MinMsbPossible << 61 | cuda_MinMsbPossible << 62,
        cuda_MinMsbPossible << 58 | cuda_MinMsbPossible << 59 | cuda_MinMsbPossible << 57
};

__device__ __constant__ static constexpr int32_t FIG_VALUES[BIT_BOARDS_GUARDED_COUNT]{
        100, 330, 330, 500, 900, 10000, -100, -330, -330, -500, -900, -10000, 0
};

static constexpr int32_t FIG_VALUES_CPU[BIT_BOARDS_GUARDED_COUNT]{
        100, 330, 330, 500, 900, 10000, -100, -330, -330, -500, -900, -10000, 0
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
        for (uint32_t i = 0; i < 12; ++i) {
            BitBoards[i] = board[i];
        }

        ElPassantField = board[12];
        Castlings = board[13];
        MovingColor = board[14];
        MaterialEval = EvaluateMaterial();

        assert(MovingColor == WHITE || MovingColor == BLACK);
        assert(Castlings <= (1 << (CASTLING_COUNT + 1)));
    }

    // ------------------------------
    // class interaction
    // ------------------------------

    [[nodiscard]] cpu::external_board DumpToExternal() const {
        cpu::external_board rv{};

        for (uint32_t i = 0; i < 12; ++i) {
            rv[i] = BitBoards[i];
        }

        rv[12] = ElPassantField;
        rv[13] = Castlings;
        rv[14] = MovingColor;

        return rv;
    }

    FAST_CALL_ALWAYS void ChangePlayingColor() { MovingColor ^= 1; }

    [[nodiscard]] FAST_DCALL_ALWAYS uint32_t GetKingMsbPos(const uint32_t col) const {
        return ExtractMsbPos(BitBoards[col * BIT_BOARDS_PER_COLOR + KING_INDEX]);
    }

    [[nodiscard]] FAST_DCALL_ALWAYS uint64_t GetFigBoard(uint32_t col, uint32_t figDesc) const {
        return BitBoards[col * BIT_BOARDS_PER_COLOR + figDesc];
    }

    FAST_DCALL_ALWAYS void SetCastlingRight(uint32_t castlingIndex, bool value) {
        SetBitBoardBit(Castlings, castlingIndex, value);
    }

    [[nodiscard]] FAST_CALL_ALWAYS bool GetCastlingRight(uint32_t castlingIndex) const {
        return Castlings & (cuda_MinMsbPossible << castlingIndex);
    }

    [[nodiscard]] int32_t EvaluateMaterial() const {
        int32_t eval{};

        for (uint32_t bIdx = 0; bIdx < BIT_BOARDS_GUARDED_COUNT; ++bIdx) {
            eval += std::popcount(BitBoards[bIdx]) * FIG_VALUES_CPU[bIdx];
        }

        return eval;
    }

    // --------------------------------
    // Main processing components
    // --------------------------------

    uint64_t BitBoards[BIT_BOARDS_GUARDED_COUNT]; // additional sentinel board
    uint64_t ElPassantField;
    uint32_t Castlings;
    uint32_t MovingColor;
    int32_t MaterialEval;
};

#endif // CUDA_BOARD_CUH
