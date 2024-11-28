//
// Created by Jlisowskyy on 18/11/24.
//

#ifndef SRC_ROOKMAPRUNTIME_CUH
#define SRC_ROOKMAPRUNTIME_CUH

#include <cuda_runtime.h>

#include "Helpers.cuh"
#include "cuda_BitOperations.cuh"

class RookMapRuntime final {
public:

    RookMapRuntime() = delete;

    ~RookMapRuntime() = delete;

    [[nodiscard]] __device__ static __uint64_t
    GetMoves(int msbInd, __uint64_t fullBoard, [[maybe_unused]] __uint64_t = 0) {
        const __uint64_t startPos = cuda_MaxMsbPossible >> msbInd;

        const int row = msbInd / 8;
        const int col = msbInd % 8;

        const int startRow = row * 8;
        const __uint64_t startRowMask = cuda_MaxMsbPossible >> startRow;

        const int endRow = startRow + 7;
        const __uint64_t endRowMask = cuda_MaxMsbPossible >> endRow;

        __uint64_t moves{};

        if ((startPos & startRowMask) == 0) {
            __uint64_t mask = startPos;

            do {
                mask <<= 1;
                moves |= mask;
            } while ((mask & startRowMask) == 0 && (mask & fullBoard) == 0);
        }

        if ((startPos & endRowMask) == 0) {
            __uint64_t mask = startPos;

            do {
                mask >>= 1;
                moves |= mask;
            } while ((mask & endRowMask) == 0 && (mask & fullBoard) == 0);
        }

        const int startCol = col;
        const __uint64_t startColMask = cuda_MaxMsbPossible >> startCol;

        const int endCol = col + 56;
        const __uint64_t endColMask = cuda_MaxMsbPossible >> endCol;

        if ((startPos & startColMask) == 0) {
            __uint64_t mask = startPos;

            do {
                mask <<= 8;
                moves |= mask;
            } while ((mask & startColMask) == 0 && (mask & fullBoard) == 0);
        }

        if ((startPos & endColMask) == 0) {
            __uint64_t mask = startPos;

            do {
                mask >>= 8;
                moves |= mask;
            } while ((mask & endColMask) == 0 && (mask & fullBoard) == 0);
        }

        return moves;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr size_t GetBoardIndex(int color) { return BIT_BOARDS_PER_COLOR * color + ROOK_INDEX; }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint32_t GetMatchingCastlingIndex(const cuda_Board &bd, __uint64_t figBoard) {
        for (__uint32_t i = 0; i < CASTLINGS_PER_COLOR; ++i)
            if (const __uint32_t index = bd.MovingColor * CASTLINGS_PER_COLOR + i;
                    bd.GetCastlingRight(index) && (CASTLING_ROOK_MAPS[index] & figBoard) != 0)
                return index;

        return SENTINEL_CASTLING_INDEX;
    }
};

#endif //SRC_ROOKMAPRUNTIME_CUH
