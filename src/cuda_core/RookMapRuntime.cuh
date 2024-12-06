#ifndef SRC_ROOKMAPRUNTIME_CUH
#define SRC_ROOKMAPRUNTIME_CUH

#include <cuda_runtime.h>

#include "Helpers.cuh"
#include "cuda_BitOperations.cuh"

class RookMapRuntime final {
public:

    RookMapRuntime() = delete;

    ~RookMapRuntime() = delete;

    [[nodiscard]] __device__ static uint64_t
    GetMoves(uint32_t msbInd, uint64_t fullBoard, [[maybe_unused]] uint64_t = 0) {
        const uint64_t startPos = cuda_MaxMsbPossible >> msbInd;

        const uint32_t row = msbInd / 8;
        const uint32_t col = msbInd % 8;

        const uint32_t startRow = row * 8;
        const uint64_t startRowMask = cuda_MaxMsbPossible >> startRow;

        const uint32_t endRow = startRow + 7;
        const uint64_t endRowMask = cuda_MaxMsbPossible >> endRow;

        uint64_t moves{};

        if ((startPos & startRowMask) == 0) {
            uint64_t mask = startPos;

            do {
                mask <<= 1;
                moves |= mask;
            } while ((mask & startRowMask) == 0 && (mask & fullBoard) == 0);
        }

        if ((startPos & endRowMask) == 0) {
            uint64_t mask = startPos;

            do {
                mask >>= 1;
                moves |= mask;
            } while ((mask & endRowMask) == 0 && (mask & fullBoard) == 0);
        }

        const uint32_t startCol = col;
        const uint64_t startColMask = cuda_MaxMsbPossible >> startCol;

        const uint32_t endCol = col + 56;
        const uint64_t endColMask = cuda_MaxMsbPossible >> endCol;

        if ((startPos & startColMask) == 0) {
            uint64_t mask = startPos;

            do {
                mask <<= 8;
                moves |= mask;
            } while ((mask & startColMask) == 0 && (mask & fullBoard) == 0);
        }

        if ((startPos & endColMask) == 0) {
            uint64_t mask = startPos;

            do {
                mask >>= 8;
                moves |= mask;
            } while ((mask & endColMask) == 0 && (mask & fullBoard) == 0);
        }

        return moves;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr uint32_t GetBoardIndex(int color) {
        return BIT_BOARDS_PER_COLOR * color + ROOK_INDEX;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr uint32_t
    GetMatchingCastlingIndex(const cuda_Board &bd, uint64_t figBoard) {
        for (uint32_t i = 0; i < CASTLINGS_PER_COLOR; ++i)
            if (const uint32_t index = bd.MovingColor * CASTLINGS_PER_COLOR + i;
                    bd.GetCastlingRight(index) && (CASTLING_ROOK_MAPS[index] & figBoard) != 0)
                return index;

        return SENTINEL_CASTLING_INDEX;
    }
};

#endif //SRC_ROOKMAPRUNTIME_CUH
