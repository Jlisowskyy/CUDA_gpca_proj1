//
// Created by Jlisowskyy on 18/11/24.
//

#ifndef SRC_BISHOPMAPRUNTIME_CUH
#define SRC_BISHOPMAPRUNTIME_CUH

#include <cuda_runtime.h>

#include "Helpers.cuh"

#include "cuda_BitOperations.cuh"

class BishopMapRuntime final {
public:

    BishopMapRuntime() = delete;

    ~BishopMapRuntime() = delete;

    [[nodiscard]] FAST_CALL static __uint64_t
    GetMoves(int msbInd, __uint64_t fullBoard, [[maybe_unused]] __uint64_t = 0) {
        const __uint64_t startPos = cuda_MaxMsbPossible >> msbInd;
        const int startRow = msbInd / 8;
        const int startCol = msbInd % 8;

        __uint64_t moves{};

        if (startRow < 7 && startCol < 7) {
            int row = startRow;
            int col = startCol;
            __uint64_t mask = startPos;

            do {
                ++row;
                ++col;

                mask >>= 9;
                moves |= mask;
            } while (row < 7 && col < 7 && (mask & fullBoard) == 0);
        }

        if (startRow > 0 && startCol > 0) {
            int row = startRow;
            int col = startCol;
            __uint64_t mask = startPos;

            do {
                --row;
                --col;

                mask <<= 9;
                moves |= mask;
            } while (row > 0 && col > 0 && (mask & fullBoard) == 0);
        }

        if (startRow < 7 && startCol > 0) {
            int row = startRow;
            int col = startCol;
            __uint64_t mask = startPos;

            do {
                ++row;
                --col;

                mask >>= 7;
                moves |= mask;
            } while (row < 7 && col > 0 && (mask & fullBoard) == 0);
        }

        if (startRow > 0 && startCol < 7) {
            int row = startRow;
            int col = startCol;
            __uint64_t mask = startPos;

            do {
                --row;
                ++col;

                mask <<= 7;
                moves |= mask;
            } while (row > 0 && col < 7 && (mask & fullBoard) == 0);
        }

        return moves;
    }
};

#endif //SRC_BISHOPMAPRUNTIME_CUH