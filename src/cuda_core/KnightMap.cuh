#ifndef KNIGHTMAP_H
#define KNIGHTMAP_H

#include "MoveGenerationUtils.cuh"
#include "Helpers.cuh"
#include "cuda_Array.cuh"

namespace KnightMapConstants {
    __device__ __constant__ static constexpr uint32_t maxMovesCount = 8;

    // Describes knight possible moves coordinates.
    __device__ __constant__ static constexpr int movesCords[] = {6, 15, 17, 10, -6, -15, -17, -10};

    // Accordingly describes y positions after the move relatively to knight's y position.
    // Used to omit errors during generation.
    __device__ __constant__ static constexpr int rowCords[] = {1, 2, 2, 1, -1, -2, -2, -1};

    alignas(128) __device__ static constexpr cuda_Array<uint64_t, BIT_BOARD_FIELDS> movesMap =
            GenStaticMoves(maxMovesCount, movesCords, rowCords);
}

class KnightMap final {
    // ---------------------------------------
    // Class creation and initialization
    // ---------------------------------------

public:
    KnightMap() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr uint32_t GetBoardIndex(int color) {
        return BIT_BOARDS_PER_COLOR * color + KNIGHT_INDEX;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr uint64_t
    GetMoves(uint32_t msbInd, [[maybe_unused]] uint64_t = 0,
             [[maybe_unused]] uint64_t = 0) { return KnightMapConstants::movesMap[msbInd]; }

};


#endif // KNIGHTMAP_H
