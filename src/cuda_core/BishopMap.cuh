//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_BISHOPMAP_CUH
#define SRC_BISHOPMAP_CUH

#include "FancyMagicBishopMap.cuh"

alignas(128) __device__ static constexpr FancyMagicBishopMap G_BISHOP_FANCY_MAP_INSTANCE{};

class BishopMap final {
    // ---------------------------------------
    // Class creation and initialization
    // ---------------------------------------

public:
    BishopMap() = delete;

    ~BishopMap() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint32_t GetBoardIndex(__uint32_t color) {
        return BIT_BOARDS_PER_COLOR * color + BISHOP_INDEX;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t
    GetMoves(__uint32_t msbInd, __uint64_t fullBoard, [[maybe_unused]] __uint64_t = 0) {
        return G_BISHOP_FANCY_MAP_INSTANCE.GetMoves(msbInd, fullBoard);
    }

};

#endif //SRC_BISHOPMAP_CUH
