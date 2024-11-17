//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_BISHOPMAP_CUH
#define SRC_BISHOPMAP_CUH

#include "FancyMagicBishopMap.cuh"

__device__ static constexpr FancyMagicBishopMap G_BISHOP_FANCY_MAP_INSTANCE{};

class BishopMap {
    // ---------------------------------------
    // Class creation and initialization
    // ---------------------------------------

public:
    BishopMap() = delete;

    ~BishopMap() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_CALL static constexpr size_t GetBoardIndex(__uint32_t color) {
        return BitBoardsPerCol * color + bishopsIndex;
    }

    [[nodiscard]] FAST_CALL static constexpr __uint64_t
    GetMoves(__uint32_t msbInd, __uint64_t fullBoard, [[maybe_unused]] __uint64_t = 0) {
        return G_BISHOP_FANCY_MAP_INSTANCE.GetMoves(msbInd, fullBoard);
    }

};

#endif //SRC_BISHOPMAP_CUH
