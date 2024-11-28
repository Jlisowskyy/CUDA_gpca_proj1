//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_ROOKMAP_CUH
#define SRC_ROOKMAP_CUH

#include "FancyMagicRookMap.cuh"

extern __device__ FancyMagicRookMap G_ROOK_FANCY_MAP_INSTANCE;

class RookMap final {
    // ---------------------------------------
    // Class creation and initialization
    // ---------------------------------------
public:

    RookMap() = delete;

    ~RookMap() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr size_t GetBoardIndex(int color) { return BIT_BOARDS_PER_COLOR * color + ROOK_INDEX; }

    [[nodiscard]] FAST_DCALL_ALWAYS static __uint64_t GetMoves(__uint32_t msbInd, __uint64_t fullBoard, [[maybe_unused]] __uint64_t = 0) {
        return G_ROOK_FANCY_MAP_INSTANCE.GetMoves(msbInd, fullBoard);
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint32_t GetMatchingCastlingIndex(const cuda_Board &bd, __uint64_t figBoard) {
        for (__uint32_t i = 0; i < CASTLINGS_PER_COLOR; ++i)
            if (const __uint32_t index = bd.MovingColor * CASTLINGS_PER_COLOR + i;
                    bd.GetCastlingRight(index) && (CASTLING_ROOK_MAPS[index] & figBoard) != 0)
                return index;

        return SENTINEL_CASTLING_INDEX;
    }

};

#endif //SRC_ROOKMAP_CUH
