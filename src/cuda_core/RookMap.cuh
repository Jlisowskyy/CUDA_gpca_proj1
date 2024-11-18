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

    [[nodiscard]] FAST_CALL static constexpr size_t GetBoardIndex(int color) { return BitBoardsPerCol * color + rooksIndex; }

    [[nodiscard]] FAST_CALL static __uint64_t GetMoves(__uint32_t msbInd, __uint64_t fullBoard, [[maybe_unused]] __uint64_t = 0) {
        return G_ROOK_FANCY_MAP_INSTANCE.GetMoves(msbInd, fullBoard);
    }

    [[nodiscard]] __device__ INLINE static constexpr size_t GetMatchingCastlingIndex(const cuda_Board &bd, __uint64_t figBoard) {
        for (size_t i = 0; i < CastlingsPerColor; ++i)
            if (const size_t index = bd.MovingColor * CastlingsPerColor + i;
                    bd.GetCastlingRight(index) && (CastlingsRookMaps[index] & figBoard) != 0)
                return index;

        return SentinelCastlingIndex;
    }

};

#endif //SRC_ROOKMAP_CUH
