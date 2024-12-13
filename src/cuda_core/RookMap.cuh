#ifndef SRC_ROOKMAP_CUH
#define SRC_ROOKMAP_CUH

#include "FancyMagicRookMap.cuh"
#include "cuda_PackedBoard.cuh"

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

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr uint32_t GetBoardIndex(const uint32_t color) {
        return BIT_BOARDS_PER_COLOR * color + ROOK_INDEX;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static uint64_t
    GetMoves(const uint32_t msbInd, const uint64_t fullBoard, [[maybe_unused]] uint64_t = 0) {
        return G_ROOK_FANCY_MAP_INSTANCE.GetMoves(msbInd, fullBoard);
    }

    template<uint32_t NUM_BOARDS>
    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr uint32_t
    GetMatchingCastlingIndex(const typename cuda_PackedBoard<NUM_BOARDS>::BoardFetcher &fetcher,
                             const uint64_t figBoard) {
        uint32_t rv{};

        uint32_t index = fetcher.MovingColor() * CASTLINGS_PER_COLOR;
        rv += (1 + index) * (fetcher.GetCastlingRight(index) && ((CASTLING_ROOK_MAPS[index] & figBoard) != 0));

        index += 1;
        rv += (1 + index) * (fetcher.GetCastlingRight(index) && ((CASTLING_ROOK_MAPS[index] & figBoard) != 0));

        return rv == 0 ? SENTINEL_CASTLING_INDEX : rv - 1;
    }
};

#endif //SRC_ROOKMAP_CUH
