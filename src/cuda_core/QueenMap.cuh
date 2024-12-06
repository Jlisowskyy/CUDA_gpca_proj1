#ifndef QUEENMAP_H
#define QUEENMAP_H

#include <cstdint>

#include "BishopMap.cuh"
#include "RookMap.cuh"

class QueenMap final {
    // ------------------------------
    // Class creation
    // ------------------------------

public:
    constexpr QueenMap() = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr uint32_t GetBoardIndex(uint32_t color) {
        return BIT_BOARDS_PER_COLOR * color + QUEEN_INDEX;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static uint64_t
    GetMoves(uint32_t msbInd, uint64_t fullMap, [[maybe_unused]] uint64_t = 0) {
        return BishopMap::GetMoves(msbInd, fullMap) | RookMap::GetMoves(msbInd, fullMap);
    }
};

#endif // QUEENMAP_H
