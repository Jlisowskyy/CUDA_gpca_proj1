//
// Created by Jlisowskyy on 12/31/23.
//

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

    [[nodiscard]] FAST_CALL static constexpr size_t GetBoardIndex(__uint32_t color) {
        return BitBoardsPerCol * color + queensIndex;
    }

    [[nodiscard]] FAST_CALL static constexpr __uint64_t
    GetMoves(__uint32_t msbInd, __uint64_t fullMap, [[maybe_unused]] __uint64_t = 0) {
        return BishopMap::GetMoves(msbInd, fullMap) | RookMap::GetMoves(msbInd, fullMap);
    }
};

#endif // QUEENMAP_H
