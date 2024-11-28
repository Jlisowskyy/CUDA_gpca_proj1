//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_MOVEGENERATIONUTILS_CUH
#define SRC_MOVEGENERATIONUTILS_CUH

#include <cuda_runtime.h>

#include "cuda_Board.cuh"
#include "cuda_BitOperations.cuh"
#include "cuda_Array.cuh"

HYBRID [[nodiscard]] constexpr cuda_Array<__uint64_t, BIT_BOARD_FIELDS>
GenStaticMoves(const size_t maxMovesCount, const int *movesCords, const int *rowCords) {
    cuda_Array<__uint64_t, BIT_BOARD_FIELDS> movesRet{};

    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            const int bInd = 8 * y + x;
            const int msbInd = ConvertToReversedPos(bInd);

            __uint64_t packedMoves = 0;
            for (size_t i = 0; i < maxMovesCount; ++i) {
                const int moveYCord = (bInd + movesCords[i]) / 8;
                const int knightYCord = bInd / 8;

                if (const int moveInd = bInd + movesCords[i];
                        moveInd >= 0 && moveInd < 64 && moveYCord == knightYCord + rowCords[i])
                    packedMoves |= 1LLU << moveInd;
            }

            movesRet[msbInd] = packedMoves;
        }
    }

    return movesRet;
}

template<class NeighborCountingFuncT>
HYBRID [[nodiscard]] constexpr size_t
CalculateTotalOfPossibleHashMapElements(NeighborCountingFuncT func) {
    size_t sum{};
    for (int x = 0; x < 8; ++x) {
        for (int y = 0; y < 8; ++y) {
            sum += func(x, y);
        }
    }

    return
            sum;
}

template<class ComparisonMethodT>
HYBRID [[nodiscard]] constexpr __uint64_t
GenMask(const int barrier, int boardIndex, const int offset, ComparisonMethodT comp) {
    __uint64_t mask = 0;

    while (comp(boardIndex += offset, barrier)) mask |= (1LLU << boardIndex);

    return mask;
}

HYBRID [[nodiscard]] constexpr __uint64_t
GenMask(const int startInd, const int boarderIndex, const int offset) {
    __uint64_t ret = 0;
    for (int i = startInd; i < boarderIndex; i += offset) ret |= (1LLU << i);
    return ret;
}

HYBRID [[nodiscard]] constexpr size_t

MyCeil(const double x) {
    return (static_cast<double>(static_cast<size_t>(x)) == x) ? static_cast<size_t>(x)
                                                              : static_cast<size_t>(x) + ((x > 0) ? 1 : 0);
}


template<class BoundaryCheckFuncT>
HYBRID constexpr __uint64_t
GenSlidingMoves(const __uint64_t neighbors, const int bInd, const int offset, const int boundary, BoundaryCheckFuncT boundaryCheck) {
    __uint64_t ret = 0;
    int actPos = bInd;

    while (boundaryCheck(actPos += offset, boundary)) {
        const __uint64_t curMove = 1LLU << actPos;
        ret |= curMove;

        if ((curMove & neighbors) != 0)
            break;
    }

    return ret;
}

template<class MoveGeneratorT, class NeighborGeneratorT, class NeighborStripT, class MapT>
HYBRID constexpr void
MoveInitializer(MapT &map, MoveGeneratorT mGen, NeighborGeneratorT nGen, NeighborStripT nStrip, const int bInd) {
    const auto [possibilities, posSize] = nGen(map.getMasks());

    for (size_t j = 0; j < posSize; ++j) {
        const __uint64_t strippedNeighbors = nStrip(possibilities[j], map.getMasks());
        const __uint64_t moves = mGen(strippedNeighbors, bInd);

        map[possibilities[j]] = moves;
    }
}

#endif //SRC_MOVEGENERATIONUTILS_CUH
