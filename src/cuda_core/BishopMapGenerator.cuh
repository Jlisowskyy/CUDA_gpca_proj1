//
// Created by Jlisowskyy on 16/11/24.
//

#ifndef SRC_BISHOPMAPGENERATOR_CUH
#define SRC_BISHOPMAPGENERATOR_CUH

#include <cuda_runtime.h>
#include <thrust/pair.h>

#include "MoveGenerationUtils.cuh"
#include "cuda_BitOperations.cuh"
#include "Helpers.cuh"

#include "cuda_Array.cuh"

__device__ __constant__ static constexpr __uint32_t MaxPossibleNeighborsWoutOverlap = 108;
__device__ __constant__ static constexpr __uint32_t MaxPossibleNeighborsWithOverlap = 512;

__device__ __constant__ static constexpr int NWOffset = 7;
__device__ __constant__ static constexpr int NEOffset = 9;
__device__ __constant__ static constexpr int SWOffset = -9;
__device__ __constant__ static constexpr int SEOffset = -7;

__device__ __constant__ static constexpr __uint32_t DirectedMaskCount = 4;


class BishopMapGenerator final {
public:
    using MasksT = cuda_Array<__uint64_t, DirectedMaskCount>;

    // ---------------------------------------
    // Class creation and initialization
    // ---------------------------------------

    BishopMapGenerator() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] __device__ static constexpr thrust::pair<cuda_Array<__uint64_t, MaxPossibleNeighborsWoutOverlap>, __uint32_t>
    GenPossibleNeighborsWoutOverlap(int bInd, const MasksT &masks) {
        cuda_Array<__uint64_t, MaxPossibleNeighborsWoutOverlap> ret{};
        __uint32_t usedFields = 0;

        const int x = bInd % 8;
        const int y = bInd / 8;

        const int NWBorder = bInd + NWOffset * cuda_min(x, 7 - y);
        const int NEBorder = bInd + NEOffset * cuda_min(7 - x, 7 - y);
        const int SWBorder = bInd + SWOffset * cuda_min(x, y);
        const int SEBorder = bInd + SEOffset * cuda_min(7 - x, y);

        const __uint64_t bPos = 1LLU << bInd;
        for (int nw = bInd; nw <= NWBorder; nw += NWOffset) {
            const __uint64_t nwPos = cuda_MinMsbPossible << nw;
            if (nwPos != bPos && (masks[nwMask] & nwPos) == 0)
                continue;

            for (int ne = bInd; ne <= NEBorder; ne += NEOffset) {
                const __uint64_t nePos = cuda_MinMsbPossible << ne;
                if (nePos != bPos && (masks[neMask] & nePos) == 0)
                    continue;

                for (int sw = bInd; sw >= SWBorder; sw += SWOffset) {
                    const __uint64_t swPos = cuda_MinMsbPossible << sw;
                    if (swPos != bPos && (masks[swMask] & swPos) == 0)
                        continue;

                    for (int se = bInd; se >= SEBorder; se += SEOffset) {
                        const __uint64_t sePos = cuda_MinMsbPossible << se;
                        if (sePos != bPos && (masks[seMask] & sePos) == 0)
                            continue;

                        const __uint64_t neighbor = (nwPos | nePos | swPos | sePos) & ~bPos;
                        ret[usedFields++] = neighbor;
                    }
                }
            }
        }

        return {ret, usedFields};
    }

    [[nodiscard]] __device__ static constexpr thrust::pair<cuda_Array<__uint64_t, MaxPossibleNeighborsWithOverlap>, __uint32_t>
    GenPossibleNeighborsWithOverlap(const MasksT &masks) {
        cuda_Array<__uint64_t, MaxPossibleNeighborsWithOverlap> ret{};
        const __uint64_t fullMask = masks[neMask] | masks[nwMask] | masks[seMask] | masks[swMask];

        __uint32_t usedFields = GenerateBitPermutations(fullMask, ret);

        return {ret, usedFields};
    }

    [[nodiscard]] __device__ static constexpr __uint64_t GenMoves(__uint64_t neighborsWoutOverlap, int bInd) {
        const int y = bInd / 8;
        const int x = bInd % 8;

        const int NWBorder = bInd + 7 * cuda_min(x, 7 - y);
        const int NEBorder = bInd + 9 * cuda_min(7 - x, 7 - y);
        const int SWBorder = bInd - 9 * cuda_min(x, y);
        const int SEBorder = bInd - 7 * cuda_min(7 - x, y);

        __uint64_t moves = 0;

        // NW direction moves
        moves |= GenSlidingMoves(
                neighborsWoutOverlap, bInd, NWOffset, NWBorder, less_equal_comp<int>
        );

        // NE direction moves
        moves |= GenSlidingMoves(
                neighborsWoutOverlap, bInd, NEOffset, NEBorder, less_equal_comp<int>
        );

        // SW direction moves
        moves |= GenSlidingMoves(
                neighborsWoutOverlap, bInd, SWOffset, SWBorder, greater_equal_comp<int>
        );

        // SE direction moves
        moves |= GenSlidingMoves(
                neighborsWoutOverlap, bInd, SEOffset, SEBorder, greater_equal_comp<int>
        );

        return moves;
    }

    [[nodiscard]] __device__ static constexpr __uint32_t PossibleNeighborWoutOverlapCountOnField(int x, int y) {
        const __uint32_t nwCount = cuda_max(1, cuda_min(x, 7 - y));
        const __uint32_t neCount = cuda_max(1, cuda_min(7 - x, 7 - y));
        const __uint32_t swCount = cuda_max(1, cuda_min(x, y));
        const __uint32_t seCount = cuda_max(1, cuda_min(7 - x, y));

        return nwCount * neCount * swCount * seCount;
    }

    [[nodiscard]] __device__ static constexpr MasksT InitMasks(int bInd) {
        cuda_Array<__uint64_t, DirectedMaskCount> ret{};
        const int x = bInd % 8;
        const int y = bInd / 8;

        const int NEBorder = bInd + NEOffset * cuda_max(0, cuda_min(7 - x, 7 - y) - 1);
        const int NWBorder = bInd + NWOffset * cuda_max(0, cuda_min(x, 7 - y) - 1);
        const int SEBorder = bInd + SEOffset * cuda_max(0, cuda_min(7 - x, y) - 1);
        const int SWBorder = bInd + SWOffset * cuda_max(0, cuda_min(x, y) - 1);

        ret[neMask] = GenMask(NEBorder, bInd, NEOffset, &less_equal_comp<int>);
        ret[nwMask] = GenMask(NWBorder, bInd, NWOffset, &less_equal_comp<int>);
        ret[seMask] = GenMask(SEBorder, bInd, SEOffset, &greater_equal_comp<int>);
        ret[swMask] = GenMask(SWBorder, bInd, SWOffset, &greater_equal_comp<int>);

        return ret;
    }

    [[nodiscard]] __device__ static constexpr __uint64_t
    StripBlockingNeighbors(__uint64_t fullBoard, const MasksT &masks) {
        const __uint64_t NWPart = ExtractLsbBit(fullBoard & masks[nwMask]);
        const __uint64_t NEPart = ExtractLsbBit(fullBoard & masks[neMask]);
        const __uint64_t SWPart = ExtractMsbBitConstexpr(fullBoard & masks[swMask]);
        const __uint64_t SEPart = ExtractMsbBitConstexpr(fullBoard & masks[seMask]);
        return NWPart | NEPart | SWPart | SEPart;
    }

    // ------------------------------
    // Class inner types
    // ------------------------------

    enum maskInd {
        nwMask,
        neMask,
        swMask,
        seMask,
    };
};

#endif //SRC_BISHOPMAPGENERATOR_CUH
