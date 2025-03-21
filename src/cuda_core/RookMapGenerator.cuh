#ifndef SRC_ROOKMAPGENERATOR_CUH
#define SRC_ROOKMAPGENERATOR_CUH

#include <cuda_runtime.h>
#include <thrust/pair.h>

#include "cuda_Array.cuh"
#include "cuda_BitOperations.cuh"
#include "MoveGenerationUtils.cuh"
#include "Helpers.cuh"

__device__ static constexpr uint32_t rook_DirectedMaskCount = 4;
__device__ static constexpr uint32_t MaxRookPossibleNeighborsWoutOverlap = 144;
__device__ static constexpr uint32_t MaxRookPossibleNeighborsWithOverlap = 4096;
__device__ static constexpr int NorthOffset = 8;
__device__ static constexpr int SouthOffset = -8;
__device__ static constexpr int WestOffset = -1;
__device__ static constexpr int EastOffset = 1;

class RookMapGenerator final {
public:
    using MasksT = cuda_Array<uint64_t, rook_DirectedMaskCount>;

    // ---------------------------------------
    // Class creation and initialization
    // ---------------------------------------

    RookMapGenerator() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] HYBRID constexpr static MasksT InitMasks(uint32_t bInd) {
        constexpr int SouthBarrier = 7;
        constexpr int NorthBarrier = 56;

        cuda_Array<uint64_t, rook_DirectedMaskCount> ret{};

        // Mask generation.
        const int westBarrier = (bInd / 8) * 8;
        const int eastBarrier = westBarrier + 7;

        // South vertical line
        ret[sMask] = GenMask(SouthBarrier, bInd, SouthOffset, greater_comp<int>);

        // West Horizontal line
        ret[wMask] = GenMask(westBarrier, bInd, WestOffset, greater_comp<int>);

        // East horizontal line
        ret[eMask] = GenMask(eastBarrier, bInd, EastOffset, less_comp<int>);

        // North vertical line
        ret[nMask] = GenMask(NorthBarrier, bInd, NorthOffset, less_comp<int>);

        return ret;
    }

    [[nodiscard]] HYBRID constexpr static uint64_t GenMoves(uint64_t neighborsWoutOverlap, uint32_t bInd) {
        constexpr int northBarrier = 64;
        constexpr int southBarrier = -1;
        const int westBarrier = (bInd / 8) * 8 - 1;
        const int eastBarrier = westBarrier + 9;

        uint64_t moves = 0;

        // North lines moves
        moves |= GenSlidingMoves(
                neighborsWoutOverlap, bInd, NorthOffset, northBarrier, less_comp<int>
        );

        // South lines moves
        moves |= GenSlidingMoves(
                neighborsWoutOverlap, bInd, SouthOffset, southBarrier, greater_comp<int>
        );

        // East line moves
        moves |= GenSlidingMoves(
                neighborsWoutOverlap, bInd, EastOffset, eastBarrier, less_comp<int>
        );

        // West line moves
        moves |= GenSlidingMoves(
                neighborsWoutOverlap, bInd, WestOffset, westBarrier, greater_comp<int>
        );

        return moves;
    }


    [[nodiscard]] HYBRID constexpr static thrust::pair<cuda_Array<uint64_t, MaxRookPossibleNeighborsWoutOverlap>, uint32_t>
    GenPossibleNeighborsWoutOverlap(int bInd, const MasksT &masks) {
        cuda_Array<uint64_t, MaxRookPossibleNeighborsWoutOverlap> ret{};
        uint32_t usedFields = 0;

        const int westBarrier = ((bInd >> 3) << 3) - 1;
        const int eastBarrier = westBarrier + 9;
        constexpr int northBarrier = 64;
        constexpr int southBarrier = -1;

        const uint64_t bPos = 1LLU << bInd;
        for (int westCord = bInd; westCord > westBarrier; westCord += WestOffset) {
            const uint64_t westPos = cuda_MinMsbPossible << westCord;
            if (westPos != bPos && (masks[wMask] & westPos) == 0)
                continue;

            for (int eastCord = bInd; eastCord < eastBarrier; eastCord += EastOffset) {
                const uint64_t eastPos = cuda_MinMsbPossible << eastCord;
                if (eastPos != bPos && (masks[eMask] & eastPos) == 0)
                    continue;

                for (int northCord = bInd; northCord < northBarrier; northCord += NorthOffset) {
                    const uint64_t northPos = cuda_MinMsbPossible << northCord;
                    if (northPos != bPos && (masks[nMask] & northPos) == 0)
                        continue;

                    for (int southCord = bInd; southCord > southBarrier; southCord += SouthOffset) {
                        const uint64_t southPos = cuda_MinMsbPossible << southCord;
                        if (southPos != bPos && (masks[sMask] & southPos) == 0)
                            continue;

                        const uint64_t neighbor = (southPos | northPos | eastPos | westPos) & ~bPos;
                        ret[usedFields++] = neighbor;
                    }
                }
            }
        }

        return {ret, usedFields};
    }

    [[nodiscard]] HYBRID constexpr static thrust::pair<cuda_Array<uint64_t, MaxRookPossibleNeighborsWithOverlap>, uint32_t>
    GenPossibleNeighborsWithOverlap(const MasksT &masks) {
        cuda_Array<uint64_t, MaxRookPossibleNeighborsWithOverlap> ret{};
        const uint64_t fullMask = masks[nMask] | masks[sMask] | masks[eMask] | masks[wMask];

        uint32_t usedFields = GenerateBitPermutations(fullMask, ret);

        return {ret, usedFields};
    }

    [[nodiscard]] HYBRID static constexpr uint32_t PossibleNeighborWoutOverlapCountOnField(int x, int y) {
        const int westCount = cuda_max(1, x);
        const int southCount = cuda_max(1, y);
        const int northCount = cuda_max(1, 7 - y);
        const int eastCount = cuda_max(1, 7 - x);

        return westCount * eastCount * southCount * northCount;
    }

    [[nodiscard]] HYBRID static constexpr uint64_t
    StripBlockingNeighbors(uint64_t fullBoard, const MasksT &masks) {
        const uint64_t northPart = ExtractLsbBit(fullBoard & masks[nMask]);
        const uint64_t southPart = ExtractMsbBitConstexpr(fullBoard & masks[sMask]);
        const uint64_t westPart = ExtractMsbBitConstexpr(fullBoard & masks[wMask]);
        const uint64_t eastPart = ExtractLsbBit(fullBoard & masks[eMask]);
        return northPart | southPart | westPart | eastPart;
    }

    // ------------------------------
    // Class inner types
    // ------------------------------

    enum maskInd {
        wMask,
        eMask,
        nMask,
        sMask,
    };
};

#endif //SRC_ROOKMAPGENERATOR_CUH
