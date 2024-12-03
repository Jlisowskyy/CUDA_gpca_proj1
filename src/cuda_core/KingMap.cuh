//
// Created by Jlisowskyy on 12/30/23.
//

#ifndef KINGMAP_H
#define KINGMAP_H

#include <cstdint>

#include "KnightMap.cuh"
#include "cuda_Board.cuh"
#include "cuda_PackedBoard.cuh"
#include "cuda_Array.cuh"

namespace KingMapConstants {
    __device__ __constant__ static constexpr __uint32_t maxMovesCount = 8;

    // Describes king possible moves coordinates.
    __device__ __constant__ static constexpr int movesCords[] = {-1, -9, -8, -7, 1, 9, 8, 7};

    // Describes accordingly y positions after the move relatively to king's y position.
    // Used to omit errors during generation.
    __device__ __constant__ static constexpr int rowCords[] = {0, -1, -1, -1, 0, 1, 1, 1};

    alignas(128) __device__ static constexpr cuda_Array<__uint64_t, BIT_BOARD_FIELDS> movesMap =
            GenStaticMoves(maxMovesCount, movesCords, rowCords);

    // Masks used to detect allowed tiles when checked by pawn
    __device__ __constant__ static constexpr __uint64_t LeftPawnDetectionMask = ~GenMask(0, 57, 8);
    __device__ __constant__ static constexpr __uint64_t RightPawnDetectionMask = ~GenMask(7, 64, 8);
}

struct KingMap final {
    // ------------------------------
    // Class creation
    // ------------------------------

    KingMap() = delete;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint32_t GetBoardIndex(__uint32_t color) {
        return BIT_BOARDS_PER_COLOR * color + KING_INDEX;
    }

    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t
    GetMoves(__uint32_t msbInd) { return KingMapConstants::movesMap[msbInd]; }

    // generates tiles on which pawns currently attacks king
    template<__uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE>
    [[nodiscard]] FAST_DCALL_ALWAYS  static __uint64_t
    GetSimpleFigCheckPawnAllowedTiles(const cuda_PackedBoard<NUM_BOARDS>::BoardFetcher &fetcher) {
        const __uint64_t detectionFields =
                fetcher.MovingColor() == WHITE ? _getWhiteKingDetectionTiles<NUM_BOARDS>(fetcher)
                                               : _getBlackKingDetectionTiles<NUM_BOARDS>(fetcher);

        return detectionFields & fetcher.GetFigBoard(SwapColor(fetcher.MovingColor()), PAWN_INDEX);
    }

    template<__uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE>
    [[nodiscard]] FAST_DCALL_ALWAYS static __uint64_t
    GetSimpleFigCheckKnightsAllowedTiles(const cuda_PackedBoard<NUM_BOARDS>::BoardFetcher &fetcher) {
        const __uint64_t detectionFields = KnightMap::GetMoves(fetcher.GetKingMsbPos(fetcher.MovingColor()));

        return detectionFields & fetcher.GetFigBoard(SwapColor(fetcher.MovingColor()), KNIGHT_INDEX);
    }

    // ------------------------------
    // Private methods
    // ------------------------------

private:
    // generates possibles tiles on which enemy pawn could attack king
    template<__uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE>
    [[nodiscard]] FAST_DCALL_ALWAYS  static constexpr __uint64_t
    _getWhiteKingDetectionTiles(const cuda_PackedBoard<NUM_BOARDS>::BoardFetcher &fetcher) {
        const __uint64_t kingMap = fetcher.GetFigBoard(WHITE, KING_INDEX);


        const __uint64_t leftDetectionTile = (kingMap & KingMapConstants::LeftPawnDetectionMask) << 7;
        const __uint64_t rightDetectionTile = (kingMap & KingMapConstants::RightPawnDetectionMask) << 9;

        return leftDetectionTile | rightDetectionTile;
    }

    // generates possibles tiles on which enemy pawn could attack king
    template<__uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE>
    [[nodiscard]] FAST_DCALL_ALWAYS static constexpr __uint64_t
    _getBlackKingDetectionTiles(const cuda_PackedBoard<NUM_BOARDS>::BoardFetcher &fetcher) {
        const __uint64_t kingMap = fetcher.GetFigBoard(BLACK, KING_INDEX);

        const __uint64_t rightDetectionTile = (kingMap & KingMapConstants::RightPawnDetectionMask) >> 7;
        const __uint64_t leftDetectionTile = (kingMap & KingMapConstants::LeftPawnDetectionMask) >> 9;

        return leftDetectionTile | rightDetectionTile;
    }

};


#endif // KINGMAP_H
