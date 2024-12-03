//
// Created by Jlisowskyy on 12/31/23.
//

#ifndef CHESSMECHANICS_CUH
#define CHESSMECHANICS_CUH

#include <cuda_runtime.h>
#include <thrust/tuple.h>
#include <thrust/pair.h>

#include "Helpers.cuh"

#include "cuda_BitOperations.cuh"
#include "cuda_Board.cuh"
#include "cuda_PackedBoard.cuh"
#include "Move.cuh"

#include "BishopMap.cuh"
#include "RookMap.cuh"
#include "KnightMap.cuh"
#include "KingMap.cuh"
#include "WhitePawnMap.cuh"
#include "BlackPawnMap.cuh"

#include <cassert>
#include <cinttypes>

struct MoveGenDataMem {
    __uint32_t checksCount{};
    __uint32_t wasCheckedBySimple{};
    __uint64_t blockedMap{};
};

//#define ASSERT(cond, msg) ASSERT_DISPLAY(&_board, cond, msg)
#define ASSERT(cond, msg) assert(cond && msg)

template<__uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE>
struct ChessMechanics {
    // ------------------------------
    // Class inner types
    // ------------------------------

    using _fetcher_t = cuda_PackedBoard<NUM_BOARDS>::BoardFetcher;

    enum checkType {
        slidingFigCheck,
        simpleFigCheck
    };

    enum class PinnedFigGen {
        WAllowedTiles,
        WoutAllowedTiles
    };

    // ------------------------------
    // Class Creation
    // ------------------------------

    ChessMechanics() = delete;

    FAST_CALL explicit ChessMechanics(_fetcher_t fetcher, MoveGenDataMem *md) : _boardFetcher(fetcher),
                                                                                _moveGenData(md) {}

    ChessMechanics(ChessMechanics &other) = delete;

    ChessMechanics &operator=(ChessMechanics &) = delete;

    ~ChessMechanics() = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    // Gets occupancy maps, which simply indicates whether some field is occupied or not. Does not distinguish colors.
    [[nodiscard]] FAST_DCALL __uint64_t GetFullBitMap() const {
        __uint64_t map = 0;

        for (__uint32_t i = 0; i < BIT_BOARDS_COUNT; ++i) {
            map |= _boardFetcher.BitBoard(i);
        }

        return map;
    }

    // Gets occupancy maps, which simply indicates whether some field is occupied or not, by desired color figures.
    [[nodiscard]] FAST_DCALL __uint64_t GetColBitMap(const __uint32_t col) const {
        ASSERT(col == 1 || col == 0, "col == 1 || col == 0");
        __uint64_t map = 0;

        for (__uint32_t i = 0; i < BIT_BOARDS_PER_COLOR; ++i) {
            map |= _boardFetcher.GetFigBoard(col, i);
        }

        return map;
    }

    // does not check kings BitBoards!!!
    [[nodiscard]] FAST_DCALL __uint32_t
    GetIndexOfContainingBitBoard(const __uint64_t map, const __uint32_t col) const {
        __uint32_t rv = 0;

        for (__uint32_t i = 0; i < BIT_BOARDS_PER_COLOR; ++i) {
            rv += ((_boardFetcher.GetFigBoard(col, i) & map) != 0) * i;
        }

        return col * BIT_BOARDS_PER_COLOR + rv;
    }

    /*      IMPORTANT NOTES:
    *  BlockedFieldsMap - indicates whether some field could be attacked by enemy figures in their next round.
    *  During generation process there are check test performed with counting, what yields 3 code branches inside main
    *  generation code. Map is mainly used when king is moving, allowing to simply predict whether king should
    *  move to that tile or not.
    *
    */

    // [blockedFigMap, checksCount, checkType]
    [[nodiscard]] __device__ thrust::tuple<__uint64_t, __uint8_t, bool>
    GetBlockedFieldBitMap(const __uint32_t movingColor, const __uint64_t fullMap) const {
        ASSERT(fullMap != 0, "Full map is empty!");

        __uint8_t checksCount{};
        __uint64_t blockedMap{};
        bool wasCheckedBySimple{};

        const __uint32_t enemyFigInd = SwapColor(movingColor) * BIT_BOARDS_PER_COLOR;
        const __uint32_t allyKingShift = ConvertToReversedPos(_boardFetcher.GetKingMsbPos(movingColor));
        const __uint64_t allyKingMap = cuda_MinMsbPossible << allyKingShift;

        // King attacks generation.
        blockedMap |= KingMap::GetMoves(_boardFetcher.GetKingMsbPos(SwapColor(movingColor)));

        // Rook attacks generation. Needs special treatment to correctly detect double check, especially with pawn promotion
        const auto [rookBlockedMap, checkCountRook] = _getRookBlockedMap(
                _boardFetcher.BitBoard(enemyFigInd + ROOK_INDEX) | _boardFetcher.BitBoard(enemyFigInd + QUEEN_INDEX),
                fullMap ^ allyKingMap,
                allyKingMap
        );

        // = 0, 1 or eventually 2 when promotion to rook like type happens
        checksCount += checkCountRook;
        blockedMap |= rookBlockedMap;

        // Bishop attacks generation.
        const __uint64_t bishopBlockedMap = _blockIterativeGenerator(
                _boardFetcher.BitBoard(enemyFigInd + BISHOP_INDEX) | _boardFetcher.BitBoard(enemyFigInd + QUEEN_INDEX),
                [=](const int pos) {
                    return BishopMap::GetMoves(pos, fullMap ^ allyKingMap);
                }
        );

        // = 1 or 0 depending on whether hits or not
        const __uint8_t wasCheckedByBishopFlag = (bishopBlockedMap & allyKingMap) >> allyKingShift;

        checksCount += wasCheckedByBishopFlag;
        blockedMap |= bishopBlockedMap;

        // Pawns attacks generation.
        const __uint64_t pawnsMap = _boardFetcher.BitBoard(enemyFigInd + PAWN_INDEX);
        const __uint64_t pawnBlockedMap =
                SwapColor(movingColor) == WHITE ? WhitePawnMap::GetAttackFields(pawnsMap)
                                                       : BlackPawnMap::GetAttackFields(pawnsMap);

        const bool wasCheckedByPawnFlag = pawnBlockedMap & allyKingMap;

        checksCount += wasCheckedByPawnFlag;
        wasCheckedBySimple |= wasCheckedByPawnFlag;
        blockedMap |= pawnBlockedMap;

        // Knight attacks generation.
        const __uint64_t knightBlockedMap = _blockIterativeGenerator(
                _boardFetcher.BitBoard(enemyFigInd + KNIGHT_INDEX),
                [=](const int pos) {
                    return KnightMap::GetMoves(pos);
                }
        );

        const bool wasCheckedByKnightFlag = knightBlockedMap & allyKingMap;

        checksCount += wasCheckedByKnightFlag;
        wasCheckedBySimple |= wasCheckedByKnightFlag;
        blockedMap |= knightBlockedMap;

        return {blockedMap, checksCount, wasCheckedBySimple};
    }

    // [blockedFigMap, checksCount, checkType]
    [[nodiscard]] __device__ thrust::tuple<__uint64_t, __uint8_t, bool>
    GetBlockedFieldBitMapSplit(const __uint32_t movingColor, const __uint64_t fullMap, const __uint32_t figIdx) const {
        ASSERT(fullMap != 0, "Full map is empty!");

        const __uint32_t enemyFigInd = SwapColor(movingColor) * BIT_BOARDS_PER_COLOR;
        const __uint32_t allyKingShift = ConvertToReversedPos(_boardFetcher.GetKingMsbPos(movingColor));
        const __uint64_t allyKingMap = cuda_MinMsbPossible << allyKingShift;

        // Specific figure processing
        switch (figIdx) {
            case PAWN_INDEX: {
                // Pawns attacks generation.
                const __uint64_t pawnsMap = _boardFetcher.BitBoard(enemyFigInd + PAWN_INDEX);
                const __uint64_t pawnBlockedMap =
                        SwapColor(movingColor) == WHITE ? WhitePawnMap::GetAttackFields(pawnsMap)
                                                               : BlackPawnMap::GetAttackFields(pawnsMap);

                const bool wasCheckedByPawnFlag = pawnBlockedMap & allyKingMap;

                atomicAdd((unsigned int *) &(_moveGenData->checksCount), wasCheckedByPawnFlag);
                atomicOr((unsigned long long int *) &(_moveGenData->blockedMap), pawnBlockedMap);
                atomicAdd((unsigned int *) &(_moveGenData->wasCheckedBySimple), wasCheckedByPawnFlag);
            }
                break;
            case KNIGHT_INDEX: {
                // Knight attacks generation.
                const __uint64_t knightBlockedMap = _blockIterativeGenerator(
                        _boardFetcher.BitBoard(enemyFigInd + KNIGHT_INDEX),
                        [=](const int pos) {
                            return KnightMap::GetMoves(pos);
                        }
                );

                const bool wasCheckedByKnightFlag = knightBlockedMap & allyKingMap;

                atomicAdd((unsigned int *) &(_moveGenData->checksCount), wasCheckedByKnightFlag);
                atomicOr((unsigned long long int *) &(_moveGenData->blockedMap), knightBlockedMap);
                atomicAdd((unsigned int *) &(_moveGenData->wasCheckedBySimple), wasCheckedByKnightFlag);
            }
                break;
            case BISHOP_INDEX: {
                // Bishop attacks generation.
                const __uint64_t bishopBlockedMap = _blockIterativeGenerator(
                        _boardFetcher.BitBoard(enemyFigInd + BISHOP_INDEX) |
                        _boardFetcher.BitBoard(enemyFigInd + QUEEN_INDEX),
                        [=](const int pos) {
                            return BishopMap::GetMoves(pos, fullMap ^ allyKingMap);
                        }
                );

                // = 1 or 0 depending on whether hits or not
                const __uint8_t wasCheckedByBishopFlag = (bishopBlockedMap & allyKingMap) >> allyKingShift;

                atomicAdd((unsigned int *) &(_moveGenData->checksCount), wasCheckedByBishopFlag);
                atomicOr((unsigned long long int *) &(_moveGenData->blockedMap), bishopBlockedMap);
            }
                break;
            case ROOK_INDEX: {
                // Rook attacks generation. Needs special treatment to correctly detect double check, especially with pawn promotion
                const auto [rookBlockedMap, checkCountRook] = _getRookBlockedMap(
                        _boardFetcher.BitBoard(enemyFigInd + ROOK_INDEX) |
                        _boardFetcher.BitBoard(enemyFigInd + QUEEN_INDEX),
                        fullMap ^ allyKingMap,
                        allyKingMap
                );

                atomicAdd((unsigned int *) &(_moveGenData->checksCount), checkCountRook);
                atomicOr((unsigned long long int *) &(_moveGenData->blockedMap), rookBlockedMap);
            }
                break;
            case QUEEN_INDEX:
                break;
            case KING_INDEX: {
                const __uint64_t kingMap = KingMap::GetMoves(
                        _boardFetcher.GetKingMsbPos(SwapColor(movingColor)));
                atomicOr((unsigned long long int *) &(_moveGenData->blockedMap), kingMap);
            }
                break;
            default:
                ASSERT(false, "Shit happens");
        }

        __syncthreads();
        return {_moveGenData->blockedMap, (__uint8_t) _moveGenData->checksCount,
                (bool) _moveGenData->wasCheckedBySimple};
    }

    [[nodiscard]] FAST_DCALL __uint64_t
    GenerateAllowedTilesForPrecisedPinnedFig(const __uint32_t movingColor, const __uint64_t figBoard,
                                             const __uint64_t fullMap) const {
        ASSERT(fullMap != 0, "Full map is empty!");
        ASSERT(CountOnesInBoard(figBoard) == 1, "Only one figure should be pinned!");

        const __uint32_t msbPos = ExtractMsbPos(figBoard);
        const __uint64_t KingBoard = _boardFetcher.GetFigBoard(movingColor, KING_INDEX);

        if (const __uint64_t RookPerspectiveMoves = RookMap::GetMoves(msbPos, fullMap);
                (RookPerspectiveMoves & KingBoard) != 0) {

            return RookPerspectiveMoves & RookMap::GetMoves(ExtractMsbPos(KingBoard), fullMap ^ figBoard);
        }

        return BishopMap::GetMoves(msbPos, fullMap) & BishopMap::GetMoves(ExtractMsbPos(KingBoard), fullMap ^ figBoard);
    }

    // returns [ pinnedFigMap, allowedTilesMap ]
    template<PinnedFigGen genType>
    [[nodiscard]] FAST_CALL thrust::pair<__uint64_t, __uint64_t>
    GetPinnedFigsMap(const __uint32_t col, const __uint64_t fullMap) const {
        ASSERT(fullMap != 0, "Full map is empty!");
        ASSERT(col == 1 || col == 0, "Invalid color!");

        const __uint32_t enemyCord = SwapColor(col) * BIT_BOARDS_PER_COLOR;

        const auto [pinnedByRooks, allowedRooks] = _getPinnedFigMaps<RookMap, genType>(
                col, fullMap,
                _boardFetcher.BitBoard(enemyCord + ROOK_INDEX) | _boardFetcher.BitBoard(enemyCord + QUEEN_INDEX)
        );

        const auto [pinnedByBishops, allowedBishops] = _getPinnedFigMaps<BishopMap, genType>(
                col, fullMap,
                _boardFetcher.BitBoard(enemyCord + BISHOP_INDEX) | _boardFetcher.BitBoard(enemyCord + QUEEN_INDEX)
        );

        return {pinnedByBishops | pinnedByRooks, allowedBishops | allowedRooks};
    }

    [[nodiscard]] FAST_DCALL_ALWAYS __uint64_t GetAllowedTilesWhenCheckedByNonSliding() const {
        __uint64_t allowedTiles{};

        allowedTiles |= KingMap::GetSimpleFigCheckKnightsAllowedTiles<NUM_BOARDS>(_boardFetcher);
        allowedTiles |= KingMap::GetSimpleFigCheckPawnAllowedTiles<NUM_BOARDS>(_boardFetcher);

        return allowedTiles;
    }


    // ------------------------------
    // private methods
    // ------------------------------

private:

    /*  Function collects information used inside the SEE algorithm it contains:
     *  - attackersBitBoard - contains every type of figure that in current state of the board could attack given field
     *  - fullMap - contains every figure on the board
     *  - xrayMap - contains every figure that attack could be potentially unlocked after other figures move,
     *              that is: queens, bishops, rooks and pawns
     * */

    FAST_DCALL thrust::pair<__uint64_t, __uint8_t>
    _getRookBlockedMap(__uint64_t rookMap, const __uint64_t fullMapWoutKing, const __uint64_t kingMap) const {
        ASSERT(kingMap != 0, "King map is empty!");

        __uint64_t blockedTiles{};
        __uint8_t checks{};

        while (rookMap) {
            const __uint32_t msbPos = ExtractMsbPos(rookMap);

            const __uint64_t moves = RookMap::GetMoves(msbPos, fullMapWoutKing);
            blockedTiles |= moves;
            checks += ((moves & kingMap) != 0);

            rookMap ^= cuda_MaxMsbPossible >> msbPos;
        }

        return {blockedTiles, checks};
    }

    template<class MoveGeneratorT>
    [[nodiscard]] FAST_DCALL_ALWAYS static __uint64_t _blockIterativeGenerator(__uint64_t board, MoveGeneratorT mGen) {
        __uint64_t blockedMap = 0;

        while (board != 0) {
            const __uint32_t figPos = ExtractMsbPos(board);
            board ^= (cuda_MaxMsbPossible >> figPos);

            blockedMap |= mGen(figPos);
        }

        return blockedMap;
    }

    // returns [ pinnedFigMap, allowedTilesMap ]
    template<class MoveMapT, PinnedFigGen type>
    [[nodiscard]] FAST_DCALL thrust::pair<__uint64_t, __uint64_t>
    _getPinnedFigMaps(const __uint32_t movingColor, const __uint64_t fullMap, const __uint64_t possiblePinningFigs) const {
        __uint64_t allowedTilesFigMap{};
        [[maybe_unused]] __uint64_t pinnedFigMap{};

        const __uint32_t kingPos = _boardFetcher.GetKingMsbPos(movingColor);
        // generating figs seen from king's rook perspective
        const __uint64_t kingFigPerspectiveAttackedFields = MoveMapT::GetMoves(kingPos, fullMap);
        const __uint64_t kingFigPerspectiveAttackedFigs = kingFigPerspectiveAttackedFields & fullMap;

        // this functions should be called only in case of single check so the value below can only be either null or the
        // map of checking figure
        if constexpr (type == PinnedFigGen::WAllowedTiles) {
            if (const __uint64_t kingSeenEnemyFigs = kingFigPerspectiveAttackedFigs & possiblePinningFigs;
                    kingSeenEnemyFigs != 0) {
                const __uint32_t msbPos = ExtractMsbPos(kingSeenEnemyFigs);
                const __uint64_t moves = MoveMapT::GetMoves(msbPos, fullMap);

                allowedTilesFigMap = (moves & kingFigPerspectiveAttackedFields) | kingSeenEnemyFigs;
            }
        }

        // removing figs seen by king
        const __uint64_t cleanedMap = fullMap ^ kingFigPerspectiveAttackedFigs;

        // generating figs, which stayed behind first ones and are actually pinning ones
        const __uint64_t kingSecondRookPerspective = MoveMapT::GetMoves(kingPos, cleanedMap);
        __uint64_t pinningFigs = possiblePinningFigs & kingSecondRookPerspective;

        // generating fields which are both seen by king and pinning figure = field on which pinned figure stays
        while (pinningFigs != 0) {
            const __uint32_t msbPos = ExtractMsbPos(pinningFigs);
            pinnedFigMap |= MoveMapT::GetMoves(msbPos, fullMap) & kingFigPerspectiveAttackedFigs;
            pinningFigs ^= cuda_MaxMsbPossible >> msbPos;
        }

        return {pinnedFigMap, allowedTilesFigMap};
    }

    // ------------------------------
    // Class fields
    // ------------------------------

protected:
    const _fetcher_t _boardFetcher;
    MoveGenDataMem *_moveGenData;
};


#endif // CHESSMECHANICS_CUH
