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
    uint32_t checksCount;
    uint32_t wasCheckedBySimple;
    uint64_t blockedMap;
};

//#define ASSERT(cond, msg) ASSERT_DISPLAY(&_board, cond, msg)
#define ASSERT(cond, msg) assert(cond && msg)

template<uint32_t NUM_BOARDS = PACKED_BOARD_DEFAULT_SIZE>
struct ChessMechanics {
    // ------------------------------
    // Class inner types
    // ------------------------------

    using _fetcher_t = typename cuda_PackedBoard<NUM_BOARDS>::BoardFetcher;

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

    [[nodiscard]] FAST_DCALL bool IsCheck(const uint32_t movingColor) const {
        const uint32_t enemyCol = SwapColor(movingColor);
        const uint32_t kingsMsb = _boardFetcher.GetKingMsbPos(movingColor);
        const uint64_t fullBoard = GetFullBitMap();

        // Checking rook's perspective
        const uint64_t enemyRooks = _boardFetcher.GetFigBoard(enemyCol, ROOK_INDEX);
        const uint64_t enemyQueens = _boardFetcher.GetFigBoard(enemyCol, QUEEN_INDEX);
        const uint64_t kingsRookPerspective = RookMap::GetMoves(kingsMsb, fullBoard);

        if ((kingsRookPerspective & (enemyRooks | enemyQueens)) != 0) {
            return true;
        }

        // Checking bishop's perspective
        const uint64_t enemyBishops = _boardFetcher.GetFigBoard(enemyCol, BISHOP_INDEX);
        const uint64_t kingsBishopPerspective = BishopMap::GetMoves(kingsMsb, fullBoard);

        if ((kingsBishopPerspective & (enemyBishops | enemyQueens)) != 0) {
            return true;
        }

        // checking knights attacks
        const uint64_t enemyKnights = _boardFetcher.GetFigBoard(enemyCol, KNIGHT_INDEX);
        const uint64_t knightsPerspective = KnightMap::GetMoves(kingsMsb);

        if ((knightsPerspective & (enemyKnights)) != 0) {
            return true;
        }

        // pawns checks
        const uint64_t enemyPawns = _boardFetcher.GetFigBoard(enemyCol, PAWN_INDEX);
        const uint64_t pawnAttacks =
                enemyCol == WHITE ? WhitePawnMap::GetAttackFields(enemyPawns) : BlackPawnMap::GetAttackFields(
                        enemyPawns);

        if ((pawnAttacks & (cuda_MaxMsbPossible >> kingsMsb)) != 0) {
            return true;
        }

        return false;
    }

    [[nodiscard]] FAST_DCALL uint32_t EvalBoardsNoMoves(uint32_t movingColor) const {
        const bool isCheck = IsCheck(movingColor);
        return isCheck ? SwapColor(movingColor) : DRAW;
    }

    // Gets occupancy maps, which simply indicates whether some field is occupied or not. Does not distinguish colors.
    [[nodiscard]] FAST_DCALL uint64_t GetFullBitMap() const {
        uint64_t map = 0;

        #pragma unroll
        for (uint32_t i = 0; i < BIT_BOARDS_COUNT; ++i) {
            map |= _boardFetcher.BitBoard(i);
        }

        return map;
    }

    // Gets occupancy maps, which simply indicates whether some field is occupied or not, by desired color figures.
    [[nodiscard]] FAST_DCALL uint64_t GetColBitMap(const uint32_t col) const {
        ASSERT(col == 1 || col == 0, "col == 1 || col == 0");
        uint64_t map = 0;

        #pragma unroll
        for (uint32_t i = 0; i < BIT_BOARDS_PER_COLOR; ++i) {
            map |= _boardFetcher.GetFigBoard(col, i);
        }

        return map;
    }

    // does not check kings BitBoards!!!
    [[nodiscard]] FAST_DCALL uint32_t
    GetIndexOfContainingBitBoard(const uint64_t map, const uint32_t col) const {
        uint32_t rv = 0;

        #pragma unroll
        for (uint32_t i = 0; i < BIT_BOARDS_PER_COLOR; ++i) {
            rv += ((_boardFetcher.GetFigBoard(col, i) & map) != 0) * i;
        }

        return col * BIT_BOARDS_PER_COLOR + rv;
    }

    // [blockedFigMap, checksCount, checkType]
    [[nodiscard]] __device__ thrust::tuple<uint64_t, uint8_t, bool>
    GetBlockedFieldBitMap(const uint32_t movingColor, const uint64_t fullMap) const {
        ASSERT(fullMap != 0, "Full map is empty!");

        uint8_t checksCount{};
        uint64_t blockedMap{};
        bool wasCheckedBySimple{};

        const uint32_t enemyFigInd = SwapColor(movingColor) * BIT_BOARDS_PER_COLOR;
        const uint32_t allyKingShift = ConvertToReversedPos(_boardFetcher.GetKingMsbPos(movingColor));
        const uint64_t allyKingMap = cuda_MinMsbPossible << allyKingShift;

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
        const uint64_t bishopBlockedMap = _blockIterativeGenerator(
                _boardFetcher.BitBoard(enemyFigInd + BISHOP_INDEX) | _boardFetcher.BitBoard(enemyFigInd + QUEEN_INDEX),
                [=](const int pos) {
                    return BishopMap::GetMoves(pos, fullMap ^ allyKingMap);
                }
        );

        // = 1 or 0 depending on whether hits or not
        const uint8_t wasCheckedByBishopFlag = (bishopBlockedMap & allyKingMap) >> allyKingShift;

        checksCount += wasCheckedByBishopFlag;
        blockedMap |= bishopBlockedMap;

        // Pawns attacks generation.
        const uint64_t pawnsMap = _boardFetcher.BitBoard(enemyFigInd + PAWN_INDEX);
        const uint64_t pawnBlockedMap =
                SwapColor(movingColor) == WHITE ? WhitePawnMap::GetAttackFields(pawnsMap)
                                                       : BlackPawnMap::GetAttackFields(pawnsMap);

        const bool wasCheckedByPawnFlag = pawnBlockedMap & allyKingMap;

        checksCount += wasCheckedByPawnFlag;
        wasCheckedBySimple |= wasCheckedByPawnFlag;
        blockedMap |= pawnBlockedMap;

        // Knight attacks generation.
        const uint64_t knightBlockedMap = _blockIterativeGenerator(
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

    [[nodiscard]] FAST_DCALL uint64_t
    GenerateAllowedTilesForPrecisedPinnedFig(const uint32_t movingColor, const uint64_t figBoard,
                                             const uint64_t fullMap) const {
        ASSERT(fullMap != 0, "Full map is empty!");
        ASSERT(CountOnesInBoard(figBoard) == 1, "Only one figure should be pinned!");

        const uint32_t msbPos = ExtractMsbPos(figBoard);
        const uint64_t KingBoard = _boardFetcher.GetFigBoard(movingColor, KING_INDEX);

        if (const uint64_t RookPerspectiveMoves = RookMap::GetMoves(msbPos, fullMap);
                (RookPerspectiveMoves & KingBoard) != 0) {

            return RookPerspectiveMoves & RookMap::GetMoves(ExtractMsbPos(KingBoard), fullMap ^ figBoard);
        }

        return BishopMap::GetMoves(msbPos, fullMap) & BishopMap::GetMoves(ExtractMsbPos(KingBoard), fullMap ^ figBoard);
    }

    // returns [ pinnedFigMap, allowedTilesMap ]
    template<PinnedFigGen genType>
    [[nodiscard]] FAST_CALL thrust::pair<uint64_t, uint64_t>
    GetPinnedFigsMap(const uint32_t col, const uint64_t fullMap) const {
        ASSERT(fullMap != 0, "Full map is empty!");
        ASSERT(col == 1 || col == 0, "Invalid color!");

        const uint32_t enemyCord = SwapColor(col) * BIT_BOARDS_PER_COLOR;

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

    [[nodiscard]] FAST_DCALL_ALWAYS uint64_t GetAllowedTilesWhenCheckedByNonSliding(const uint32_t movingColor) const {
        uint64_t allowedTiles{};

        allowedTiles |= KingMap::GetSimpleFigCheckKnightsAllowedTiles<NUM_BOARDS>(movingColor, _boardFetcher);
        allowedTiles |= KingMap::GetSimpleFigCheckPawnAllowedTiles<NUM_BOARDS>(movingColor, _boardFetcher);

        return allowedTiles;
    }


    // ------------------------------
    // private methods
    // ------------------------------

private:

    FAST_DCALL thrust::pair<uint64_t, uint8_t>
    _getRookBlockedMap(uint64_t rookMap, const uint64_t fullMapWoutKing, const uint64_t kingMap) const {
        ASSERT(kingMap != 0, "King map is empty!");

        uint64_t blockedTiles{};
        uint8_t checks{};

        while (rookMap) {
            const uint32_t msbPos = ExtractMsbPos(rookMap);

            const uint64_t moves = RookMap::GetMoves(msbPos, fullMapWoutKing);
            blockedTiles |= moves;
            checks += ((moves & kingMap) != 0);

            rookMap ^= cuda_MaxMsbPossible >> msbPos;
        }

        return {blockedTiles, checks};
    }

    template<class MoveGeneratorT>
    [[nodiscard]] FAST_DCALL_ALWAYS static uint64_t _blockIterativeGenerator(uint64_t board, MoveGeneratorT mGen) {
        uint64_t blockedMap = 0;

        while (board != 0) {
            const uint32_t figPos = ExtractMsbPos(board);
            board ^= (cuda_MaxMsbPossible >> figPos);

            blockedMap |= mGen(figPos);
        }

        return blockedMap;
    }

    // returns [ pinnedFigMap, allowedTilesMap ]
    template<class MoveMapT, PinnedFigGen type>
    [[nodiscard]] FAST_DCALL thrust::pair<uint64_t, uint64_t>
    _getPinnedFigMaps(const uint32_t movingColor, const uint64_t fullMap, const uint64_t possiblePinningFigs) const {
        uint64_t allowedTilesFigMap{};
        [[maybe_unused]] uint64_t pinnedFigMap{};

        const uint32_t kingPos = _boardFetcher.GetKingMsbPos(movingColor);
        // generating figs seen from king's rook perspective
        const uint64_t kingFigPerspectiveAttackedFields = MoveMapT::GetMoves(kingPos, fullMap);
        const uint64_t kingFigPerspectiveAttackedFigs = kingFigPerspectiveAttackedFields & fullMap;

        // this functions should be called only in case of single check so the value below can only be either null or the
        // map of checking figure
        if constexpr (type == PinnedFigGen::WAllowedTiles) {
            if (const uint64_t kingSeenEnemyFigs = kingFigPerspectiveAttackedFigs & possiblePinningFigs;
                    kingSeenEnemyFigs != 0) {
                const uint32_t msbPos = ExtractMsbPos(kingSeenEnemyFigs);
                const uint64_t moves = MoveMapT::GetMoves(msbPos, fullMap);

                allowedTilesFigMap = (moves & kingFigPerspectiveAttackedFields) | kingSeenEnemyFigs;
            }
        }

        // removing figs seen by king
        const uint64_t cleanedMap = fullMap ^ kingFigPerspectiveAttackedFigs;

        // generating figs, which stayed behind first ones and are actually pinning ones
        const uint64_t kingSecondRookPerspective = MoveMapT::GetMoves(kingPos, cleanedMap);
        uint64_t pinningFigs = possiblePinningFigs & kingSecondRookPerspective;

        // generating fields which are both seen by king and pinning figure = field on which pinned figure stays
        while (pinningFigs != 0) {
            const uint32_t msbPos = ExtractMsbPos(pinningFigs);
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
