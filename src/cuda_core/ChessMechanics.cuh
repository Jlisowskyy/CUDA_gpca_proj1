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
#include "Move.cuh"

#include "BishopMap.cuh"
#include "RookMap.cuh"
#include "KnightMap.cuh"
#include "KingMap.cuh"
#include "WhitePawnMap.cuh"
#include "BlackPawnMap.cuh"

#include <cassert>
#include <cinttypes>

struct ChessMechanics {
    // ------------------------------
    // Class inner types
    // ------------------------------

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

    FAST_CALL explicit ChessMechanics(const cuda_Board &bd) : _board(bd) {}

    ChessMechanics(ChessMechanics &other) = delete;

    ChessMechanics &operator=(ChessMechanics &) = delete;

    ~ChessMechanics() = default;

    // ------------------------------
    // Class interaction
    // ------------------------------

    [[nodiscard]] __device__ bool IsCheck() const {
        const __uint32_t enemyCol = SwapColor(_board.MovingColor);
        const __uint32_t kingsMsb = _board.GetKingMsbPos(_board.MovingColor);
        const __uint64_t fullBoard = GetFullBitMap();

        // Checking rook's perspective
        const __uint64_t enemyRooks = _board.GetFigBoard(enemyCol, rooksIndex);
        const __uint64_t enemyQueens = _board.GetFigBoard(enemyCol, queensIndex);

        const __uint64_t kingsRookPerspective = RookMap::GetMoves(kingsMsb, fullBoard);

        if ((kingsRookPerspective & (enemyRooks | enemyQueens)) != 0)
            return true;

        // Checking bishop's perspective
        const __uint64_t enemyBishops = _board.GetFigBoard(enemyCol, bishopsIndex);

        const __uint64_t kingsBishopPerspective = BishopMap::GetMoves(kingsMsb, fullBoard);

        if ((kingsBishopPerspective & (enemyBishops | enemyQueens)) != 0)
            return true;

        // checking knights attacks
        const __uint64_t enemyKnights = _board.GetFigBoard(enemyCol, knightsIndex);

        const __uint64_t knightsPerspective = KnightMap::GetMoves(kingsMsb);

        if ((knightsPerspective & (enemyKnights)) != 0)
            return true;

        // pawns checks
        const __uint64_t enemyPawns = _board.GetFigBoard(enemyCol, pawnsIndex);
        const __uint64_t pawnAttacks =
                enemyCol == WHITE ? WhitePawnMap::GetAttackFields(enemyPawns) : BlackPawnMap::GetAttackFields(
                        enemyPawns);

        if ((pawnAttacks & (cuda_MaxMsbPossible >> kingsMsb)) != 0)
            return true;

        return false;
    }


    // Gets occupancy maps, which simply indicates whether some field is occupied or not. Does not distinguish colors.
    [[nodiscard]] FAST_CALL __uint64_t GetFullBitMap() const {
        __uint64_t map = 0;
        for (size_t i = 0; i < BitBoardsCount; ++i) map |= _board.BitBoards[i];
        return map;
    }

    // Gets occupancy maps, which simply indicates whether some field is occupied or not, by desired color figures.
    [[nodiscard]] FAST_CALL __uint64_t GetColBitMap(const __uint32_t col) const {
        assert(col == 1 || col == 0);

        __uint64_t map = 0;
        for (size_t i = 0; i < BitBoardsPerCol; ++i) map |= _board.BitBoards[BitBoardsPerCol * col + i];
        return map;
    }

    // does not check kings BitBoards!!!
    [[nodiscard]] FAST_CALL size_t GetIndexOfContainingBitBoard(const __uint64_t map, const __uint32_t col) const {
        const size_t colIndex = col * BitBoardsPerCol;
        size_t rv = 0;
        for (size_t i = 0; i < BitBoardsPerCol; ++i) {
            rv += ((_board.BitBoards[colIndex + i] & map) != 0) * i;
        }
        return colIndex + rv;
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
    GetBlockedFieldBitMap(__uint64_t fullMap) const {
        assert(fullMap != 0 && "Full map is empty!");

        __uint8_t checksCount{};
        __uint64_t blockedMap{};
        bool wasCheckedBySimple{};

        const size_t enemyFigInd = SwapColor(_board.MovingColor) * BitBoardsPerCol;
        const __uint32_t allyKingShift = ConvertToReversedPos(_board.GetKingMsbPos(_board.MovingColor));
        const __uint64_t allyKingMap = cuda_MinMsbPossible << allyKingShift;

        // King attacks generation.
        blockedMap |= KingMap::GetMoves(_board.GetKingMsbPos(SwapColor(_board.MovingColor)));

        // Rook attacks generation. Needs special treatment to correctly detect double check, especially with pawn promotion
        const auto [rookBlockedMap, checkCountRook] = _getRookBlockedMap(
                _board.BitBoards[enemyFigInd + rooksIndex] | _board.BitBoards[enemyFigInd + queensIndex],
                fullMap ^ allyKingMap,
                allyKingMap
        );

        // = 0, 1 or eventually 2 when promotion to rook like type happens
        checksCount += checkCountRook;
        blockedMap |= rookBlockedMap;

        // Bishop attacks generation.
        const __uint64_t bishopBlockedMap = _blockIterativeGenerator(
                _board.BitBoards[enemyFigInd + bishopsIndex] | _board.BitBoards[enemyFigInd + queensIndex],
                [=](const int pos) {
                    return BishopMap::GetMoves(pos, fullMap ^ allyKingMap);
                }
        );

        // = 1 or 0 depending on whether hits or not
        const __uint8_t wasCheckedByBishopFlag = (bishopBlockedMap & allyKingMap) >> allyKingShift;

        checksCount += wasCheckedByBishopFlag;
        blockedMap |= bishopBlockedMap;

        // Pawns attacks generation.
        const __uint64_t pawnsMap = _board.BitBoards[enemyFigInd + pawnsIndex];
        const __uint64_t pawnBlockedMap =
                SwapColor(_board.MovingColor) == WHITE ? WhitePawnMap::GetAttackFields(pawnsMap)
                                                       : BlackPawnMap::GetAttackFields(pawnsMap);

        const bool wasCheckedByPawnFlag = pawnBlockedMap & allyKingMap;

        checksCount += wasCheckedByPawnFlag;
        wasCheckedBySimple |= wasCheckedByPawnFlag;
        blockedMap |= pawnBlockedMap;

        // Knight attacks generation.
        const __uint64_t knightBlockedMap = _blockIterativeGenerator(
                _board.BitBoards[enemyFigInd + knightsIndex],
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

    [[nodiscard]] FAST_DCALL __uint64_t
    GenerateAllowedTilesForPrecisedPinnedFig(__uint64_t figBoard, __uint64_t fullMap) const {
        assert(fullMap != 0 && "Full map is empty!");
        assert(CountOnesInBoard(figBoard) == 1 && "Only one figure should be pinned!");

        const int msbPos = ExtractMsbPos(figBoard);
        const __uint64_t KingBoard = _board.BitBoards[_board.MovingColor * BitBoardsPerCol + kingIndex];

        const __uint64_t RookPerspectiveMoves = RookMap::GetMoves(msbPos, fullMap);
        if ((RookPerspectiveMoves & KingBoard) != 0)
            return RookPerspectiveMoves & RookMap::GetMoves(ExtractMsbPos(KingBoard), fullMap ^ figBoard);

        const __uint64_t BishopPerspectiveMoves = BishopMap::GetMoves(msbPos, fullMap);
        return BishopPerspectiveMoves & BishopMap::GetMoves(ExtractMsbPos(KingBoard), fullMap ^ figBoard);
    }

    // returns [ pinnedFigMap, allowedTilesMap ]
    template<PinnedFigGen genType>
    [[nodiscard]] FAST_CALL thrust::pair<__uint64_t, __uint64_t> GetPinnedFigsMap(__uint32_t col, __uint64_t fullMap) const {
        assert(fullMap != 0 && "Full map is empty!");
        assert(col == 1 || col == 0 && "Invalid color!");

        const size_t enemyCord = SwapColor(col) * BitBoardsPerCol;

        const auto [pinnedByRooks, allowedRooks] = _getPinnedFigMaps<RookMap, genType>(
                fullMap, _board.BitBoards[enemyCord + rooksIndex] | _board.BitBoards[enemyCord + queensIndex]
        );

        const auto [pinnedByBishops, allowedBishops] = _getPinnedFigMaps<BishopMap, genType>(
                fullMap, _board.BitBoards[enemyCord + bishopsIndex] | _board.BitBoards[enemyCord + queensIndex]
        );

        return {pinnedByBishops | pinnedByRooks, allowedBishops | allowedRooks};
    }

    [[nodiscard]] FAST_CALL __uint64_t GetAllowedTilesWhenCheckedByNonSliding() const {
        __uint64_t allowedTiles{};

        allowedTiles |= KingMap::GetSimpleFigCheckKnightsAllowedTiles(_board);
        allowedTiles |= KingMap::GetSimpleFigCheckPawnAllowedTiles(_board);

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

    FAST_DCALL static thrust::pair<__uint64_t, __uint8_t>
    _getRookBlockedMap(__uint64_t rookMap, __uint64_t fullMapWoutKing, __uint64_t kingMap) {
        assert(kingMap != 0 && "King map is empty!");

        __uint64_t blockedTiles{};
        __uint8_t checks{};

        while (rookMap) {
            const int msbPos = ExtractMsbPos(rookMap);

            const __uint64_t moves = RookMap::GetMoves(msbPos, fullMapWoutKing);
            blockedTiles |= moves;
            checks += ((moves & kingMap) != 0);

            rookMap ^= cuda_MaxMsbPossible >> msbPos;
        }

        return {blockedTiles, checks};
    }

    template<class MoveGeneratorT>
    [[nodiscard]] FAST_DCALL static __uint64_t _blockIterativeGenerator(__uint64_t board, MoveGeneratorT mGen) {
        __uint64_t blockedMap = 0;

        while (board != 0) {
            const int figPos = ExtractMsbPos(board);
            board ^= (cuda_MaxMsbPossible >> figPos);

            blockedMap |= mGen(figPos);
        }

        return blockedMap;
    }

    // returns [ pinnedFigMap, allowedTilesMap ]
    template<class MoveMapT, PinnedFigGen type>
    [[nodiscard]] FAST_DCALL thrust::pair<__uint64_t, __uint64_t>
    _getPinnedFigMaps(__uint64_t fullMap, __uint64_t possiblePinningFigs) const {
        __uint64_t allowedTilesFigMap{};
        [[maybe_unused]] __uint64_t pinnedFigMap{};

        const __uint32_t kingPos = _board.GetKingMsbPos(_board.MovingColor);
        // generating figs seen from king's rook perspective
        const __uint64_t kingFigPerspectiveAttackedFields = MoveMapT::GetMoves(kingPos, fullMap);
        const __uint64_t kingFigPerspectiveAttackedFigs = kingFigPerspectiveAttackedFields & fullMap;

        // this functions should be called only in case of single check so the value below can only be either null or the
        // map of checking figure
        if constexpr (type == PinnedFigGen::WAllowedTiles)
            if (const __uint64_t kingSeenEnemyFigs = kingFigPerspectiveAttackedFigs & possiblePinningFigs;
                    kingSeenEnemyFigs != 0) {
                const int msbPos = ExtractMsbPos(kingSeenEnemyFigs);
                const __uint64_t moves = MoveMapT::GetMoves(msbPos, fullMap);

                allowedTilesFigMap = (moves & kingFigPerspectiveAttackedFields) | kingSeenEnemyFigs;
            }

        // removing figs seen by king
        const __uint64_t cleanedMap = fullMap ^ kingFigPerspectiveAttackedFigs;

        // generating figs, which stayed behind first ones and are actually pinning ones
        const __uint64_t kingSecondRookPerspective = MoveMapT::GetMoves(kingPos, cleanedMap);
        __uint64_t pinningFigs = possiblePinningFigs & kingSecondRookPerspective;

        // generating fields which are both seen by king and pinning figure = field on which pinned figure stays
        while (pinningFigs != 0) {
            const int msbPos = ExtractMsbPos(pinningFigs);
            pinnedFigMap |= MoveMapT::GetMoves(msbPos, fullMap) & kingFigPerspectiveAttackedFigs;
            pinningFigs ^= cuda_MaxMsbPossible >> msbPos;
        }

        return {pinnedFigMap, allowedTilesFigMap};
    }

    // ------------------------------
    // Class fields
    // ------------------------------

protected:
    const cuda_Board &_board;
};


#endif // CHESSMECHANICS_CUH
